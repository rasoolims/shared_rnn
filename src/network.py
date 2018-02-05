import dynet as dy
import os, sys, math, random, gzip
from linalg import *

reload(sys)
sys.setdefaultencoding('utf8')


class Network:
    def __init__(self, pos, chars, options):
        self.model = dy.Model()
        self.UNK = 0
        self.PAD = 1
        self.options = options
        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.dropout = False if options.dropout == 0.0 else True
        self.pos = {word: ind + 2 for ind, word in enumerate(pos)}
        self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))
        edim = options.we
        self.cut_value = lambda x: dy.bmin(0.0001 * x, x)
        self.activations = {'tanh': dy.tanh, 'sigmoid': dy.logistic, 'relu': dy.rectify, 'leaky': (lambda x: dy.bmax(.1 * x, x))}
        self.activation = self.activations['leaky']

        self.chars = dict()
        self.evocab = dict()
        self.clookup = dict()
        self.char_lstm = dict()
        self.proj_mat = dict()
        external_embedding = dict()
        word_index = 2
        input_dim = edim + options.pe if self.options.use_pos else edim
        for f in os.listdir(options.external_embedding):
            lang = f[:-3]
            efp = gzip.open(options.external_embedding + '/' + f, 'r')
            external_embedding[lang] = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]]
                                        for line in efp if len(line.split(' ')) > 2}
            efp.close()
            self.evocab[lang] = {word: i + word_index for i, word in enumerate(external_embedding[lang])}
            word_index += len(self.evocab[lang])

            if len(external_embedding[lang]) > 0:
                edim = len(external_embedding[lang].values()[0])
            self.chars[lang] = {c: i + 2 for i, c in enumerate(chars[lang])}

            print 'Loaded vector', edim, 'and', len(external_embedding[lang]), 'for', lang
            self.clookup[lang] = self.model.add_lookup_parameters((len(chars[lang]) + 2, options.ce))
            self.char_lstm[lang] = dy.BiRNNBuilder(1, options.ce, edim, self.model, dy.VanillaLSTMBuilder)
            self.proj_mat[lang] = self.model.add_parameters((input_dim, input_dim))

        self.elookup = self.model.add_lookup_parameters((word_index, edim))
        self.num_all_words = word_index
        self.elookup.set_updated(False)
        self.elookup.init_row(0, [0] * edim)
        self.elookup.init_row(1, [0] * edim)
        for lang in self.evocab.keys():
            for word in external_embedding[lang].keys():
                self.elookup.init_row(self.evocab[lang][word], external_embedding[lang][word])

        self.lang2id = {lang: i for i, lang in enumerate(self.evocab.keys())}
        self.lang_lookup = self.model.add_lookup_parameters((len(self.lang2id), options.le))

        self.deep_lstms = dy.BiRNNBuilder(options.layer, input_dim + options.le, options.rnn * 2, self.model,
                                          dy.VanillaLSTMBuilder)
        for i in range(len(self.deep_lstms.builder_layers)):
            builder = self.deep_lstms.builder_layers[i]
            b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
            b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
            self.deep_lstms.builder_layers[i] = (b0, b1)

        w_mlp_arc = orthonormal_initializer(options.arc_mlp, options.rnn * 2)
        w_mlp_label = orthonormal_initializer(options.label_mlp, options.rnn * 2)
        self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.rnn * 2),
                                                      init=dy.NumpyInitializer(w_mlp_arc))
        self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init=dy.ConstInitializer(0))
        self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.rnn * 2),
                                                        init=dy.NumpyInitializer(w_mlp_label))
        self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init=dy.ConstInitializer(0))
        self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.rnn * 2),
                                                     init=dy.NumpyInitializer(w_mlp_arc))
        self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init=dy.ConstInitializer(0))
        self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.rnn * 2),
                                                       init=dy.NumpyInitializer(w_mlp_label))
        self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init=dy.ConstInitializer(0))

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                if self.options.use_pos:
                    tag_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                    scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                    word_mask *= scale
                    tag_mask *= scale
                    word_mask = dy.inputTensor(word_mask, batched=True)
                    tag_mask = dy.inputTensor(tag_mask, batched=True)
                    ret.append((word_mask, tag_mask))
                else:
                    scale = 2. / (2. * word_mask + 1e-12)
                    word_mask *= scale
                    word_mask = dy.inputTensor(word_mask, batched=True)
                    ret.append(word_mask)
            return ret

        self.generate_emb_mask = _emb_mask_generator

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

    def bi_rnn(self, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fb.set_dropouts(dropout_x, dropout_h)
            bb.set_dropouts(dropout_x, dropout_h)
            if batch_size is not None:
                fb.set_dropout_masks(batch_size)
                bb.set_dropout_masks(batch_size)
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs

    def rnn_mlp(self, batch, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = batch

        all_inputs = [0] * len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [dy.lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(words[lang].shape[0]):
                cnn_reps[i] = dy.pick_batch(crnns, char_batches[lang][i], 1)
            wembed = [dy.lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [dy.lookup_batch(self.plookup, pos_tags[lang][i]) for i in
                        range(len(pos_tags[lang]))] if self.options.use_pos else None
            lang_embeds = [dy.lookup_batch(self.lang_lookup, [self.lang2id[lang]] * len(pos_tags[lang][i])) for i in
                           range(len(pos_tags[lang]))]

            if not train:
                inputs = [dy.concatenate([w, pos]) for w, pos in
                          zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                          zip(wembed, posembed, emb_masks)] if self.options.use_pos \
                    else [dy.cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            inputs = [dy.concatenate([inp, lembed]) for inp, lembed in zip(inputs, lang_embeds)]
            all_inputs[l] = inputs

        lstm_input = [dy.concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in
                      range(len(all_inputs[0]))]
        d = self.options.dropout
        h_out = self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)
        h = dy.dropout_dim(dy.concatenate_cols(h_out), 1, d) if train else dy.concatenate_cols(h_out)
        H = self.activation(dy.affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(dy.affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(dy.affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(dy.affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

        if train:
            H, M, HL, ML = dy.dropout_dim(H, 1, d), dy.dropout_dim(M, 1, d), dy.dropout_dim(HL, 1, d), dy.dropout_dim(ML, 1, d)
        return H, M, HL, ML

    def train(self, mini_batch):
        pwords, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = mini_batch
        # Getting the last hidden layer from BiLSTM.
        H, M, HL, ML = self.rnn_mlp(mini_batch, True)
        dim_0, dim_1, dim_2 = H.dim()[0][1], H.dim()[0][0], H.dim()[1]
        ldim_0, ldim_1, ldim_2 = HL.dim()[0][1], HL.dim()[0][0], HL.dim()[1]
        H_i = [dy.transpose(dy.reshape(dy.pick(H, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        M_i = [dy.transpose(dy.reshape(dy.pick(M, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        HL_i = [dy.transpose(dy.reshape(dy.pick(HL, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        ML_i = [dy.transpose(dy.reshape(dy.pick(ML, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        # Calculating the kq values for NCE.
        loss_values = []
        last_pos = H.dim()[0][1] - 1

        for b in batch_num:
            for i in range(len(batch_num[b])):
                lang1 = langs[b][i]
                pos1 = positions[b][i]
                b1 = batch_num[b][i]
                HVec1 = H_i[pos1][b1]
                MVec1 = M_i[pos1][b1]
                HLVec1 = HL_i[pos1][b1]
                MLVec1 = ML_i[pos1][b1]

                for j in range(i + 1, len(batch_num[b])):
                    lang2 = langs[b][j]
                    pos2 = positions[b][j]
                    b2 = batch_num[b][j]
                    if lang1 != lang2:
                        HVec2 = H_i[pos2][b2]
                        MVec2 = M_i[pos2][b2]
                        HLVec2 = HL_i[pos2][b2]
                        MLVec2 = ML_i[pos2][b2]

                        ps_loss = -dy.sqrt(dy.squared_distance(HVec1, HVec2))
                        term = -dy.log(dy.logistic(ps_loss))
                        loss_values.append(term)

                        ps_loss = -dy.sqrt(dy.squared_distance(MVec1, MVec2))
                        term = -dy.log(dy.logistic(ps_loss))
                        loss_values.append(term)

                        ps_loss = -dy.sqrt(dy.squared_distance(HLVec1, HLVec2))
                        term = -dy.log(dy.logistic(ps_loss))
                        loss_values.append(term)

                        ps_loss = -dy.sqrt(dy.squared_distance(MLVec1, MLVec2))
                        term = -dy.log(dy.logistic(ps_loss))
                        loss_values.append(term)

                        # alignment-based negative position.
                        for _ in range(5):
                            s_neg_position , t_neg_position = random.randint(0, last_pos), random.randint(0, last_pos)
                            if s_neg_position != pos1:
                                s_vec = H_i[s_neg_position][b1]
                                d_s = dy.sqrt(dy.squared_distance(s_vec, HVec2))
                                term = -dy.log(dy.logistic(-d_s))
                                loss_values.append(term)

                                s_vec = M_i[s_neg_position][b1]
                                d_s = dy.sqrt(dy.squared_distance(s_vec, MVec2))
                                term = -dy.log(dy.logistic(-d_s))
                                loss_values.append(term)

                                s_vec = HL_i[s_neg_position][b1]
                                d_s = dy.sqrt(dy.squared_distance(s_vec, HLVec2))
                                term = -dy.log(dy.logistic(-d_s))
                                loss_values.append(term)

                                s_vec = ML_i[s_neg_position][b1]
                                d_s = dy.sqrt(dy.squared_distance(s_vec, MLVec2))
                                term = -dy.log(dy.logistic(-d_s))
                                loss_values.append(term)
                            if t_neg_position != pos2:
                                t_vec = H_i[t_neg_position][b2]
                                d_t = dy.sqrt(dy.squared_distance(HVec1, t_vec))
                                term = -dy.log(dy.logistic(-d_t))
                                loss_values.append(term)

                                t_vec = M_i[t_neg_position][b2]
                                d_t = dy.sqrt(dy.squared_distance(MVec1, t_vec))
                                term = -dy.log(dy.logistic(-d_t))
                                loss_values.append(term)

                                t_vec = HL_i[t_neg_position][b2]
                                d_t = dy.sqrt(dy.squared_distance(HLVec1, t_vec))
                                term = -dy.log(dy.logistic(-d_t))
                                loss_values.append(term)

                                t_vec = ML_i[t_neg_position][b2]
                                d_t = dy.sqrt(dy.squared_distance(MLVec1, t_vec))
                                term = -dy.log(dy.logistic(-d_t))
                                loss_values.append(term)

        err_value = 0
        if len(loss_values) > 0:
            err = dy.esum(loss_values) / len(loss_values)
            err.forward()
            err_value = err.value()
            err.backward()
            self.trainer.update()
        dy.renew_cg()
        return err_value

    def eval(self, mini_batch):
        pwords, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = mini_batch
        # Getting the last hidden layer from BiLSTM.
        H, M, HL, ML = self.rnn_mlp(mini_batch, False)
        dim_0, dim_1, dim_2 = H.dim()[0][1], H.dim()[0][0], H.dim()[1]
        ldim_0, ldim_1, ldim_2 = HL.dim()[0][1], HL.dim()[0][0], HL.dim()[1]
        H_i = [dy.transpose(dy.reshape(dy.pick(H, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        M_i = [dy.transpose(dy.reshape(dy.pick(M, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        HL_i = [dy.transpose(dy.reshape(dy.pick(HL, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        ML_i = [dy.transpose(dy.reshape(dy.pick(ML, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        positive_loss, negative_loss, lm_loss = [], [], []
        last_pos = H.dim()[0][1] - 1

        for b in batch_num:
            for i in range(len(batch_num[b])):
                lang1 = langs[b][i]
                pos1 = positions[b][i]
                b1 = batch_num[b][i]
                HVec1 = H_i[pos1][b1]
                MVec1 = M_i[pos1][b1]
                HLVec1 = HL_i[pos1][b1]
                MLVec1 = ML_i[pos1][b1]

                for j in range(i + 1, len(batch_num[b])):
                    lang2 = langs[b][j]
                    pos2 = positions[b][j]
                    b2 = batch_num[b][j]
                    if lang1 != lang2:
                        HVec2 = H_i[pos2][b2]
                        MVec2 = M_i[pos2][b2]
                        HLVec2 = HL_i[pos2][b2]
                        MLVec2 = ML_i[pos2][b2]

                        ps_loss = dy.sqrt(dy.squared_distance(HVec1, HVec2))
                        positive_loss.append(ps_loss)

                        ps_loss = dy.sqrt(dy.squared_distance(MVec1, MVec2))
                        positive_loss.append(ps_loss)

                        ps_loss = dy.sqrt(dy.squared_distance(HLVec1, HLVec2))
                        positive_loss.append(ps_loss)

                        ps_loss = dy.sqrt(dy.squared_distance(MLVec1, MLVec2))
                        positive_loss.append(ps_loss)

                        s_neg_position, t_neg_position = random.randint(0, last_pos), random.randint(0, last_pos)
                        if s_neg_position != pos1:
                            s_vec = H_i[s_neg_position][b1]
                            d_s = dy.sqrt(dy.squared_distance(s_vec, HVec2))
                            negative_loss.append(d_s)

                            s_vec = M_i[s_neg_position][b1]
                            d_s = dy.sqrt(dy.squared_distance(s_vec, MVec2))
                            negative_loss.append(d_s)

                            s_vec = HL_i[s_neg_position][b1]
                            d_s = dy.sqrt(dy.squared_distance(s_vec, HLVec2))
                            negative_loss.append(d_s)

                            s_vec = ML_i[s_neg_position][b1]
                            d_s = dy.sqrt(dy.squared_distance(s_vec, MLVec2))
                            negative_loss.append(d_s)
                        if t_neg_position != pos2:
                            t_vec = H_i[t_neg_position][b2]
                            d_t = dy.sqrt(dy.squared_distance(HVec1, t_vec))
                            negative_loss.append(d_t)

                            t_vec = M_i[t_neg_position][b2]
                            d_t = dy.sqrt(dy.squared_distance(MVec1, t_vec))
                            negative_loss.append(d_t)

                            t_vec = HL_i[t_neg_position][b2]
                            d_t = dy.sqrt(dy.squared_distance(HLVec1, t_vec))
                            negative_loss.append(d_t)

                            t_vec = ML_i[t_neg_position][b2]
                            d_t = dy.sqrt(dy.squared_distance(MLVec1, t_vec))
                            negative_loss.append(d_t)

        pl, nl = dy.esum(positive_loss).value() / len(positive_loss), dy.esum(negative_loss).value() / len(negative_loss)
        dy.renew_cg()
        return pl, nl
