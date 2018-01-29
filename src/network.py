import dynet as dy
import gzip
import os,sys,math
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
            efp = gzip.open(options.external_embedding+'/'+f, 'r')
            external_embedding[lang] = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]]
                                        for line in efp if len(line.split(' ')) > 2}
            efp.close()
            self.evocab[lang] = {word: i  + word_index for i, word in enumerate(external_embedding[lang])}
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

        self.lang2id = {lang:i for i,lang in enumerate(self.evocab.keys())}
        self.lang_lookup = self.model.add_lookup_parameters((len(self.lang2id), options.le))

        self.deep_lstms = dy.BiRNNBuilder(options.layer, input_dim +  options.le, options.rnn * 2, self.model, dy.VanillaLSTMBuilder)
        for i in range(len(self.deep_lstms.builder_layers)):
            builder = self.deep_lstms.builder_layers[i]
            b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
            b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
            self.deep_lstms.builder_layers[i] = (b0, b1)

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

        all_inputs = [0]*len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [dy.lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd  = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(words[lang].shape[0]):
                cnn_reps[i] = dy.pick_batch(crnns, [char_batches[lang][j][i] for j in range(words[lang].shape[1])], 1)
            wembed = [dy.lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [dy.lookup_batch(self.plookup, pos_tags[lang][i]) for i in range(len(pos_tags[lang]))] if self.options.use_pos else None
            lang_embeds = [dy.lookup_batch(self.lang_lookup, [self.lang2id[lang]]*len(pos_tags[lang][i])) for i in range(len(pos_tags[lang]))]

            if not train:
                inputs = [dy.concatenate([w, pos]) for w, pos in zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in zip(wembed, posembed, emb_masks)] if self.options.use_pos\
                    else [dy.cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            inputs = [dy.concatenate([inp, lembed]) for inp, lembed in zip(inputs, lang_embeds)]
            all_inputs[l] = inputs

        lstm_input = [dy.concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in range(len(all_inputs[0]))]
        d = self.options.dropout
        return self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)

    def train(self, mini_batch, num_train, k):
        pwords, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = mini_batch
        # Getting the last hidden layer from BiLSTM.
        rnn_out = self.rnn_mlp(mini_batch, True)
        t_out_ds = [dy.reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1])) for h_out in rnn_out]
        t_outs = [dy.transpose(t_out_d) for t_out_d in t_out_ds]

        # Calculating the kq values for NCE.
        kq = dy.scalarInput(float(k) / num_train)
        lkq = dy.log(kq)
        loss_values = []

        for b in batch_num:
            for i in range(len(batch_num[b])):
                if signs[b][i] == 0: continue
                lang1 = langs[b][i]
                pos1 = positions[b][i]
                b1 = batch_num[b][i]
                vec1 = t_outs[pos1][b1]
                for j in range(i+1, len(batch_num[b])):
                    lang2 = langs[b][j]
                    pos2 = positions[b][j]
                    b2 = batch_num[b][j]
                    if lang1 != lang2:
                        vec2 = t_outs[pos2][b2]
                        lu = -dy.squared_distance(vec1, vec2)
                        denom = dy.log(dy.exp(lu) + kq)
                        if signs[b][i] == signs[b][j]:  # both one
                            nom = lu
                        else:
                            nom = lkq
                        loss_values.append(denom - nom)

        err_value = 0
        if len(loss_values)>0:
            err = dy.esum(loss_values) / len(loss_values)
            err.forward()
            err_value = err.value()
            err.backward()
            self.trainer.update()
        dy.renew_cg()
        return err_value

    def eval(self, mini_batch):
        words, pos_tags, chars, langs, signs, masks = mini_batch
        h_out = self.rnn_mlp(mini_batch, False)[-1]
        t_out = dy.transpose(dy.reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1])))

        sims = []
        for i in range(len(langs)):
            for j in range(i+1, len(langs)):
                sims.append(dy.sqrt(dy.squared_distance(t_out[i], t_out[j])))
        sim = dy.esum(sims)
        sim.forward()
        sim_value = sim.value() / len(sims)
        dy.renew_cg()
        return sim_value