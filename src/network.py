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

        lang_set = {'de', 'en', 'es'}
        self.chars = dict()
        self.evocab = dict()
        self.clookup = dict()
        self.char_lstm = dict()
        self.proj_mat = dict()
        external_embedding = dict()
        word_index = 2
        for f in os.listdir(options.external_embedding):
            lang = f[:-3]
            if not lang in lang_set:
                continue
            efp = gzip.open(options.external_embedding+'/'+f, 'r')
            external_embedding[lang] = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]]
                                        for line in efp if len(line.split(' ')) > 2}
            efp.close()
            self.evocab[lang] = {word: i  + word_index for i, word in enumerate(external_embedding[lang])}
            word_index += len(self.evocab[lang])

            if len(external_embedding[lang]) > 0:
                edim = len(external_embedding[lang].values()[0])
            self.chars[lang] = {c: i + 2 for i, c in enumerate(chars)}

            print 'Loaded vector', edim, 'and', len(external_embedding[lang]), 'for', lang
            self.clookup[lang] = self.model.add_lookup_parameters((len(chars) + 2, options.ce))
            self.char_lstm[lang] = dy.BiRNNBuilder(1, options.ce, edim, self.model, dy.VanillaLSTMBuilder)
            self.proj_mat[lang] = self.model.add_parameters((edim + options.pe, edim + options.pe))

        self.elookup = self.model.add_lookup_parameters((word_index, edim))
        self.num_all_words = word_index
        self.elookup.set_updated(False)
        self.elookup.init_row(0, [0] * edim)
        self.elookup.init_row(1, [0] * edim)
        for lang in self.evocab.keys():
            for word in external_embedding[lang].keys():
                self.elookup.init_row(self.evocab[lang][word], external_embedding[lang][word])

        input_dim = edim + options.pe if self.options.use_pos else edim
        self.deep_lstms = dy.BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model, dy.VanillaLSTMBuilder)
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

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
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
        words, pos_tags, chars, langs, signs, masks = batch

        all_inputs = [0]*len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [dy.lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd  = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(len(words[lang])):
                cnn_reps[i] = dy.pick_batch(crnns, [j * words[lang].shape[0] + i for j in range(words[lang].shape[1])], 1)
            wembed = [dy.lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [dy.lookup_batch(self.plookup, pos_tags[lang][i]) for i in range(len(pos_tags[lang]))] if self.options.use_pos else None

            if not train:
                inputs = [dy.concatenate([w, pos]) for w, pos in zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in zip(wembed, posembed, emb_masks)] if self.options.use_pos\
                    else [dy.cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            all_inputs[l] = inputs

        lstm_input = [dy.concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in range(len(all_inputs[0]))]
        d = self.options.dropout
        return self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)


    def norms(self, v):
        norms = []
        for i in range(v.dim()[0][0]):
            norms.append(dy.squared_norm(v[i]))
        return dy.concatenate(norms)

    def train(self, mini_batch):
        words, pos_tags, chars, langs, signs, masks = mini_batch
        # Getting the last hidden layer from BiLSTM.
        h_out = self.rnn_mlp(mini_batch, True)[-1]
        t_out_d = dy.reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1]))
        t_out = dy.transpose(t_out_d)

        # Calculating the kq values for NCE.
        k = float(t_out.dim()[0][0] - len(chars))
        kq = dy.scalarInput(k / self.num_all_words)
        lkq = dy.log(kq)

        '''
        # Getting outer product (all possible permutations)
        products = dy.reshape(t_out * t_out_d, (len(langs) * len(langs),))

        # Normalize products by their l2-norms.
        #normalized_products = dy.cmult(products, norm_prod_inv_tensor)
        normalized_products = self.cut_value(products)

        # Getting u(x,\theta).
        exp_prods = dy.exp(normalized_products)

        # Masks for useless parts.
        final_mask = [0]*len(langs)*len(langs)
        final_signs = [0]*len(langs)*len(langs)
        index, num_elems = 0, 0
        for i in range(len(langs)):
            for j in range(len(langs)):
                if j>i and (langs[i] != langs[j]) and (signs[i] == 1 or signs[j]==1):
                    final_mask[index] = 1
                    num_elems += 1
                    if signs[i] == signs[j]:
                        final_signs[index] = 1
                    else:
                        final_signs[index] = -1
                index += 1
        final_mask = dy.inputVector(final_mask)

        # NCE nominator and denominator.
        ls_vec = dy.log(exp_prods + kq)
        other_vec =  dy.concatenate([normalized_products[i] if final_signs[i]==1 else lkq for i in range(len(final_signs))])

        # NCE loss.
        loss_vec = dy.cmult(-other_vec + ls_vec, final_mask)

        # Loss calculation.
        err = dy.sum_elems(loss_vec) / num_elems
        '''

        loss_values = []
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                if (langs[i] != langs[j]) and (signs[i] == 1 or signs[j] == 1):
                    # lu = dot_product(t_out[i], t_out[j]) / (norm_vals[i]*norm_vals[j])
                    lu = -dy.squared_distance(t_out[i], t_out[j])
                    ls = -dy.log(dy.exp(lu) + kq)
                    if signs[i] == signs[j]:  # both one
                        ls += lu
                    else:
                        ls += lkq
                    loss_values.append(-ls)
        err = dy.esum(loss_values) / len(loss_values)
        err.forward()
        err_value = err.value()
        print 'err_value', err_value
        err.backward()
        self.trainer.update()
        dy.renew_cg(immediate_compute = True, check_validity = True) #todo
        return err_value

    def eval(self, mini_batch):
        words, pos_tags, chars, langs, signs, masks = mini_batch
        h_out = self.rnn_mlp(mini_batch, False)[-1]
        t_out = dy.transpose(dy.reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1])))

        sims = []
        for i in range(len(langs)):
            for j in range(i+1, len(langs)):
                sims.append(dy.squared_distance(t_out[i], t_out[j]))
                #sims.append(dot_product(t_out[i], t_out[j])/(norm_vals[i]*norm_vals[j]))
        sim = dy.esum(sims)
        sim.forward()
        sim_value = sim.value() / len(sims)
        dy.renew_cg()
        return sim_value