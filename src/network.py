from dynet import *
from operator import itemgetter
import utils, time, random, gzip
import numpy as np
import codecs, os,sys
from linalg import *
reload(sys)
sys.setdefaultencoding('utf8')

class Network:
    def __init__(self, pos, options):
        self.model = Model()
        self.UNK = 0
        self.PAD = 1
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.dropout = False if options.dropout == 0.0 else True
        self.pos = {word: ind + 2 for ind, word in enumerate(pos)}
        self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))
        edim = options.we

        lang_set = {'de', 'en', 'es', 'fr'}
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
            ch = set()
            for word in external_embedding[lang].keys():
                for c in word:
                    ch.add(c.lower())
            ch = list(sorted(ch))
            self.chars[lang] = {c: i + 2 for i, c in enumerate(ch)}


            print 'Loaded vector', edim, 'and', len(external_embedding[lang]), 'for', lang
            self.clookup[lang] = self.model.add_lookup_parameters((len(ch) + 2, options.ce))
            self.char_lstm[lang] = BiRNNBuilder(1, options.ce, edim, self.model, VanillaLSTMBuilder)
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
        self.deep_lstms = BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model, VanillaLSTMBuilder)
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
                    word_mask = inputTensor(word_mask, batched=True)
                    tag_mask = inputTensor(tag_mask, batched=True)
                    ret.append((word_mask, tag_mask))
                else:
                    scale = 2. / (2. * word_mask + 1e-12)
                    word_mask *= scale
                    word_mask = inputTensor(word_mask, batched=True)
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
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs


    def rnn_mlp(self, batch, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words, pos_tags, chars, langs, signs, masks = batch

        all_inputs = [0]*len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd  = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = reshape(concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(len(words[lang])):
                cnn_reps[i] = pick_batch(crnns, [j * words[lang].shape[0] + i for j in range(words[lang].shape[1])], 1)
            wembed = [lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [lookup_batch(self.plookup, pos_tags[lang][i]) for i in range(len(pos_tags[lang]))] if self.options.use_pos else None

            if not train:
                inputs = [concatenate([w, pos]) for w, pos in zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [concatenate([cmult(w, wm), cmult(pos, posm)]) for w, pos, (wm, posm) in zip(wembed, posembed, emb_masks)] if self.options.use_pos\
                    else [cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            all_inputs[l] = inputs

        lstm_input = [concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in range(len(all_inputs[0]))]
        d = self.options.dropout
        return self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)


    def norms(self, v):
        norms = []
        for i in range(v.dim()[0][0]):
            norms.append(squared_norm(v[i]))
        return concatenate(norms)

    def train(self, mini_batch):
        words, pos_tags, chars, langs, signs, masks = mini_batch
        h_out = self.rnn_mlp(mini_batch, True)[-1]
        t_out = transpose(reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1])))
        norm_vals = self.norms(t_out).value()

        k = float(t_out.dim()[0][0] - len(chars))
        kq = scalarInput(k/self.num_all_words)
        lkq = log(kq)
        loss_values = []
        for i in range(len(langs)):
            for j in range(i+1, len(langs)):
                if (langs[i] != langs[j]) and (signs[i] == 1 or signs[j]==1):
                    lu = dot_product(t_out[i], t_out[j]) / (norm_vals[i]*norm_vals[j])
                    ls = -log(exp(lu) + kq)
                    if signs[i] == signs[j]: # both one
                        ls += lu
                    else:
                        ls += lkq
                    loss_values.append(-ls)
        err = esum(loss_values)
        err.forward()
        err_value = err.value() / len(loss_values)
        err.backward()
        self.trainer.update()
        renew_cg()
        return err_value

    def eval(self, mini_batch):
        words, pos_tags, chars, langs, signs, masks = mini_batch
        h_out = self.rnn_mlp(mini_batch, False)[-1]
        t_out = transpose(reshape(h_out, (h_out.dim()[0][0], h_out.dim()[1])))

        sims = []
        norm_vals = self.norms(t_out).value()
        for i in range(len(langs)):
            for j in range(i+1, len(langs)):
                sims.append(dot_product(t_out[i], t_out[j])/(norm_vals[i]*norm_vals[j]))
        sim = esum(sims)
        sim.forward()
        sim_value = sim.value() / len(sims)
        renew_cg()
        return sim_value