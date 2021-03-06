import sys, os, codecs, random, pickle, gzip
from collections import defaultdict
import numpy as np
from utils import *

class Data:
    def __init__(self, bible_folder):
        lang_sentences_set = defaultdict(set)
        self.langs = set()
        de2dict = defaultdict(list)
        chars = defaultdict(set)
        print 'creating dictionaries'
        for flat_dir in os.listdir(bible_folder):
            l1 = flat_dir[:flat_dir.rfind('_')]
            l2 = flat_dir[flat_dir.rfind('_') + 1:]
            if l1 != 'de' and l2 != 'de':
                continue
            if l2 == 'de':
                l1, l2 = l2, l1
            self.langs.add(l1)
            self.langs.add(l2)

            f = bible_folder + flat_dir + '/'
            print f
            l1_tag = f + 'corpus.tok.clean.' + l1 + '.conll.tag'
            l2_tag = f + 'corpus.tok.clean.' + l2 + '.conll.tag'

            src_sens = codecs.open(l1_tag, 'r').read().strip().split('\n')
            dst_sens = codecs.open(l2_tag, 'r').read().strip().split('\n')

            assert len(src_sens) == len(dst_sens)
            for i in range(len(src_sens)):
                de2dict[normalize_sent(src_sens[i])].append((l2, normalize_sent(dst_sens[i])))

        self.de2dict, self.de2dict_dev = dict(), dict()

        for de_sen in de2dict.keys():
            if random.randint(0, 99) != 99:
                self.de2dict[de_sen] = de2dict[de_sen]
                for lsenPair in de2dict[de_sen]:
                    l2, sen = lsenPair
                    lang_sentences_set['de'].add(de_sen)
                    words, tags = get_words_tags(de_sen)
                    for word in words:
                        for ch in word:
                            chars['de'].add(ch)
                    lang_sentences_set[l2].add(sen)
                    words, tags = get_words_tags(sen)
                    for word in words:
                        for ch in word:
                            chars[l2].add(ch)
            else:
                self.de2dict_dev[de_sen] = de2dict[de_sen]

        self.neg_examples = defaultdict(list)
        for lang in lang_sentences_set.keys():
            self.neg_examples[lang] = list(lang_sentences_set[lang])
            print lang, len(self.neg_examples[lang])

        self.chars = dict()
        for l in chars.keys():
            self.chars[l] = sorted(list(chars[l]))
        self.langs = list(self.langs)

        output_list = defaultdict(list)
        for de_sen in self.de2dict_dev.keys():
            output_list['de'].append(de_sen)
            for pr in self.de2dict_dev[de_sen]:
                output_list[pr[0]].append(pr[1])

        self.shuffled_dict = dict()
        num_sen = len(output_list['de'])
        for i in range(num_sen):
            self.shuffled_dict[output_list['de'][i]] = list()
            for lang in output_list.keys():
                if lang == 'de': continue
                j = random.randint(0, len(output_list[lang]) - 1)
                self.shuffled_dict[output_list['de'][i]].append((lang, output_list[lang][j]))
        print 'Object data is completely loaded!', len(self.de2dict), len(self.de2dict_dev)

    def get_next(self, num_langs=3, neg_num = 1):
        l = len(self.neg_examples['de'])
        r = random.randint(0, l-1)
        de_sen = self.neg_examples['de'][r]
        langs_to_use = list(self.langs)
        random.shuffle(langs_to_use)
        langs_to_use = langs_to_use[:num_langs]

        output = []
        if 'de' in langs_to_use:
            output = ['de', de_sen]
        neg_examples_ = self.neg_examples['de']
        len_ = len(neg_examples_)
        i_ = [random.randint(1, len_ - 1) for _ in range(5)]

        neg_sens = []
        neg_ids = []

        if 'de' in langs_to_use:
            neg_sens = [neg_examples_[ind] for ind in i_]
            neg_ids = ['de' for _ in i_]

        langs_used = set()
        for pr in self.de2dict[de_sen]:
            if (pr[0] in langs_to_use) and (not pr[0] in langs_used): # second condition for making sure we don't have two English negative examples.
                langs_used.add(pr[0])
                output.append(pr[0])
                output.append(pr[1])
                neg_examples_ = self.neg_examples[pr[0]]
                if len(neg_examples_)==0:
                    print pr[0]
                    assert len(neg_examples_)>0
                len_ = len(neg_examples_)
                i_ = [random.randint(1, len_ - 1) for _ in range(neg_num)]
                neg_sens = neg_sens + [neg_examples_[ind] for ind in i_]
                neg_ids = neg_ids + [pr[0] for _ in i_]
        neg_output = []
        if len(output) >= 4:
            for i in range(len(neg_sens)):
                neg_output.append(neg_ids[i])
                neg_output.append(neg_sens[i])

        if len(neg_output)>0:
            return '\t'.join(output)+'\n'+'\t'.join(neg_output)

    def get_dev_batches(self, model, dev_dict):
        dev_batches = list()
        for de_sen in dev_dict.keys():
            output = ['de', de_sen]
            for pr in dev_dict[de_sen]:
                output.append(pr[0])
                output.append(pr[1])
            batch = defaultdict(list)
            c_len, w_len = defaultdict(int), 0
            for i in range(0, len(output), 2):
                lang_id = output[i].strip()
                words, tags = get_words_tags(output[i + 1])
                c_len[lang_id] = max(c_len[lang_id], max([len(w) for w in words]))
                w_len = max(w_len, len(words))
                batch[lang_id].append((words, tags, lang_id, 1))
            dev_batches.append(self.get_minibatch(batch, c_len, w_len, model))
        return dev_batches


    def get_next_batch(self, model, num_langs, neg_num):
        lines = None
        while lines is None:
            output = self.get_next(num_langs, neg_num)
            if output:
                lines = output.strip().split('\n')
        spl = lines[0].strip().split('\t')
        batch = defaultdict(list)
        c_len, w_len = defaultdict(int), 0
        for i in range(0, len(spl), 2):
            lang_id = spl[i].strip()
            words, tags = get_words_tags(spl[i + 1])
            c_len[lang_id] = max(c_len[lang_id], max([len(w) for w in words]))
            w_len = max(w_len, len(words))
            batch[lang_id].append((words, tags, lang_id, 1))
        spl = lines[1].strip().split('\t')
        for i in range(0, len(spl), 2):
            lang_id = spl[i].strip()
            words, tags = get_words_tags(spl[i + 1])
            c_len[lang_id] = max(c_len[lang_id], max([len(w) for w in words]))
            batch[lang_id].append((words, tags, lang_id, 0))

        return self.get_minibatch(batch, c_len, w_len, model)

    def get_minibatch(self, batch, cur_c_len, cur_len, model):
        all_batches = []
        for lang_id in batch.keys():
            all_batches += batch[lang_id]
        max_c_len = max(cur_c_len.values())
        signs = [all_batches[i][3] for i in range(len(all_batches))]
        langs = [all_batches[i][2] for i in range(len(all_batches))]
        chars, pwords, pos = dict(), dict(), dict()
        for lang_id in batch.keys():
            chars_ = [list() for _ in range(max_c_len)]
            for c_pos in range(max_c_len):
                ch = [model.PAD] * (len(batch[lang_id]) * cur_len)
                offset = 0
                for w_pos in range(cur_len):
                    for sen_position in range(len(batch[lang_id])):
                        if w_pos < len(batch[lang_id][sen_position][0]) and c_pos < len(batch[lang_id][sen_position][0][w_pos]):
                            ch[offset] = model.chars[lang_id].get(batch[lang_id][sen_position][0][w_pos][c_pos], 0)
                        offset += 1
                chars_[c_pos] = np.array(ch)
            chars[lang_id] = np.array(chars_)
            pwords[lang_id] = np.array([np.array(
                [model.evocab[langs[i]].get(batch[lang_id][i][0][j], 0) if j < len(batch[lang_id][i][0]) else model.PAD
                 for i in
                 range(len(batch[lang_id]))]) for j in range(cur_len)])
            pos[lang_id] = np.array([np.array(
                [model.pos.get(batch[lang_id][i][1][j], 0) if j < len(batch[lang_id][i][1]) else model.PAD for i in
                 range(len(batch[lang_id]))]) for j in range(cur_len)])
        masks = np.array([np.array([1 if 0 < j < len(all_batches[i][0]) else 0 for i in range(len(all_batches))])
                          for j in range(cur_len)])
        mini_batch = (pwords, pos, chars, langs, signs, masks)
        return mini_batch