import sys, os, codecs, random, pickle, gzip
from collections import defaultdict
import numpy as np
from utils import *

class AlignmentInstance:
    def __init__(self, src_lang, dst_lang, src_words, src_tags, dst_words, dst_tags, align_line):
        self.src_lang, self.dst_lang = src_lang, dst_lang
        self.src_words, self.src_tags = src_words, src_tags
        self.dst_words, self.dst_tags = dst_words, dst_tags
        self.alignment = [[int(spl) for spl in spls.split('-')] for spls in align_line.strip().split()]


class Data:
    def __init__(self, bible_folder, pos_tags):
        self.pos_tags = pos_tags
        lang_word_set = defaultdict(set)
        self.langs = set()
        chars = defaultdict(set)
        print 'creating dictionaries'

        self.alignments, self.dev_alignments = list(), list()

        for flat_dir in os.listdir(bible_folder):
            l1 = flat_dir[:flat_dir.rfind('_')]
            l2 = flat_dir[flat_dir.rfind('_') + 1:]
            f = bible_folder + flat_dir + '/'
            print f
            l1_tag = f + 'corpus.tok.clean.' + l1 + '.conll.tag'
            l2_tag = f + 'corpus.tok.clean.' + l2 + '.conll.tag'
            intesect_file = f + l1 + '_' + l2 + '.intersect'
            src_sens = codecs.open(l1_tag, 'r').read().strip().split('\n')
            dst_sens = codecs.open(l2_tag, 'r').read().strip().split('\n')
            intersections = open(intesect_file, 'r').read().strip().split('\n')
            self.langs.add(l1)
            self.langs.add(l2)

            assert len(src_sens) == len(dst_sens) == len(intersections)
            for i in range(len(src_sens)):
                src_words, src_tags = get_words_tags(normalize_sent(src_sens[i]))
                dst_words, dst_tags = get_words_tags(normalize_sent(dst_sens[i]))
                alignment = AlignmentInstance(l1, l2, src_words, src_tags, dst_words, dst_tags, intersections[i])
                if random.randint(0, 99) != 99:
                    self.alignments.append(alignment)
                    for word in src_words:
                        lang_word_set[l1].add(word)
                    for word in dst_words:
                        lang_word_set[l2].add(word)
                else:
                    self.dev_alignments.append(alignment)

        self.neg_examples = defaultdict(list)
        for lang in lang_word_set.keys():
            self.neg_examples[lang] = list(lang_word_set[lang])
            for word in self.neg_examples[lang]:
                for c in list(word):
                    chars[lang].add(c)
            print lang, len(self.neg_examples[lang]), len(chars[lang])

        self.chars = dict()
        for l in chars.keys():
            self.chars[l] = sorted(list(chars[l]))
        self.langs = list(self.langs)

        print 'Object data is completely loaded!', len(self.alignments), len(self.dev_alignments)

    def get_next_batch(self, model, batch_size=5, neg_num=5):
        batch = defaultdict(list)
        c_len = defaultdict(int)
        w_len = 0
        for b in range(batch_size):
            random_instance = self.alignments[random.randint(0, len(self.alignments)-1)]
            random_alignment_instance = random_instance.alignment[random.randint(0, len(random_instance.alignment)-1)]
            src_position, dst_position = random_alignment_instance
            src_words, dst_words = random_instance.src_words, random_instance.dst_words
            w_len = max(w_len, max(len(src_words), len(dst_words)))
            src_tags, dst_tags = random_instance.src_tags, random_instance.dst_tags
            src_lang, dst_lang = random_instance.src_lang, random_instance.dst_lang
            c_len[src_lang] = max(c_len[src_lang], max([len(w) for w in random_instance.src_words]))
            c_len[dst_lang] = max(c_len[dst_lang], max([len(w) for w in random_instance.dst_words]))
            batch[src_lang].append((src_words, src_tags, src_lang, 1, src_position, b))
            batch[dst_lang].append((dst_words, dst_tags, dst_lang, 1, dst_position, b))
            for r in range(neg_num):
                src_neg_tag = self.pos_tags[random.randint(0, len(self.pos_tags)-1)]
                dst_neg_tag = self.pos_tags[random.randint(0, len(self.pos_tags)-1)]
                src_neg_word = self.neg_examples[src_lang][random.randint(0, len(self.neg_examples[src_lang])-1)]
                dst_neg_word = self.neg_examples[dst_lang][random.randint(0, len(self.neg_examples[dst_lang])-1)]
                c_len[src_lang] = max(c_len[src_lang], len(src_neg_word))
                c_len[dst_lang] = max(c_len[dst_lang], len(dst_neg_word))
                src_neg_words = src_words[:src_position] + [src_neg_word] + src_words[src_position+1:]
                dst_neg_words = dst_words[:dst_position] + [dst_neg_word] + dst_words[dst_position+1:]
                src_neg_tags = src_tags[:src_position] + [src_neg_tag] + src_tags[src_position+1:]
                dst_neg_tags = dst_tags[:dst_position] + [dst_neg_tag] + dst_tags[dst_position+1:]
                batch[src_lang].append((src_neg_words, src_neg_tags, src_lang, 0, src_position, b))
                batch[dst_lang].append((dst_neg_words, dst_neg_tags, dst_lang, 0, dst_position, b))
        return self.get_minibatch(batch, c_len, w_len, model)

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

    def get_minibatch(self, batch, cur_c_len, cur_len, model):
        all_batches = []
        for lang_id in batch.keys():
            all_batches += batch[lang_id]
        langs = [all_batches[i][2] for i in range(len(all_batches))]
        batch_num, positions, signs, langs_in_batch = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(len(all_batches)):
            batch_num[all_batches[i][5]].append(i)
            positions[all_batches[i][5]].append(all_batches[i][4])
            signs[all_batches[i][5]].append(all_batches[i][3])
            langs_in_batch[all_batches[i][5]].append(all_batches[i][2])
        chars, pwords, pos = dict(), dict(), dict()
        char_batches = dict()
        uniq_words = dict()
        for lang_id in batch.keys():
            char_batches[lang_id] = dict()
            uniq_words[lang_id] = list()
            lang_words = dict()
            for sen_position in range(len(batch[lang_id])):
                char_batches[lang_id][sen_position] = dict()
                for w_pos in range(cur_len):
                    if w_pos < len(batch[lang_id][sen_position][0]):
                        w = batch[lang_id][sen_position][0][w_pos]
                    else:
                        w = '<PAD>'
                    if not w in lang_words:
                        lang_words[w] = len(uniq_words[lang_id])
                        uniq_words[lang_id].append(w)
                    char_batches[lang_id][sen_position][w_pos] = lang_words[w]

        for lang_id in batch.keys():
            chars_ = [list() for _ in range(cur_c_len[lang_id])]
            for c_pos in range(cur_c_len[lang_id]):
                ch = [model.PAD] * len(uniq_words[lang_id])
                offset = 0
                for w in uniq_words[lang_id]:
                    if c_pos < len(w):
                        ch[offset] = model.chars[lang_id].get(w[c_pos], 0)
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
        mini_batch = (pwords, pos, chars, langs_in_batch, signs, positions, batch_num, char_batches, masks)
        return mini_batch