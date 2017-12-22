import re, codecs,sys, random, gzip
import numpy as np
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf8')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

def get_batches(file_path, model, is_dev = False):
    mini_batches = []
    reader = gzip.open(file_path, 'r')
    line = reader.readline()
    lang_set = {'de', 'en', 'es', 'fr'}
    while line:
        spl = line.strip().split('\t')
        batch = defaultdict(list)
        c_len, w_len = defaultdict(int), 0
        for i in range(0, len(spl), 2):
            lang_id = spl[i].strip()
            if not lang_id in lang_set:
                continue
            words, tags = [], []
            for sen_t in spl[i+1].strip().split():
                r = sen_t.rfind('_')
                words.append(sen_t[:r])
                tags.append(sen_t[r+1:])
            c_len[lang_id] = max(c_len[lang_id], max([len(w) for w in words]))
            w_len = max(w_len, len(words))
            batch[lang_id].append((words, tags, lang_id, 1))
        if not is_dev:
            line = reader.readline()
            spl = line.strip().split('\t')
            for i in range(0, len(spl), 2):
                lang_id = spl[i].strip()
                if not lang_id in lang_set:
                    continue
                words, tags = [], []
                for sen_t in spl[i+1].strip().split():
                    r = sen_t.rfind('_')
                    words.append(sen_t[:r])
                    tags.append(sen_t[r+1:])
                c_len[lang_id] = max(c_len[lang_id], max([len(w) for w in words]))
                batch[lang_id].append((words, tags, lang_id, 0))
        add_to_minibatch(batch, c_len, w_len, mini_batches, model)
        #if len(mini_batches)>100: break #todo
    return mini_batches



def add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model):
    all_batches = []
    for lang_id in batch.keys():
        all_batches +=  batch[lang_id]
    max_c_len = max(cur_c_len.values())
    signs = [all_batches[i][3] for i in range(len(all_batches))]
    langs = [all_batches[i][2] for i in range(len(all_batches))]
    chars, pwords, pos = dict(), dict(), dict()
    for lang_id in batch.keys():
        chars[lang_id] = np.array([[[model.chars[lang_id].get(batch[lang_id][i][0][j][c].lower(), 0)
                        if 0 < j < len(batch[lang_id][i][0]) and c < len(batch[lang_id][i][0][j]) else (1 if j == 0 and c == 0 else 0)
                        for i in range(len(batch[lang_id]))] for j in range(cur_len)] for
                        c in range(max_c_len)])
        chars[lang_id] = np.transpose(np.reshape(chars[lang_id], (len(batch[lang_id]) * cur_len, max_c_len)))
        pwords[lang_id] = np.array([np.array(
            [model.evocab[langs[i]].get(batch[lang_id][i][0][j], 0) if j < len(batch[lang_id][i][0]) else model.PAD for i in
             range(len(batch[lang_id]))]) for j in range(cur_len)])
        pos[lang_id] = np.array([np.array(
            [model.pos.get(batch[lang_id][i][1][j], 0) if j < len(batch[lang_id][i][1]) else model.PAD for i in
             range(len(batch[lang_id]))]) for j in range(cur_len)])
    masks = np.array([np.array([1 if 0 < j < len(all_batches[i][0]) else 0 for i in range(len(all_batches))])
                      for j in range(cur_len)])
    mini_batches.append((pwords, pos, chars, langs, signs, masks))


def is_punc(pos):
	return  pos=='.' or pos=='PUNC' or pos =='PUNCT' or \
        pos=="#" or pos=="''" or pos=="(" or \
		pos=="[" or pos=="]" or pos=="{" or pos=="}" or \
		pos=="\"" or pos=="," or pos=="." or pos==":" or \
		pos=="``" or pos=="-LRB-" or pos=="-RRB-" or pos=="-LSB-" or \
		pos=="-RSB-" or pos=="-LCB-" or pos=="-RCB-" or pos=='"' or pos==')'

