import re, codecs, sys, random, gzip, pickle
import numpy as np
from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf8')
lang_set = {'de', 'en', 'es'}

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def is_punc(pos):
    return pos == '.' or pos == 'PUNC' or pos == 'PUNCT' or \
           pos == "#" or pos == "''" or pos == "(" or \
           pos == "[" or pos == "]" or pos == "{" or pos == "}" or \
           pos == "\"" or pos == "," or pos == "." or pos == ":" or \
           pos == "``" or pos == "-LRB-" or pos == "-RRB-" or pos == "-LSB-" or \
           pos == "-RSB-" or pos == "-LCB-" or pos == "-RCB-" or pos == '"' or pos == ')'
