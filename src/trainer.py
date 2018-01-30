from optparse import OptionParser
from network import Network
from utils import *
import pickle, time, os, sys
from data_loader import Data


def eval(data, network):
    pl, nl, c = 0, 0, 0
    for dev_batch in data.get_dev_batches(network, 1):
        p, n = network.eval(dev_batch)
        pl+= p
        nl+= n
        c+= 1
    pl /= c
    nl /= c
    return pl, nl

def save(path):
    with open(path, 'w') as paramsfp:
        deep_lstm_params = []
        for i in range(len(network.deep_lstms.builder_layers)):
            builder = network.deep_lstms.builder_layers[i]
            params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
            d_par = dict()
            for j in range(len(params)):
                d_par[j] = params[j].expr().npvalue()
            deep_lstm_params.append(d_par)

        char_lstm_params = dict()
        for lang in network.char_lstm.keys():
            char_lstm_params[lang] = []
            for i in range(len(network.char_lstm[lang].builder_layers)):
                builder = network.char_lstm[lang].builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                d_par = dict()
                for j in range(len(params)):
                    d_par[j] = params[j].expr().npvalue()
                char_lstm_params[lang].append(d_par)

        proj_mat_params = dict()
        for lang in network.proj_mat.keys():
            proj_mat_params[lang] = network.proj_mat[lang].expr().npvalue()

        clookup_params = dict()
        for lang in network.clookup.keys():
            clookup_params[lang] = network.clookup[lang].expr().npvalue()

        plookup_params = network.plookup.expr().npvalue()
        lang_lookup_params = network.lang_lookup.expr().npvalue()

        pickle.dump((data.chars, network.lang2id, options, deep_lstm_params, char_lstm_params, clookup_params,
                     proj_mat_params, plookup_params, lang_lookup_params), paramsfp)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_data",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=5)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--ce", type="int", dest="ce", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--le", type="int", dest="le", help="language embedding", default=25)
    parser.add_option("--t", type="int", dest="t", default=50000)
    parser.add_option("--lr", type="float", dest="lr", default=0.001)
    parser.add_option("--num_lang", type="int", dest="num_lang", help="number of languages per training instance", default=4)
    parser.add_option("--neg_num", type="int", dest="neg_num", help="number of negative example per language", default=5)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.999)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--rnn", type="int", dest="rnn", help='dimension of rnn in each direction', default=200)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--eval_non_avg", action="store_true", dest="eval_non_avg", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--no_char", action="store_false", dest="use_char", default=True)
    parser.add_option("--no_pos", action="store_false", dest="use_pos", default=True)
    parser.add_option("--stop", type="int", dest="stop", default=50)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=512)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)
    parser.add_option("--dynet-l2", type="float", dest="dynet-l2", default=0)

    (options, args) = parser.parse_args()
    print 'loading chars'
    universal_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    data = Data(options.train_data, universal_tags)
    network = Network(universal_tags, data.chars, options)
    print 'splitting train data'
    print 'starting epochs'

    best_performance, nl =  eval(data, network)
    print 'dev sim/random:', best_performance, nl
    for e in range(10):
        print 'epochs', (e+1)
        errors = []
        progress = 0
        train_len = len(data.alignments)
        start = time.time()


        for i in range(train_len):
            minibatch = data.get_next_batch(network, options.batch, options.neg_num)
            errors.append(network.train(minibatch, train_len, options.neg_num))
            progress += 1
            if len(errors) >= 100:
                print 'time',float(time.time()-start),'progress', round(float(100*progress)/train_len, 2), '%, loss', sum(errors)/len(errors)
                start = time.time()
                errors = []
            if (i+1) % 100 == 0:
                dev_perform, nl = eval(data, network)
                print 'dev sim/random:', dev_perform, nl
                if dev_perform < best_performance:
                    best_performance = dev_perform
                    print 'saving', best_performance
                    save(os.path.join(options.output,"model"))

        dev_perform, nl = eval(data, network)
        print 'dev sim/random:', dev_perform, nl
        if dev_perform < best_performance:
            best_performance = dev_perform
            print 'saving', best_performance
            save(os.path.join(options.output, "model"))
