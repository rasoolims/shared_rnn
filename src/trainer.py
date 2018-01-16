from optparse import OptionParser
from network import Network
from utils import *
import pickle, time, os, sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_data",  metavar="FILE", default=None)
    parser.add_option("--dev", dest="dev_data", metavar="FILE", default=None)
    parser.add_option("--test", dest="test_data", metavar="FILE", default=None)
    parser.add_option("--output", dest="conll_output",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=5000)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--ce", type="int", dest="ce", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--t", type="int", dest="t", default=50000)
    parser.add_option("--lr", type="float", dest="lr", default=0.002)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.9)
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
    chars = read_chars(options.train_data)
    with open(os.path.join(options.output, "params.pickle"), 'w') as paramsfp:
        pickle.dump((chars, options), paramsfp)
    universal_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    network = Network(universal_tags, chars, options)
    print 'splitting train data'
    num_train_batches = split_data(options.train_data, options.output+'/train.')
    num_dev_batches = split_data(options.dev_data, options.output+'/dev.')
    print 'starting epochs'
    for e in range(10):
        print 'epochs', (e+1)
        errors = []
        progress = 0
        for i in range(num_train_batches):
            r = random.randint(0, num_train_batches-1)
            start = time.time()
            train_minibatch = get_batches(options.output+'/train.'+str(r), network)
            random.shuffle(train_minibatch)
            for mini_batch in train_minibatch:
                errors.append(network.train(mini_batch, num_train_batches*10))
            progress += 1

            if len(errors) >= 10:
                print 'time',float(time.time()-start),'progress', round(float(100*progress)/num_train_batches, 2), '%, loss', sum(errors)/len(errors)

                errors = []
                dev_perf, num_item  = 0.0, 0
        for d in range(num_dev_batches):
            dev_minibatch = get_batches(options.output+'/dev.'+str(d), network)
            dev_perf += sum([network.eval(b) for b in dev_minibatch])
            num_item += num_dev_batches
        dev_perf /= num_item
        print 'dev sim for iteration', e+1, 'is', dev_perf
        network.save(options.output+'/model.'+str(e+1))