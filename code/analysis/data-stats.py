import pickle
import numpy as np
import os


with open('data/sst2-sentence/neg_db', 'rb') as f:
    negation_database = pickle.load(f)


class Dataset(object):
    def __init__(self, dir, filename, rev_vocab):
        self.dir = dir
        self.filename = filename
        self.rev_vocab = rev_vocab

        with open(os.path.join(dr, filename), 'rb') as f:
            data = pickle.load(f)

        self.sentences = \
            [' '.join([rev_vocab[x] for x in sent['sentence'] if x != 0]) for sent in data]

    def stats(self):
        text = self.filename
        print("=" * (len(text) + 4))
        print("| %s |" % text)
        print("=" * (len(text) + 4))
        print("total sentences :- %d" % len(self.sentences))
        length = [len(sent.split()) for sent in self.sentences]
        print("average length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))
        a_but_b = [sent for sent in self.sentences if has_but(sent) is True]
        print("total A-but-B :- %d" % len(a_but_b))
        length = [len(sent.split()) for sent in a_but_b]
        print("average A-but-B length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))
        length = [sent.split().index('but') for sent in a_but_b]
        print("average A length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))
        length = [len(sent.split()) - sent.split().index('but') - 1 for sent in a_but_b]
        print("average B length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))

        negation = [sent for sent in self.sentences if has_negation(sent) is True]
        print("total negation :- %d" % len(negation))
        length = [len(sent.split()) for sent in negation]
        print("average negation length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))

        discourse = [sent for sent in self.sentences if has_discourse(sent) is True]
        print("total discourse :- %d" % len(discourse))
        length = [len(sent.split()) for sent in discourse]
        print("average discourse length :- %.4f +/- %.4f" % (np.mean(length), np.std(length)))

        with open('analysis/discourse_%s.tsv' % self.filename, 'w') as f:
            f.write('\n'.join(discourse))


def has_but(sentence):
    return ' but ' in sentence


def has_negation(sentence):
    return sentence in negation_database


def has_discourse(sentence):
    return has_but(sentence) or has_negation(sentence)


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab

dirs = ['data/sst2/']
files = ['train.pickle', 'dev.pickle', 'test.pickle']


for dr in dirs:
    print("=" * (len(dr) + 4))
    print("| %s |" % dr)
    print("=" * (len(dr) + 4))
    vocab, rev_vocab = load_vocab(os.path.join(dr, 'vocab'))
    for file in files:
        dataset = Dataset(dr, file, rev_vocab)
        dataset.stats()
