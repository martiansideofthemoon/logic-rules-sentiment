import cPickle
import numpy as np
import os

from collections import defaultdict


class Dataset(object):
    def __init__(self, dir, filename, rev_vocab):
        self.dir = dir
        self.filename = filename
        self.rev_vocab = rev_vocab

        with open(os.path.join(dr, filename), 'rb') as f:
            data = cPickle.load(f)

        self.sentences = \
            [' '.join([rev_vocab[x] for x in sent['sentence'] if x != 0]) for sent in data]

    def stats(self):
        a_but_b = [sent.split() for sent in self.sentences if has_but(sent) is True]
        a_words, b_words = defaultdict(int), defaultdict(int)
        total_a, total_b = 0, 0
        for sent in a_but_b:
            but_index = sent.index('but')
            for word in sent[:but_index]:
                a_words[word] += 1
                total_a += 1
            for word in sent[but_index + 1:]:
                b_words[word] += 1
                total_b += 1
        common_words = set(a_words.keys()).intersection(b_words)
        print("Total distinct A words :- %d" % len(a_words))
        print("Total distinct B words :- %d" % len(b_words))
        print("Number of common distinct words :- %d\n" % len(common_words))
        total_common = 0
        for word in list(common_words):
            total_common += min(a_words[word], b_words[word])
        print("Total A words :- %d" % total_a)
        print("Total B words :- %d" % total_b)
        print("Number of common words :- %d\n\n" % total_common)


def has_but(sentence):
    return ' but ' in sentence


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab

dirs = ['data/sst2-sentence/']
# dirs = ['data/mr/%d' % i for i in range(10)]
files = ['train.pickle', 'dev.pickle', 'test.pickle']


for dr in dirs:
    print("=" * (len(dr) + 4))
    print("| %s |" % dr)
    print("=" * (len(dr) + 4))
    vocab, rev_vocab = load_vocab(os.path.join(dr, 'vocab'))
    for file in files:
        dataset = Dataset(dr, file, rev_vocab)
        dataset.stats()
