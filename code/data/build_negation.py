import cPickle
import os

from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'../../stanford-corenlp-full-2018-02-27')


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab


def has_negation(sentence):
    tags = {x[0]: 1 for x in nlp.dependency_parse(sentence)}
    return 'neg' in tags

dirs = ['sst2-sentence/']
files = ['train.pickle', 'dev.pickle', 'test.pickle']
negation_database = {}

for dr in dirs:
    vocab, rev_vocab = load_vocab(os.path.join(dr, 'vocab'))
    for filename in files:
        print(filename)
        with open(os.path.join(dr, filename), 'rb') as f:
            data = cPickle.load(f)
        sentences = [' '.join([rev_vocab[x] for x in sent['sentence'] if x != 0]) for sent in data]
        for i, sentence in enumerate(sentences):
            if i % 100 == 0:
                print('Completed %d / %d in file %s' % (i, len(sentences), filename))
            if has_negation(sentence) is True:
                negation_database[sentence] = 1

with open(os.path.join(dr, 'neg_db'), 'wb') as f:
    data = cPickle.dump(negation_database, f)
