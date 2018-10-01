import pickle
import os
import random
import re
import sys

import numpy as np

from collections import defaultdict

np.random.seed(1)
random.seed(1)

MAX_LEN = 53
MAX_PAD = 4
TOTAL_LEN = MAX_LEN + 2 * MAX_PAD


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs, layer1_size


def build_data(filename, word_freq, clean_string=True):
    """
    Loads data
    """
    revs = []
    with open(filename, "rb") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            label = int(line[0])
            input_str = line[2:].strip()
            if clean_string:
                orig_rev = clean_str(input_str)
            else:
                orig_rev = input_str.lower()
            words = set(orig_rev.split())
            for word in words:
                word_freq[word] += 1
            orig_rev = "<PAD> <PAD> <PAD> <PAD> " + orig_rev
            orig_rev += " <PAD>" * (TOTAL_LEN - len(orig_rev.split()))
            datum = {
                "label": label,
                "text": orig_rev,
                "num_words": TOTAL_LEN,
                "sentence_id": line_no
            }
            revs.append(datum)
    random.shuffle(revs)
    return revs


def clean_str(string, trec=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if trec else string.strip().lower()


def build_vocab(word_freq):
    rev_vocab = ['<PAD>']
    rev_vocab.extend([x[0] for x in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)])
    vocab = {v: k for k, v in enumerate(rev_vocab)}
    return rev_vocab, vocab


def add_random_vectors(word_to_vec, rev_vocab, vector_size):
    for word in rev_vocab:
        if word not in word_to_vec:
            word_to_vec[word] = np.random.uniform(-0.25, 0.25, vector_size)


def write_pickle(pickle_path, data, vocab):
    pickle_output = []
    for i, datum in enumerate(data):
        sentence = [vocab[x] for x in datum['text'].split()]
        sentence_len = datum['num_words']
        label = datum['label']
        sentence_id = datum['sentence_id']
        # Sanity check before save
        if sentence_len != len(sentence):
            print("error!")
            sys.exit(0)
        pickle_output.append({
            'sentence': sentence,
            'label': label,
            'sentence_len': sentence_len,
            "sentence_id": sentence_id,
            'order_id': i,
            'pad_string': datum['text']
        })
    pickle.dump(pickle_output, open(pickle_path, "wb"))


if __name__ == "__main__":
    stsa_path = sys.argv[1]
    w2v_file = sys.argv[2]

    train_data_file = os.path.join(stsa_path, "stsa.binary.train")
    dev_data_file = os.path.join(stsa_path, "stsa.binary.dev")
    test_data_file = os.path.join(stsa_path, "stsa.binary.test")
    database = {
        'train': {
            'filename': train_data_file
        },
        'dev': {
            'filename': dev_data_file
        },
        'test': {
            'filename': test_data_file
        }
    }

    word_freq = defaultdict(int)
    for v in database.values():
        v['data'] = build_data(v['filename'], word_freq)

    # Next, convert the vocabulary to correct form
    rev_vocab, vocab = build_vocab(word_freq)
    with open(os.path.join(stsa_path, 'vocab'), 'w') as f:
        f.write('\n'.join(rev_vocab))

    # Next, load Google's word vectors
    word_to_vec, vector_size = load_bin_vec(w2v_file, vocab)
    word_to_vec['<PAD>'] = np.zeros(vector_size)
    # Finally, add random vectors for unknown words
    add_random_vectors(word_to_vec, rev_vocab, vector_size)
    # Saving the word map
    word_map = []
    for word in rev_vocab:
        word_map.append({
            'word': word,
            'vector': word_to_vec[word]
        })

    pickle.dump(word_map, open(os.path.join(stsa_path, "w2v.pickle"), "wb"))

    # Finally, build Pickle files
    for k, v in database.items():
        pickle_path = os.path.join(stsa_path, k + ".pickle")
        write_pickle(pickle_path, v['data'], vocab)
