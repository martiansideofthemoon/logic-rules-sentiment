import pickle
import os
import sys

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def load_pickle(args, split):
    logger.info("Loading split '%s'", split)
    with open(os.path.join(args.data_dir, split + ".pickle"), 'rb') as f:
        data = pickle.load(f)
    logger.info("Total # of %s samples - %d", split, len(data))
    return data


def load_vocab(args):
    vocab_file = os.path.join(args.data_dir, args.vocab_file)
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab


def load_w2v(args, rev_vocab):
    with open(os.path.join(args.data_dir, args.w2v_file), 'rb') as f:
        w2v = pickle.load(f)
    # Sanity check of the order of vectors
    for i, word in enumerate(rev_vocab):
        if w2v[i]['word'] != word:
            logger.info("Incorrect w2v file")
            sys.exit(0)
    w2v_array = np.array([x['vector'] for x in w2v])
    return w2v_array
