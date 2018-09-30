import cPickle
import glob
import os
import random
import sys

import tensorflow as tf
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


def load_train_tfrecords(args):
    logger.info("Loading training data from %s", args.data_dir)
    train_files = glob.glob(os.path.join(args.data_dir, "train*.tfrecords"))
    logger.info("%d training file(s) used", len(train_files))
    number_of_instances = 0
    for i, train_file in enumerate(train_files):
        number_of_instances += sum([1 for _ in tf.python_io.tf_record_iterator(train_file)])
        # Using ceil below since we allow for smaller final batch
    batches_per_epoch = int(np.ceil(number_of_instances / float(args.config.batch_size)))
    logger.info("Total # of minibatches per epoch - %d", batches_per_epoch)
    return train_files, number_of_instances


def load_pickle(args, split):
    logger.info("Loading split '%s'", split)
    with open(os.path.join(args.data_dir, split + ".pickle"), 'rb') as f:
        data = cPickle.load(f)
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
        w2v = cPickle.load(f)
    # Sanity check of the order of vectors
    for i, word in enumerate(rev_vocab):
        if w2v[i]['word'] != word:
            logger.info("Incorrect w2v file")
            sys.exit(0)
    w2v_array = np.array([x['vector'] for x in w2v])
    return w2v_array
