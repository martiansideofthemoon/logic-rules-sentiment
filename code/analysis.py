import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logicnn

from model.nn import SentimentModel
from utils.data_utils import (
    load_pickle,
    load_vocab,
    load_w2v
)
from utils.logger import get_logger
from utils.initialize import initialize_weights

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy
from scipy.spatial.distance import cosine


pp = PdfPages('analysis/embeddings.pdf')
plt.figure()
plt.clf()


logger = get_logger(__name__)


def has_but(sentence):
    return ' but ' in sentence


def ids_to_sent(ids, rev_vocab):
    return ' '.join([rev_vocab[x] for x in ids['sentence'] if x != 0])


def elmo_embedding_analysis(sess, model_test, test_set):
    sentence = "there are slow and repetitive parts , but it has just enough spice to keep it interesting"
    zero_list = [(8, 15)]
    # sentence = "marisa tomei is good , but just a kiss is just a mess"
    # zero_list = [(2, 9), (6, 10), (7, 11)]
    # sentence = "the irwins emerge unscathed , but the fictional footage is unconvincing and criminally badly acted"
    # zero_list = [(0, 6)]
    # sentence = "all ends well , sort of , but the frenzied comic moments never click"
    # zero_list = [(3, 6)]
    model_test.use_elmo()
    feed_dict = {
        model_test.input_str: [sentence]
    }
    embeddings = sess.run(model_test.elmo_embeddings, feed_dict)
    elmo = np.squeeze(embeddings['elmo'])
    grid = np.zeros((elmo.shape[0], elmo.shape[0]))
    for i in range(elmo.shape[0]):
        for j in range(elmo.shape[0]):
            grid[i, j] = 1 - cosine(elmo[i], elmo[j])

    mininum = np.min(grid)
    for i in range(elmo.shape[0]):
        grid[i, i] = mininum

    for zl in zero_list:
        grid[zl[0], zl[1]] = mininum
        grid[zl[1], zl[0]] = mininum

    ax = plt.gca()
    im = ax.imshow(grid)
    cbar = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(elmo.shape[0]))
    ax.set_yticks(np.arange(elmo.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(sentence.split(), rotation='vertical')
    ax.set_yticklabels(sentence.split())

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    pp.savefig(bbox_inches="tight")
    pp.close()


def w2v_embedding_analysis(sess, model_test, test_set):
    sentence = "there are slow and repetitive parts , but it has just enough spice to keep it interesting"
    zero_list = [(8, 15)]
    # sentence = "marisa tomei is good , but just a kiss is just a mess"
    # zero_list = [(2, 9), (6, 10), (7, 11)]
    # sentence = "the irwins emerge unscathed , but the fictional footage is unconvincing and criminally badly acted"
    # zero_list = [(0, 6)]
    # sentence = "all ends well , sort of , but the frenzied comic moments never click"
    # zero_list = [(3, 6)]
    input_str = None
    pad_str = None
    for sent in test_set:
        if sentence in sent['pad_string']:
            input_str = np.array([sent['sentence']])
            break
    if input_str is None:
        import pdb; pdb.set_trace()
    feed_dict = {
        model_test.inputs: input_str
    }
    embeddings = sess.run(model_test.embedding_lookup, feed_dict)
    embed = np.squeeze(embeddings)[4:4 + len(sentence.split())]

    grid = np.zeros((embed.shape[0], embed.shape[0]))
    for i in range(embed.shape[0]):
        for j in range(embed.shape[0]):
            grid[i, j] = 1 - cosine(embed[i], embed[j])

    mininum = np.min(grid)
    for i in range(embed.shape[0]):
        grid[i, i] = mininum

    for zl in zero_list:
        grid[zl[0], zl[1]] = mininum
        grid[zl[1], zl[0]] = mininum

    ax = plt.gca()
    im = ax.imshow(grid)
    cbar = ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set_xticks(np.arange(embed.shape[0]))
    ax.set_yticks(np.arange(embed.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(sentence.split(), rotation='vertical')
    ax.set_yticklabels(sentence.split())

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    pp.savefig(bbox_inches="tight")
    pp.close()


def analysis(args):
    if args.thread_restrict is True:
        cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    else:
        cfg_proto = None
    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)
        # Creating test model

        train_set = load_pickle(args, split='train')
        args.config.seq_len = train_set[0]['sentence_len']
        args.config.eval_batch_size = 1
        # Creating training model
        if args.config.elmo is True:
            elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        else:
            elmo = None

        with tf.variable_scope("model", reuse=None):
            model_test = SentimentModel(args, queue=None, mode='eval', elmo=elmo)

        # Reload model from checkpoints, if any
        steps_done = initialize_weights(sess, model_test, args, mode='test')
        logger.info("loaded %d completed steps", steps_done)

        logicnn.append_features(args, train_set, model_test, vocab, rev_vocab)

        dev_set = load_pickle(args, split='dev')
        logicnn.append_features(args, dev_set, model_test, vocab, rev_vocab)

        test_set = load_pickle(args, split='test')
        logicnn.append_features(args, test_set, model_test, vocab, rev_vocab)

        if args.config.elmo is True:
            elmo_embedding_analysis(sess, model_test, test_set)
        else:
            w2v_embedding_analysis(sess, model_test, test_set)
