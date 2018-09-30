import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import perceptron

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


def calculate(sess, model_dev, data, args, vocab, rev_vocab):
    batch_size = args.config.eval_batch_size
    num_batches = int(np.ceil(float(len(data)) / batch_size))
    grads_l2 = np.zeros([len(data), data[0]['sentence_len']])
    for i in range(num_batches):
        split = data[i * batch_size:(i + 1) * batch_size]
        total = len(split)
        # The last batch is likely to be smaller than batch_size
        split.extend([split[-1]] * (batch_size - total))
        seq_len = np.array([x['sentence_len'] for x in split])
        max_seq_len = np.max(seq_len)
        # sentence_id = np.array([x['sentence_id'] for x in split])
        # labels = np.array([x['label'] for x in split])
        sents = [np.array(x['sentence']) for x in split]
        sentences = np.array([np.lib.pad(x, (0, max_seq_len - len(x)), 'constant') for x in sents])
        sentence_mask = perceptron.compute_mask(split)
        feed_dict = {
            model_dev.inputs.name: sentences,
            model_dev.sentence_mask: sentence_mask
        }

        # Tensorflow gradient computation code
        grads_l2[i * batch_size:(i + 1) * batch_size] = sess.run(model_dev.grads_l2, feed_dict=feed_dict)[:total]

    grad_stats(grads_l2, data, vocab)


def grad_stats(grads_l2, data, vocab):
    avg_grads = []
    avg_no_pad_grads = []
    avg_A_grads = []
    avg_A_no_pad_grads = []
    avg_B_grads = []
    avg_B_no_pad_grads = []

    for grad, instance in zip(grads_l2, data):
        grad_no_pad = np.array([x for token, x in zip(instance['sentence'], grad) if token != 0])
        sent_no_pad = [x for x in instance['sentence'] if x != 0]
        avg_grads.append(np.mean(grad))
        avg_no_pad_grads.append(np.mean(grad_no_pad))

        if vocab['but'] in instance['sentence']:
            # Try to see values of gradient before and after but
            but_location = instance['sentence'].index(vocab['but'])
            avg_A_grads.append(np.mean(grad[:but_location]))
            avg_B_grads.append(np.mean(grad[but_location:]))
            but_no_pad_location = sent_no_pad.index(vocab['but'])
            if but_no_pad_location != 0:
                avg_A_no_pad_grads.append(np.mean(grad_no_pad[:but_no_pad_location]))
                avg_B_no_pad_grads.append(np.mean(grad_no_pad[but_no_pad_location:]))

    logger.info("Average gradients :- %.4f", np.mean(avg_grads))
    logger.info("Average no pad gradients :- %.4f", np.mean(avg_no_pad_grads))
    logger.info("Average A gradients :- %.4f", np.mean(avg_A_grads))
    logger.info("Average A no pad gradients :- %.4f", np.mean(avg_A_no_pad_grads))
    logger.info("Average B gradients :- %.4f", np.mean(avg_B_grads))
    logger.info("Average B no pad gradients :- %.4f", np.mean(avg_B_no_pad_grads))


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

        perceptron.append_features(args, train_set, model_test, vocab, rev_vocab)

        dev_set = load_pickle(args, split='dev')
        perceptron.append_features(args, dev_set, model_test, vocab, rev_vocab)

        test_set = load_pickle(args, split='test')
        perceptron.append_features(args, test_set, model_test, vocab, rev_vocab)

        if args.config.elmo is True:
            elmo_embedding_analysis(sess, model_test, test_set)
        else:
            w2v_embedding_analysis(sess, model_test, test_set)
