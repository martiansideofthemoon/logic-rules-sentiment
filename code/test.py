import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from model.nn import SentimentModel
from utils.data_utils import (
    load_pickle,
    load_vocab
)
from utils.logger import get_logger
from utils.initialize import initialize_weights


logger = get_logger(__name__)


def evaluate(sess, model_dev, data, args):
    batch_size = args.config.eval_batch_size
    num_batches = int(np.ceil(float(len(data)) / batch_size))
    losses = 0.0
    incorrect = []
    incorrect_probs = []
    correct = []
    correct_probs = []
    for i in range(num_batches):
        split = data[i * batch_size:(i + 1) * batch_size]
        total = len(split)
        # The last batch is likely to be smaller than batch_size
        split.extend([split[-1]] * (batch_size - total))

        seq_len = np.array([x['sentence_len'] for x in split])
        max_seq_len = np.max(seq_len)
        sentence_id = np.array([x['sentence_id'] for x in split])
        labels = np.array([x['label'] for x in split])
        sents = [np.array(x['sentence']) for x in split]
        sentences = np.array([np.lib.pad(x, (0, max_seq_len - len(x)), 'constant') for x in sents])
        feed_dict = {
            model_dev.inputs.name: sentences,
            model_dev.labels: labels
        }
        if args.config.elmo is True:
            feed_dict.update({
                model_dev.input_strings.name: [x['pad_string'] for x in split]
            })
        softmax, loss = sess.run([model_dev.softmax, model_dev.loss1], feed_dict=feed_dict)
        sentence_id, softmax, labels, loss = \
            sentence_id[:total], softmax[:total], labels[:total], loss[:total]
        losses += np.sum(loss)
        outputs = np.argmax(softmax, axis=1)

        correct.extend(sentence_id[outputs == labels].tolist())
        correct_probs.extend(softmax[outputs == labels].tolist())
        incorrect.extend(sentence_id[outputs != labels].tolist())
        incorrect_probs.extend(softmax[outputs != labels].tolist())

    results = {
        'correct': correct,
        'correct_probs': correct_probs,
        'incorrect': incorrect,
        'incorrect_probs': incorrect_probs
    }
    return results, losses


def evaluate_perceptron(args, weights, data, probs):
    incorrect = []
    incorrect_probs = []
    correct = []
    correct_probs = []
    for i, x in enumerate(data):
        label, sentence_id = x['label'], x['sentence_id']
        if np.argmax(probs[i]) == label:
            correct.append(sentence_id)
            correct_probs.append(probs[i].tolist())
        else:
            incorrect.append(sentence_id)
            incorrect_probs.append(probs[i].tolist())

    results = {
        'correct': correct,
        'correct_probs': correct_probs,
        'incorrect': incorrect,
        'incorrect_probs': incorrect_probs
    }
    return results


def detailed_results(args, split, test_set, rev_vocab, results):
    # Convert `incorrect` back into sentences
    incorrect, incorrect_probs = results['incorrect'], results['incorrect_probs']
    correct, correct_probs = results['correct'], results['correct_probs']
    incorrect_sents = ""
    correct_sents = ""
    for sent in test_set:
        sentence_id = sent['sentence_id']
        sentence = ' '.join([rev_vocab[x] for x in sent['sentence'] if x != 0])

        if sentence_id in incorrect:
            probs = incorrect_probs[incorrect.index(sentence_id)]
            probs = np.around(probs, 4)
            incorrect_sents += "%d\t%s\t%s\n" % (sent['label'], str(probs), sentence)
        elif sentence_id in correct:
            probs = correct_probs[correct.index(sentence_id)]
            probs = np.around(probs, 4)
            correct_sents += "%d\t%s\t%s\n" % (sent['label'], str(probs), sentence)
        else:
            logger.error("Wrong sentence id")
            sys.exit(0)

    with open(os.path.join(args.train_dir, 'incorrect_%s.txt' % split), 'w') as f:
        f.write(str(incorrect_sents))
    with open(os.path.join(args.train_dir, 'correct_%s.txt' % split), 'w') as f:
        f.write(str(correct_sents))


def test(args):
    if args.thread_restrict is True:
        cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    else:
        cfg_proto = None
    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)
        # Creating test model

        # Hacky way to get seq_len
        test_set = load_pickle(args, split='test')
        args.config.seq_len = test_set[0]['sentence_len']

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

        for split in args.eval_splits.split(','):
            test_set = load_pickle(args, split=split)
            results, losses = evaluate(sess, model_test, test_set, args)
            if args.mode != 'train':
                detailed_results(args, split, test_set, rev_vocab, results)
            percent_correct = float(len(results['correct'])) * 100.0 / len(test_set)
            logger.info("correct predictions on %s - %.4f. Eval Losses - %.4f",
                        split, percent_correct, losses)
