import time
import random

import tensorflow as tf
import numpy as np

from model.features import features
from model.nn import SentimentModel
from model.perceptron_sgd import PerceptronModel
from test import evaluate_perceptron, detailed_results
from utils.data_utils import (
    load_pickle,
    load_vocab
)
from utils.logger import get_logger
from utils.initialize import initialize_weights

logger = get_logger(__name__)


def compute_mask(split):
    return np.array([x['features'][1].A_mask for x in split])


def append_features(args, data, model, vocab, rev_vocab):
    info = {
        'model_eval': model,
        'vocab': vocab,
        'rev_vocab': rev_vocab
    }
    for i, instance in enumerate(data):
        instance['features'] = [ft(args, instance, info) for ft in features]


def compute_features(args, data, sess, model):
    feature_data = np.zeros([len(features), len(data), args.config.num_classes])
    batch_size = args.config.eval_batch_size
    num_batches = int(np.ceil(float(len(data)) / batch_size))
    for i in range(num_batches):
        split = data[i * batch_size:(i + 1) * batch_size]
        total = len(split)
        # The last batch is likely to be smaller than batch_size
        split.extend([split[-1]] * (batch_size - total))
        for j, ft in enumerate(features):
            if ft.needs_nn is False:
                output = np.stack([x['features'][j].generate() for x in split], axis=0)
            else:
                feed_dict = {
                    model.inputs.name: np.stack([x['features'][j].final_inputs for x in split], axis=0)
                }
                if args.config.elmo is True:
                    feed_dict.update({
                        model.input_strings.name: [x['features'][j].final_string for x in split]
                    })
                output = sess.run(model.softmax, feed_dict=feed_dict)
                if ft.postprocess is True:
                    output = np.stack(
                        [x['features'][j].postprocess_func(output[k]) for k, x in enumerate(split)], axis=0
                    )
            feature_data[j, i * batch_size:(i + 1) * batch_size, :] = output[:total]
    return feature_data


def compute_probability(args, weights, split, split_features):

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    probs = np.zeros([len(split), args.config.num_classes])
    for i, x in enumerate(split):
        logits = np.array([
            np.dot(split_features[:, i, 0], weights),
            np.dot(split_features[:, i, 1], weights)
        ])
        probs[i] = softmax(logits)
    return probs


def perceptron_train(args, data, train_features):
    weights = np.zeros(len(features))
    epochs = args.config.iterative_epochs
    all_weights = np.zeros([epochs * len(data), len(features)])
    data_indices = range(len(data))
    for e in range(epochs):
        if args.config.perceptron_shuffle is True:
            random.shuffle(data_indices)
        for i, index in enumerate(data_indices):
            truth = data[index]['label']
            out_zero = np.dot(train_features[:, index, 0], weights)
            out_one = np.dot(train_features[:, index, 1], weights)
            if out_zero >= out_one and truth != 0:
                weights += train_features[:, index, 1] - train_features[:, index, 0]
            elif out_zero < out_one and truth != 1:
                weights += train_features[:, index, 0] - train_features[:, index, 1]
            all_weights[e * len(data) + i] = weights
    avg_weight = np.mean(all_weights, axis=0)
    if args.config.rescale is True:
        avg_weight = avg_weight / avg_weight[0]
    return avg_weight


def perceptron_sgd(args, sess, perceptron, data, train_features):
    epochs = args.config.iterative_epochs
    batch_size = args.config.perceptron_batch_size
    num_batches = int(np.floor(float(len(data)) / batch_size))
    sess.run([x.initializer for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='perceptron_model')])
    data_indices = range(len(data))
    for e in range(epochs):
        if args.config.perceptron_shuffle is True:
            random.shuffle(data_indices)
        for i in range(num_batches):
            batch_indices = data_indices[i * batch_size:(i + 1) * batch_size]
            split = [data[x] for x in batch_indices]
            split_features = train_features[:, batch_indices, :]

            labels = np.array([x['label'] for x in split])
            inputs = np.transpose(split_features, [1, 2, 0])
            feed_dict = {
                perceptron.inputs.name: inputs,
                perceptron.labels.name: labels
            }
            _, loss = sess.run([perceptron.updates, perceptron.cost], feed_dict=feed_dict)
    return sess.run(perceptron.weights)


def perceptron_weighted(args, data, train_features):
    # This function assumes only two features are present

    def evaluate(weight):
        # A helper function over which binary search will be applied
        wt = np.array([weight, 1 - weight])
        prod = 0.0
        for i, x in enumerate(data):
            truth = x['label']
            prod += np.log(np.dot(wt, train_features[:, i, truth]))
        return prod

    granularity = args.config.granularity
    maximum = -float('inf')
    max_wt = 0.0
    wt = 0.0
    while wt <= 1.0:
        test = evaluate(wt)
        if test > maximum:
            maximum, max_wt = test, wt
        wt += granularity
    return np.array([max_wt, 1 - max_wt])


def perceptron(args):
    if args.thread_restrict is True:
        cfg_proto = tf.ConfigProto(intra_op_parallelism_threads=2)
    else:
        cfg_proto = None
    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)
        args.config.batch_size = 1

        train_data = load_pickle(args, split='train')
        args.config.seq_len = train_data[0]['sentence_len']

        # Creating test model
        with tf.variable_scope("model", reuse=None):
            model_eval = SentimentModel(args, queue=None, mode='eval')
        if args.config.algorithm == 'sgd':
            with tf.variable_scope("perceptron_model", reuse=None):
                perceptron_model = PerceptronModel(args)
        # Reload model from checkpoints, if any
        steps_done = initialize_weights(sess, model_eval, args, mode='test')
        logger.info("loaded %d completed steps", steps_done)
        # Processing algorithm's features from here
        append_features(args, train_data, model_eval, vocab, rev_vocab)
        train_feats = compute_features(args, train_data, sess, model_eval)
        logger.info("Loaded perceptron features")
        if args.config.algorithm == 'sgd':
            weights = perceptron_sgd(args, sess, perceptron_model, train_data, train_feats)
        elif args.config.algorithm == 'weighted':
            weights = perceptron_weighted(args, train_data, train_feats)
        elif args.config.algorithm == 'perceptron':
            weights = perceptron_train(args, train_data, train_feats)
        elif args.config.algorithm == 'logicnn':
            weights = np.array([1.0, 6.0])
        else:
            weights = np.array([1.0, 0.0])
        logger.info("Perceptron training done")
        logger.info("Weights learned are %s", str(np.around(weights, 4)))

        for split in args.eval_splits.split(','):
            logger.info("Evaluation on %s data", split)
            if split == 'train':
                data, feats = train_data, train_feats
            else:
                data = load_pickle(args, split=split)
                append_features(args, data, model_eval, vocab, rev_vocab)
                feats = compute_features(args, data, sess, model_eval)
            probs = compute_probability(args, weights, data, feats)
            results = evaluate_perceptron(args, weights, data, probs)
            detailed_results(args, 'perceptron_%s' % split, data, rev_vocab, results)
            percent_correct = float(len(results['correct'])) * 100.0 / len(data)
            logger.info("correct predicitions on %s - %.4f.", split, percent_correct)
