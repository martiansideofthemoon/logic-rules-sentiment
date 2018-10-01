import os
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import logicnn

from model.nn import SentimentModel

from test import evaluate, detailed_results
from utils import l1_schedule
from utils.data_utils import (
    load_pickle,
    load_vocab,
    load_w2v
)
from utils.logger import get_logger
from utils.initialize import initialize_w2v, initialize_weights

logger = get_logger(__name__)


def train(args):
    """Training uses TensorFlow placeholders. Easier data handling, bad memory utilization."""
    max_epochs = args.config.max_epochs
    batch_size = args.config.batch_size

    gpu_options = tf.GPUOptions(allow_growth=True)
    if args.thread_restrict is True:
        cfg_proto = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=2)
    else:
        cfg_proto = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=cfg_proto) as sess:
        # Loading the vocabulary files
        vocab, rev_vocab = load_vocab(args)
        args.vocab_size = len(rev_vocab)

        # Loading all the training data
        train_data = load_pickle(args, split='train')
        # Assuming a constant sentence length across the dataset
        args.config.seq_len = train_data[0]['sentence_len']

        # Creating training model
        if args.config.elmo is True:
            elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        else:
            elmo = None

        with tf.variable_scope("model", reuse=None):
            model = SentimentModel(args, queue=None, mode='train', elmo=elmo)

        # Reload model from checkpoints, if any
        steps_done = initialize_weights(sess, model, args, mode='train')
        logger.info("loaded %d completed steps", steps_done)

        # Load the w2v embeddings
        if steps_done == 0 and args.config.cnn_mode != 'rand':
            w2v_array = load_w2v(args, rev_vocab)
            initialize_w2v(sess, model, w2v_array)

        # Reusing weights for evaluation model
        with tf.variable_scope("model", reuse=True):
            model_eval = SentimentModel(args, None, mode='eval', elmo=elmo)

        if args.config.iterative is True or args.config.gradient is True:
            logicnn.append_features(args, train_data, model_eval, vocab, rev_vocab)
        num_batches = int(np.floor(float(len(train_data)) / batch_size))
        # Loading the dev data
        dev_set = load_pickle(args, split='dev')
        logicnn.append_features(args, dev_set, model_eval, vocab, rev_vocab)
        # Loading the test data
        test_set = load_pickle(args, split='test')
        logicnn.append_features(args, test_set, model_eval, vocab, rev_vocab)

        # This need not be zero due to incomplete runs
        epoch = model.epoch.eval()
        # Best percentage
        if os.path.exists(os.path.join(args.train_dir, 'best.txt')):
            with open(os.path.join(args.train_dir, 'best.txt'), 'r') as f:
                percent_best = float(f.read().strip())
        else:
            percent_best = 0.0

        start_batch = model.global_step.eval() % num_batches
        while epoch < max_epochs:
            # Shuffle training data every epoch
            if args.config.shuffle is True:
                random.shuffle(train_data)
            logger.info("Epochs done - %d", epoch)
            epoch_start = time.time()
            for i in range(start_batch, num_batches):
                split = train_data[i * batch_size:(i + 1) * batch_size]
                # Padding ignored here since it is time consuming and not necessary
                # Assumption is all sentences are pre-padded to same length
                labels = np.array([x['label'] for x in split])
                sentences = [np.array(x['sentence']) for x in split]
                feed_dict = {
                    model.inputs.name: sentences,
                    model.labels.name: labels
                }

                # Interfacing the logicnn algorithm
                if args.config.iterative is True:
                    # generate dynamic features for whole dataset
                    split_feats = logicnn.compute_features(args, split, sess, model_eval)
                    # weight settings in Hu et al. 2016
                    weights = np.array([1.0, 6.0])

                    # calculate logicnn probabilities
                    soft_labels = logicnn.compute_probability(
                        args, weights, split, split_feats
                    )
                    schedule = getattr(l1_schedule, args.config.l1_schedule)
                    feed_dict.update({
                        model.l1_weight.name: schedule(
                            epoch, num_batches * epoch + i, num_batches, args.config.l1_val
                        ),
                        model.soft_labels.name: soft_labels
                    })

                if args.config.elmo is True:
                    feed_dict.update({
                        model.input_strings.name: [x['pad_string'] for x in split]
                    })

                output_feed = [
                    model.updates,
                    model.clip,
                    model.final_cost
                ]

                _, _, final_cost = sess.run(output_feed, feed_dict)
                if (i + 1) % 1000 == 0:
                    logger.info(
                        "Epoch %d, minibatches done %d / %d. Avg Training Loss %.4f. Time elapsed in epoch %.4f.",
                        epoch, i + 1, num_batches, final_cost, (time.time() - epoch_start) / 3600.0
                    )
                if (i + 1) == num_batches:
                    logger.info("Evaluating model after %d minibatches", i + 1)
                    weights = np.array([1.0, 6.0])

                    dev_p_results, _ = evaluate(sess, model_eval, dev_set, args)
                    dev_feats = logicnn.compute_features(args, dev_set, sess, model_eval)
                    dev_probs = logicnn.compute_probability(args, weights, dev_set, dev_feats)
                    dev_q_results = logicnn.evaluate_logicnn(args, weights, dev_set, dev_probs)
                    dev_p = float(len(dev_p_results['correct'])) * 100.0 / len(dev_set)
                    dev_q = float(len(dev_q_results['correct'])) * 100.0 / len(dev_set)

                    test_p_results, _ = evaluate(sess, model_eval, test_set, args)
                    test_feats = logicnn.compute_features(args, test_set, sess, model_eval)
                    test_probs = logicnn.compute_probability(args, weights, test_set, test_feats)
                    test_q_results = logicnn.evaluate_logicnn(args, weights, test_set, test_probs)
                    test_p = float(len(test_p_results['correct'])) * 100.0 / len(test_set)
                    test_q = float(len(test_q_results['correct'])) * 100.0 / len(test_set)

                    detailed_results(args, 'p_epoch_%d' % epoch, test_set, rev_vocab, test_p_results)
                    detailed_results(args, 'q_epoch_%d' % epoch, test_set, rev_vocab, test_q_results)

                    logger.info(
                        "dev_p: %.4f, test_p: %.4f, dev_q: %.4f, test_q: %.4f",
                        dev_p, test_p, dev_q, test_q
                    )

                    if dev_p > percent_best:
                        percent_best = dev_p
                        logger.info("Saving Best Model")
                        checkpoint_path = os.path.join(args.best_dir, "sentence.ckpt")
                        model.best_saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
                        with open(os.path.join(args.train_dir, 'best.txt'), 'w') as f:
                            f.write(str(dev_p))
                    # Also save the model for continuing in future
                    checkpoint_path = os.path.join(args.train_dir, "sentence.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
            # Update epoch counter
            sess.run(model.epoch_incr)
            epoch += 1
            checkpoint_path = os.path.join(args.train_dir, "sentence.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
            start_batch = 0
