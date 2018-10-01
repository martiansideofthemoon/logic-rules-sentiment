import tensorflow as tf

from utils.logger import get_logger

logger = get_logger(__name__)


def initialize_w2v(sess, model, w2v_array):
    feed_dict = {
        model.w2v_embeddings.name: w2v_array
    }
    sess.run(model.load_embeddings, feed_dict=feed_dict)
    logger.info("loaded word2vec values")


def initialize_weights(sess, model, args, mode='train'):
    ckpt = tf.train.get_checkpoint_state(args.load_dir)
    ckpt_best = tf.train.get_checkpoint_state(args.best_dir)
    if mode == 'test' and ckpt_best:
        logger.info("Reading best model parameters from %s", ckpt_best.model_checkpoint_path)
        model.saver.restore(sess, ckpt_best.model_checkpoint_path)
        steps_done = int(ckpt_best.model_checkpoint_path.split('-')[-1])
        # Since local variables are not saved
        sess.run([
            tf.local_variables_initializer()
        ])
    elif mode == 'train' and ckpt:
        logger.info("Reading model parameters from %s", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        steps_done = int(ckpt.model_checkpoint_path.split('-')[-1])
        # Since local variables are not saved
        sess.run([
            tf.local_variables_initializer()
        ])
    else:
        steps_done = 0
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])
    return steps_done
