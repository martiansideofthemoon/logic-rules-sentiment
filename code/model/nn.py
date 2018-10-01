import math

import tensorflow as tf

from utils.logger import get_logger


def random_uniform(limit):
    return tf.random_uniform_initializer(-limit, limit)


class SentimentModel(object):
    def __init__(self, args, queue=None, mode='train', elmo=None):
        self.logger = logger = get_logger(__name__)
        self.config = config = args.config
        self.elmo = elmo

        # Epoch variable and its update op
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_incr = self.epoch.assign(self.epoch + 1)
        self.queue = queue
        self.embedding_size = e_size = config.embedding_size
        self.num_classes = num_classes = config.num_classes
        # We can keep a larger batch size during evaluation to speed up computation
        if mode == 'train':
            self.batch_size = batch_size = config.batch_size
        else:
            self.batch_size = batch_size = config.eval_batch_size
        self.keep_prob = keep_prob = config.keep_prob
        self.clipped_norm = clipped_norm = config.clipped_norm

        # Learning rate variable and it's update op
        self.learning_rate = tf.get_variable(
            "lr", shape=[], dtype=tf.float32, trainable=False,
            initializer=tf.constant_initializer(config.lr)
        )
        self.global_step = tf.Variable(0, trainable=False)

        # Feeding inputs for evaluation
        self.inputs = tf.placeholder(tf.int64, [batch_size, args.config.seq_len])
        self.labels = tf.placeholder(tf.int64, [batch_size])
        self.segment_id = tf.placeholder(tf.int64, [batch_size])

        # Logic for embeddings
        self.w2v_embeddings = tf.placeholder(tf.float32, [args.vocab_size, e_size])
        if config.cnn_mode == 'static':
            embeddings = tf.get_variable(
                "embedding", [args.vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=False
            )
        else:
            embeddings = tf.get_variable(
                "embedding", [args.vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=True
            )
        # Used in the static / non-static configurations
        self.load_embeddings = embeddings.assign(self.w2v_embeddings)
        # Looking up input embeddings
        self.embedding_lookup = tf.nn.embedding_lookup(embeddings, self.inputs)

        if config.elmo is True:
            # Load the embeddings from the feed_dict
            self.input_strings = tf.placeholder(tf.string, [batch_size])
            self.embedding_lookup = elmo(self.input_strings, signature='default', as_dict=True)['elmo']
            self.input_vectors = input_vectors = tf.expand_dims(
                self.embedding_lookup, axis=3
            )
            self.embedding_size = e_size = 1024
        else:
            self.input_vectors = input_vectors = tf.expand_dims(self.embedding_lookup, axis=3)

        # Apply a convolutional layer
        conv_outputs = []
        self.debug = []
        for i, filter_specs in enumerate(config.conv_filters):
            size = filter_specs['size']
            channels = filter_specs['channels']
            debug = {}
            with tf.variable_scope("conv%d" % i):
                # Convolution Layer begins
                debug['filter'] = conv_filter = tf.get_variable(
                    "conv_filter%d" % i, [size, e_size, 1, channels],
                    initializer=random_uniform(0.01)
                )
                debug['bias'] = bias = tf.get_variable(
                    "conv_bias%d" % i, [channels],
                    initializer=tf.zeros_initializer()
                )
                debug['conv_out'] = output = tf.nn.conv2d(input_vectors, conv_filter, [1, 1, 1, 1], "VALID") + bias
                # Applying non-linearity
                output = tf.nn.relu(output)
                # Pooling layer, max over time for each channel
                debug['output'] = output = tf.reduce_max(output, axis=[1, 2])
                conv_outputs.append(output)
                self.debug.append(debug)

        # Concatenate all different filter outputs before fully connected layers
        conv_outputs = tf.concat(conv_outputs, axis=1)
        total_channels = conv_outputs.get_shape()[-1]

        # Adding a dropout layer during training
        # tf.nn.dropout is an inverted dropout implementation
        if mode == 'train':
            conv_outputs = tf.nn.dropout(conv_outputs, keep_prob=keep_prob)

        # Apply a fully connected layer
        with tf.variable_scope("full_connected"):
            self.W = W = tf.get_variable(
                "fc_weight", [total_channels, num_classes],
                initializer=random_uniform(math.sqrt(6.0 / (total_channels.value + num_classes)))
            )
            self.clipped_W = clipped_W = tf.clip_by_norm(W, clipped_norm)
            self.b = b = tf.get_variable(
                "fc_bias", [num_classes],
                initializer=tf.zeros_initializer()
            )
            self.logits = tf.matmul(conv_outputs, W) + b

        # Declare the vanilla cross-entropy loss function
        self.softmax = tf.nn.softmax(self.logits)
        self.one_hot_labels = tf.one_hot(self.labels, num_classes)
        self.loss1 = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.one_hot_labels
        )
        self.cost1 = tf.reduce_sum(self.loss1) / batch_size

        # Declare the soft-label distillation loss function
        self.soft_labels = tf.placeholder(tf.float32, [batch_size, num_classes])
        self.loss2 = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits,
            labels=self.soft_labels
        )
        self.cost2 = tf.reduce_sum(self.loss2) / batch_size

        # Interpolate the loss functions
        self.l1_weight = tf.placeholder(tf.float32)
        if config.iterative is False:
            self.final_cost = self.cost1
        else:
            self.final_cost = self.l1_weight * self.cost1 + (1.0 - self.l1_weight) * self.cost2

        if config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.learning_rate,
                rho=0.95,
                epsilon=1e-6
            )
        else:
            opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

        if mode == 'train':
            for variable in tf.trainable_variables():
                logger.info("%s - %s", variable.name, str(variable.get_shape()))
        # Apply optimizer to minimize loss
        self.updates = opt.minimize(self.final_cost, global_step=self.global_step)

        # Clip fully connected layer's norm
        with tf.control_dependencies([self.updates]):
            self.clip = W.assign(clipped_W)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def use_elmo(self):
        self.input_str = tf.placeholder(tf.string, [1])
        self.elmo_embeddings = self.elmo(
            self.input_str, signature='default', as_dict=True
        )
