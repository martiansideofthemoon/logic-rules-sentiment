import numpy as np

from model.features import features
from utils.logger import get_logger

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
