import cPickle
import glob
import numpy as np
import os
import re
import sys

from scipy.stats import entropy

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages


pp = PdfPages('analysis/avg_epochs.pdf')
pyplot.figure()
pyplot.clf()

with open('data/sst2-sentence/neg_db', 'rb') as f:
    negation_database = cPickle.load(f)

# open crowd-sourced data
crowd_database = {}
with open('data/sst2-sentence/sst_crowd_discourse.txt', 'r') as f:
    data = f.read().split('\n')

for d in data:
    if len(d.strip()) == 0:
        continue
    crowd_database[d.split('\t')[2]] = float(d.split('\t')[0])


# Global variable information
def has_but(sentence):
    return ' but ' in sentence


def has_negation(sentence):
    return sentence in negation_database


def has_discourse(sentence):
    return has_but(sentence) or has_negation(sentence)


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab


class Result(object):
    re1 = re.compile(r'dev_p:\s(\d*\.?\d*),\stest_p:\s(\d*\.?\d*),\sdev_q:\s(\d*\.?\d*),\stest_q:\s(\d*\.?\d*)')

    def __init__(self, log, run, prior):
        self.log = log.split('\n')
        self.epochs = []
        self.run = run
        self.prior = prior
        epoch = 0
        for line in self.log:
            matches = Result.re1.findall(line)
            if len(matches) == 0:
                continue
            matches = matches[0]
            epoch += 1
            self.epochs.append({
                'epoch': epoch,
                'val_q': float(matches[2]),
                'val_p': float(matches[0]),
                'test_q': float(matches[3]),
                'test_p': float(matches[1]),
            })
        if len(self.epochs) > 0 and self.epochs[-1]['val_q'] < 80:
            print(run)
        # Getting the mistakes data
        for e in range(len(self.epochs)):
            # Building p data
            f1 = 'save/%s_seed_%d/incorrect_p_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            mistakes_p = []
            probs_p = {}
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_p.append(line.split('\t')[2])
                probs = line.split('\t')[1].replace('[', '').replace(']', '')
                probs = np.maximum(
                    np.array([float(x) for x in probs.split()]), 1e-5
                )
                if np.sum(probs) < 0.999 or np.sum(probs) > 1.001:
                    import pdb; pdb.set_trace()
                    print("Probs not summing to one")
                    sys.exit(0)
                probs_p[line.split('\t')[2]] = probs

            f1 = 'save/%s_seed_%d/correct_p_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            for line in log:
                if len(line.strip()) == 0:
                    continue
                probs = line.split('\t')[1].replace('[', '').replace(']', '')
                probs = np.maximum(
                    np.array([float(x) for x in probs.split()]), 1e-5
                )
                if np.sum(probs) < 0.999 or np.sum(probs) > 1.001:
                    import pdb; pdb.set_trace()
                    print("Probs not summing to one")
                    sys.exit(0)
                probs_p[line.split('\t')[2]] = probs

            self.epochs[e]['mistakes_p'] = mistakes_p
            self.epochs[e]['probs_p'] = probs_p
            self.epochs[e]['pred_p'] = {k: np.argmax(v) for k, v in probs_p.items()}
            self.epochs[e]['but_p'] = sum(map(has_but, mistakes_p))
            self.epochs[e]['neg_p'] = sum(map(has_negation, mistakes_p))
            self.epochs[e]['discourse_p'] = sum(map(has_discourse, mistakes_p))
            # Building q data
            f1 = 'save/%s_seed_%d/incorrect_q_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            mistakes_q = []
            probs_q = {}
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_q.append(line.split('\t')[2])
                probs = line.split('\t')[1].replace('[', '').replace(']', '')
                probs = np.maximum(
                    np.array([float(x) for x in probs.split()]), 1e-5
                )
                if np.sum(probs) < 0.999 or np.sum(probs) > 1.001:
                    import pdb; pdb.set_trace()
                    print("Probs not summing to one")
                    sys.exit(0)
                probs_q[line.split('\t')[2]] = probs

            f1 = 'save/%s_seed_%d/correct_q_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            for line in log:
                if len(line.strip()) == 0:
                    continue
                probs = line.split('\t')[1].replace('[', '').replace(']', '')
                probs = np.maximum(
                    np.array([float(x) for x in probs.split()]), 1e-5
                )
                if np.sum(probs) < 0.999 or np.sum(probs) > 1.001:
                    import pdb; pdb.set_trace()
                    print("Probs not summing to one")
                    sys.exit(0)
                probs_q[line.split('\t')[2]] = probs

            self.epochs[e]['mistakes_q'] = mistakes_q
            self.epochs[e]['probs_q'] = probs_q
            self.epochs[e]['pred_q'] = {k: np.argmax(v) for k, v in probs_q.items()}
            self.epochs[e]['but_q'] = sum(map(has_but, mistakes_q))
            self.epochs[e]['neg_q'] = sum(map(has_negation, mistakes_q))
            self.epochs[e]['discourse_q'] = sum(map(has_discourse, mistakes_q))

    def best(self, mode='val_q'):
        return max(self.epochs, key=lambda x: x[mode])

    def epoch(self, epoch=1):
        return self.epochs[epoch - 1]

    def num_epochs(self):
        return len(self.epochs)


def print_result(text, results, keys=None, silent=False):
    if silent is False:
        print("=" * (len(text) + 4))
        print("| %s |" % text)
        print("=" * (len(text) + 4))
    avgs, stds, maxs = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))
    if keys is None:
        keys = results[0].keys()
    for i, key in enumerate(keys):
        array = [result[key] for result in results]
        value, std, max_val, min_val = np.mean(array), np.std(array), np.max(array), np.min(array)
        if silent is False:
            print("%s :- average = %.4f +/- %.4f; range = %.4f to %.4f" % (key, value, std, min_val, max_val))
        avgs[i], stds[i], maxs[i] = value, std, max_val
    if silent is False:
        print("")
    return avgs, stds, maxs


def print_result_crowd(text, results, corpus):
    print("=" * (len(text) + 4))
    print("| %s |" % text)
    print("=" * (len(text) + 4))
    perfs_p = []
    perfs_q = []
    perfs_p_but = []
    perfs_q_but = []
    for result in results:
        pred_p = result['pred_p']
        pred_q = result['pred_q']
        perf_p = 0
        perf_q = 0
        perf_p_but = 0
        perf_q_but = 0
        total_but = 0
        for k, v in corpus.items():
            if has_but(k):
                total_but += 1
            if pred_p[k] == np.round(v):
                perf_p += 1
                if has_but(k):
                    perf_p_but += 1
            if pred_q[k] == np.round(v):
                perf_q += 1
                if has_but(k):
                    perf_q_but += 1
        perfs_p.append(float(perf_p) / len(corpus.keys()))
        perfs_q.append(float(perf_q) / len(corpus.keys()))
        perfs_p_but.append(float(perf_p_but) / total_but)
        perfs_q_but.append(float(perf_q_but) / total_but)
    value, std, max_val, min_val = np.mean(perfs_p), np.std(perfs_p), np.max(perfs_p), np.min(perfs_p)
    print("%s :- average = %.4f +/- %.4f; range = %.4f to %.4f" % ("test_p", value, std, min_val, max_val))
    value, std, max_val, min_val = np.mean(perfs_q), np.std(perfs_q), np.max(perfs_q), np.min(perfs_q)
    print("%s :- average = %.4f +/- %.4f; range = %.4f to %.4f" % ("test_q", value, std, min_val, max_val))
    value, std, max_val, min_val = np.mean(perfs_p_but), np.std(perfs_p_but), np.max(perfs_p_but), np.min(perfs_p_but)
    print("%s :- average = %.4f +/- %.4f; range = %.4f to %.4f" % ("but_p", value, std, min_val, max_val))
    value, std, max_val, min_val = np.mean(perfs_q_but), np.std(perfs_q_but), np.max(perfs_q_but), np.min(perfs_q_but)
    print("%s :- average = %.4f +/- %.4f; range = %.4f to %.4f" % ("but_q", value, std, min_val, max_val))
    print("")

# priors = ['grad2_99_logicnn', 'grad2_999_logicnn', 'grad2_90', 'grad2_999', 'grad2_99', 'grad_100', 'no_iter']
# priors = ['grad_sent_100', 'grad2_sent_99', 'grad2_elmo_sent_100', 'grad2_elmo_sent_99', 'grad2_elmo_sent_90', 'grad2_elmo_sent_999', 'grad2_elmo_sent_9999']
priors = ['sent_iter', 'grad_sent_100', 'grad2_elmo_sent_100']
# priors = ['no_iter', 'iter']
all_results = []

for prior in priors:
    print("Printing %s results" % prior)
    results = []
    for i in range(150):
        f1 = "logs/%s_seed_%d.log" % (prior, i)
        if os.path.exists(f1) is False:
            continue
        with open(f1, 'r') as f:
            log = f.read()
        r = Result(log, i, prior=prior)
        if r.num_epochs() == 20:
            results.append(r)
    results = results[:100]
    all_results.append(results)
    print("Result files found - %d\n" % len(results))

    keys = ['test_p', 'test_q', 'but_p', 'but_q']
    print_result("early stopping on val_p", [result.best('val_p') for result in results], keys)


# for ci in [0.5, 0.6666, 0.75, 0.9]:
for ci in [x / 18.0 for x in range(9, 18)]:
    print("Confidence Interval %.4f" % ci)
    corpus = {}
    for k, v in crowd_database.items():
        if v >= np.round(1 - ci, 3) and v <= np.round(ci, 3):
            continue
        corpus[k] = v
    print("Size of corpus %d" % len(corpus.keys()))
    for results in all_results:
        print_result_crowd(
            results[0].prior,
            [result.best('val_p') for result in results],
            corpus
        )
