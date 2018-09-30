import cPickle
import glob
import numpy as np
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('analysis/avg_epochs.pdf')
pyplot.figure()
pyplot.clf()


# Global variable information
def has_but(sentence):
    return ' but ' in sentence


def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        rev_vocab = f.read().split('\n')
    vocab = {v: i for i, v in enumerate(rev_vocab)}
    return vocab, rev_vocab


_, rev_vocab = load_vocab('data/mr/vocab')


class Result(object):
    re1 = re.compile(r'dev_p:\s(\d*\.?\d*),\stest_p:\s(\d*\.?\d*),\sdev_q:\s(\d*\.?\d*),\stest_q:\s(\d*\.?\d*)')

    def __init__(self, log, run, prior):
        self.log = log.split('\n')
        self.epochs = []
        self.run = run
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
        # Getting the mistakes data
        for e in range(len(self.epochs)):
            # Building p data
            f1 = 'save/%s_cv_%d/incorrect_p_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            mistakes_p = []
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_p.append(line.split('\t')[2])
            self.epochs[e]['mistakes_p'] = mistakes_p
            self.epochs[e]['but_p'] = sum(map(has_but, mistakes_p))
            # Building q data
            f1 = 'save/%s_cv_%d/incorrect_q_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            mistakes_q = []
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_q.append(line.split('\t')[2])
            self.epochs[e]['mistakes_q'] = mistakes_q
            self.epochs[e]['but_q'] = sum(map(has_but, mistakes_q))

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


priors = ['grad2_mr_dev_99', 'grad_mr_dev_100']
all_results = []

for prior in priors:
    print("Printing %s results" % prior)
    results = []
    for i in range(100):
        f1 = "logs/%s_cv_%d.log" % (prior, i)
        if os.path.exists(f1) is False:
            continue
        with open(f1, 'r') as f:
            log = f.read()
        r = Result(log, i, prior=prior)
        if r.num_epochs() == 20:
            results.append(r)

    all_results.append(results)

    print("Result files found - %d\n" % len(results))

    keys = ['test_p', 'test_q', 'but_p', 'but_q']

    print_result("early stopping on val_p", [result.best('val_p') for result in results], keys)
    print_result("early stopping on test_p", [result.best('test_p') for result in results], keys)


# Plotting results
keys = ['test_p', 'test_q']
# keys = ['but_p', 'but_q']
for prior, results in zip(priors, all_results):
    averages, stds, maxs = np.zeros([20, len(keys)]), np.zeros([20, len(keys)]), np.zeros([20, len(keys)])
    for i in range(1, 21):
        averages[i - 1], stds[i - 1], maxs[i - 1] = \
            print_result("Epoch %d" % i, [result.epoch(i) for result in results], keys, silent=True)
    for i, k in enumerate(keys):
        pyplot.errorbar(np.array(range(1, 21)), averages[:, i], yerr=stds[:, i], label="%s_%s" % (prior, k))


legend = pyplot.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
pyplot.xlabel('epochs')
pyplot.ylabel('mistakes (%d jobs)' % len(results))
pp.savefig(bbox_inches="tight")
pp.close()
