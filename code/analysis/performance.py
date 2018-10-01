import pickle
import numpy as np
import os
import re

TOTAL_BUT = 210.0
TOTAL_NEG = 314.0
TOTAL_DISC = 447.0

with open('data/sst2-sentence/neg_db', 'rb') as f:
    negation_database = pickle.load(f)


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
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_p.append(line.split('\t')[2])
            self.epochs[e]['mistakes_p'] = mistakes_p
            self.epochs[e]['but_p'] = (TOTAL_BUT - sum(map(has_but, mistakes_p))) / TOTAL_BUT
            self.epochs[e]['neg_p'] = (TOTAL_NEG - sum(map(has_negation, mistakes_p))) / TOTAL_NEG
            self.epochs[e]['discourse_p'] = (TOTAL_DISC - sum(map(has_discourse, mistakes_p))) / TOTAL_DISC
            # Building q data
            f1 = 'save/%s_seed_%d/incorrect_q_epoch_%d.txt' % (prior, run, e)
            with open(f1, 'r') as f:
                log = f.read().split('\n')
            mistakes_q = []
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_q.append(line.split('\t')[2])
            self.epochs[e]['mistakes_q'] = mistakes_q
            self.epochs[e]['but_q'] = (TOTAL_BUT - sum(map(has_but, mistakes_q))) / TOTAL_BUT
            self.epochs[e]['neg_q'] = (TOTAL_NEG - sum(map(has_negation, mistakes_q))) / TOTAL_NEG
            self.epochs[e]['discourse_q'] = (TOTAL_DISC - sum(map(has_discourse, mistakes_q))) / TOTAL_DISC

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


def print_best_n(text, results, silent=False, n=3):
    if silent is False:
        print("=" * (len(text) + 4))
        print("| %s |" % text)
        print("=" * (len(text) + 4))
    valid_results = np.array([result['val_p'] for result in results])
    test_results = np.array([result['test_p'] for result in results])
    best_n_indices = valid_results.argsort()[-n:][::-1]
    avg_valid, std_valid = np.mean(valid_results[best_n_indices]), np.std(valid_results[best_n_indices])
    best_n_indices = test_results.argsort()[-n:][::-1]
    avg_test, std_test = np.mean(test_results[best_n_indices]), np.std(test_results[best_n_indices])
    print("best %s :- average = %.2f \pm %.2f;" % ("valid", avg_valid, std_valid))
    print("best %s :- average = %.2f \pm %.2f;" % ("test", avg_test, std_test))

    worst_n_indices = valid_results.argsort()[:n]
    avg_valid, std_valid = np.mean(valid_results[worst_n_indices]), np.std(valid_results[worst_n_indices])
    worst_n_indices = test_results.argsort()[:n]
    avg_test, std_test = np.mean(test_results[worst_n_indices]), np.std(test_results[worst_n_indices])
    print("worst %s :- average = %.2f \pm %.2f;" % ("valid", avg_valid, std_valid))
    print("worst %s :- average = %.2f \pm %.2f;" % ("test", avg_test, std_test))

priors = ['default', 'logicnn', 'elmo']
all_results = []

for prior in priors:
    print("Printing %s results" % prior)
    results = []
    for i in range(1, 101):
        f1 = "logs/%s_seed_%d.log" % (prior, i)
        if os.path.exists(f1) is False:
            continue
        with open(f1, 'r') as f:
            log = f.read()
        r = Result(log, i, prior=prior)
        if r.num_epochs() == 20:
            results.append(r)

    all_results.append(results)

    print("Result files found - %d\n" % len(results))

    keys = ['test_p', 'test_q', 'but_p', 'but_q', 'neg_p', 'neg_q', 'discourse_p', 'discourse_q']

    print_result("early stopping on val_p", [result.best('val_p') for result in results], keys)
