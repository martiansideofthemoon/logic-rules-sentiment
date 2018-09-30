import cPickle
import glob
import numpy as np
import os
import operator
import re

from collections import defaultdict


with open('data/sst2-sentence/neg_db', 'rb') as f:
    negation_database = cPickle.load(f)


# Global variable information
def has_but(sentence):
    return ' but ' in sentence


def has_negation(sentence):
    sentence = sentence.split('\t')[-1]
    return sentence in negation_database


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
            mistakes_p_probs = []
            for line in log:
                if len(line.strip()) == 0:
                    continue
                mistakes_p.append(line.split('\t')[0] + '\t' + line.split('\t')[2])
                mistakes_p_probs.append(line.split('\t')[1])
            self.epochs[e]['mistakes_p'] = mistakes_p
            self.epochs[e]['mistakes_p_probs'] = mistakes_p_probs
            self.epochs[e]['but_p'] = sum(map(has_but, mistakes_p))
            self.epochs[e]['neg_p'] = sum(map(has_negation, mistakes_p))

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
            self.epochs[e]['but_q'] = sum(map(has_but, mistakes_q))
            self.epochs[e]['neg_q'] = sum(map(has_negation, mistakes_q))

    def best(self, mode='val_q'):
        return max(self.epochs, key=lambda x: x[mode])

    def epoch(self, epoch=1):
        return self.epochs[epoch - 1]

    def num_epochs(self):
        return len(self.epochs)


prior = 'grad_sent_100'

print("Printing %s results" % prior)
results = []
for i in range(150):
    if i % 10 == 0:
        print(i)
    f1 = "logs/%s_seed_%d.log" % (prior, i)
    if os.path.exists(f1) is False:
        continue
    with open(f1, 'r') as f:
        log = f.read()
    r = Result(log, i, prior=prior)
    if r.num_epochs() == 20:
        results.append(r)

results = results[:100]
print(len(results))

best_results_mistakes = [result.best('val_p')['mistakes_p'] for result in results]

but_mistakes = defaultdict(int)
neg_mistakes = defaultdict(int)
common = defaultdict(int)

for run in best_results_mistakes:
    for sentence in run:
        if has_but(sentence) is True:
            but_mistakes[sentence] += 1
        if has_negation(sentence) is True:
            neg_mistakes[sentence] += 1
        if has_negation(sentence) is True and has_but(sentence) is True:
            common[sentence] += 1

but_lengths = []
for sentence in but_mistakes.keys():
    but_lengths.append(len(sentence.split()) - 1)

neg_lengths = []
for sentence in neg_mistakes.keys():
    neg_lengths.append(len(sentence.split()) - 1)

print("Average Length of A-but-B mistakes = %.4f +/- %.4f" % (np.mean(but_lengths), np.std(but_lengths)))
print("Average Length of negation mistakes = %.4f +/- %.4f" % (np.mean(neg_lengths), np.std(neg_lengths)))

sorted_but = sorted(but_mistakes.items(), key=operator.itemgetter(1), reverse=True)
sorted_neg = sorted(neg_mistakes.items(), key=operator.itemgetter(1), reverse=True)
sorted_common = sorted(common.items(), key=operator.itemgetter(1), reverse=True)

with open('analysis/baseline2_but_mistakes.txt', 'w') as f:
    f.write("\n".join(["%d\t%s" % (v, k) for k, v in sorted_but]))

with open('analysis/baseline2_neg_mistakes.txt', 'w') as f:
    f.write("\n".join(["%d\t%s" % (v, k) for k, v in sorted_neg]))

with open('analysis/baseline2_common_sentences.txt', 'w') as f:
    f.write("\n".join(["%d\t%s" % (v, k) for k, v in sorted_common]))
