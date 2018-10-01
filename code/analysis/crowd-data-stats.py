import pickle
import numpy as np

with open('data/sst2-sentence/neg_db', 'rb') as f:
    negation_database = pickle.load(f)


# Global variable information
def has_but(sentence):
    return ' but ' in sentence


def has_negation(sentence):
    return sentence in negation_database


# open SST2 data
with open('data/sst2-sentence/test.pickle', 'rb') as f:
    data = pickle.load(f)

sst2_database = {}
for d in data:
    no_pad = d['pad_string'].split()
    no_pad = ' '.join(filter(lambda a: a != '<PAD>', no_pad))
    sst2_database[no_pad] = d['label']

# open crowd-sourced data
crowd_database = {}
with open('data/sst2-sentence/sst_crowd_discourse.txt', 'r') as f:
    data = f.read().split('\n')

for d in data:
    if len(d.strip()) == 0:
        continue
    crowd_database[d.split('\t')[2]] = float(d.split('\t')[0])

for ci in [0.5, 0.6666, 0.75, 0.9]:
    neutral = 0
    flipped = 0
    neutral_but = 0
    flipped_but = 0
    flipped_sent = []
    for k, v in crowd_database.items():
        if v >= np.round(1 - ci, 3) and v <= np.round(ci, 3):
            neutral += 1
            if has_but(k):
                neutral_but += 1
        elif sst2_database[k] != np.round(v):
            flipped += 1
            flipped_sent.append([
                k, v, sst2_database[k]
            ])
            if has_but(k):
                flipped_but += 1
    print("Confidence Interval %.4f" % ci)
    print("Total neutral sentences = %d / %d" % (neutral, len(crowd_database.keys())))
    print("Total flipped sentences = %d / %d" % (flipped, len(crowd_database.keys())))
    print("Total neutral but sentences = %d / %d" % (neutral_but, len([k for k in crowd_database.keys() if has_but(k)])))
    print("Total flipped but sentences = %d / %d" % (flipped_but, len([k for k in crowd_database.keys() if has_but(k)])))
    print("Flipped Sentences - ")
    for i1, i2, i3 in flipped_sent:
        print("%.4f, %.4f, %s" % (i3, i2, i1))
