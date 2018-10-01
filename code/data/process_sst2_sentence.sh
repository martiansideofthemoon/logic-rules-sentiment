#!/bin/bash

# preprocess raw data
python2 process-sst2-sentence.py ./sst2-sentence/ ./w2v/GoogleNews-vectors-negative300.bin
