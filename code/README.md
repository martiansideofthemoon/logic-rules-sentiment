## Logic Rules for Sentiment Classification

This folder contains the code accompanying our EMNLP 2018 paper "Revisiting the Importance of Encoding Logic Rules in Sentiment Classification".

## Setup

1. Download Google's `word2vec` embeddings and place them inside `data/w2v/`. This is a large file (~ `3.5G`). You may `git clone` [this](https://github.com/mmihaltz/word2vec-GoogleNews-vectors), or use the URL provided in that repository's README. In case you want to use another folder for the `word2vec` embeddings, please edit `data/process_sst2_sentence.sh` accordingly.
2. This code has been tested for Python 3.6.5, TensorFlow 1.10.0, with CUDA 9.0 and cuDNN 7.0, with the whole set of requirements given in `requirements.txt`. To install dependencies, run this in a new `virtualenv`,
```
pip install -r requirements.txt
```
To re-build the negation database, you will need `stanfordcorenlp`. This database has been preprocessed and added to the repository under `data/sst2-sentence/neg_db`.
3. Pre-process the data by using,
```
cd data
./process_sst2.sh
./process-sst2-sentence.sh
```
For the experiments in the paper, only sentence level SST2 has been used, so it is sufficient to run `./process-sst2-sentence.sh`.
4. Run `./run.sh` with `mode` set to `default`, `logicnn` or `elmo`. This script will loop over 100 random seeds and store the results in the `log` and `save` folders.
5. Run the evaluation script using `python analysis/performance.py`.
6. You could also analyze saved models using the flag `--save-model` while training and running `--mode analysis` with the same `--job-id`. You could extract some dataset statistics using `python analysis/data-stats.py` and crowd data statistics using `python analysis/crowd-data-stats.py`.

## Model Settings
The argument and model configuration details have been added to `CONFIG.md`.

## Results

This code has been used to produce all results in the paper except `Figure 2`. `Figure 2` has been generated using [ZhitingHu/logicnn](https://github.com/ZhitingHu/logicnn/), to measure reproducibility against the author's original codebase.

## Contributing
Feel free to add Issues and PRs (for the existing issues). It should be fairly easy to understand the code.
