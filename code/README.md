# Sentence Classification in TensorFlow

This project is roughly an exact TensorFlow implementation of Yoon Kim's paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) (EMNLP 2014). His original Theano code can be found [here](https://github.com/yoonkim/CNN_sentence). Alternate to this, you can look at Denny Britz's TensorFlow implementation, [here](https://github.com/dennybritz/cnn-text-classification-tf).

## Setup

1. Download Google's `word2vec` embeddings and place them inside `data/w2v/`. This is a large file (~ `3.5G`). You may `git clone` [this](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).
2. Ensure you have a working `tensorflow` or `tensorflow-gpu` (version 1.1). Additional dependencies include `yaml`, `bunch` and `cPickle`.
3. Pre-process the data by using,
```
cd data
chmod +x process_sst2.sh process_sst2_sentence.sh
./process_sst2.sh
./process-sst2-sentence.sh
```
4. Run `python train.py` to train the model, and run `python train.py --mode test` to evaluate the model.

## Model Configuration
The model hyperparameters and mode (`nonstatic`, `static` and `rand`) are configured via YAML files inside `config/`. All hyperparameters (except `batch_size`) are identical to those reported in the paper. You may change the training directory via the `--job_id` parameter, and the random seed using `--seed`. Look at `config/arguments.py` for more details.

## Contributing
Feel free to add Issues and PRs (for the existing issues). It should be fairly easy to understand the code.
