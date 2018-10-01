## Setup

1. Download Google's `word2vec` embeddings and place them inside `data/w2v/`. This is a large file (~ `3.5G`). You may `git clone` [this](https://github.com/mmihaltz/word2vec-GoogleNews-vectors).
2. Ensure you have a working `tensorflow` or `tensorflow-gpu` (version 1.1). You will also need `tensorflow_hub` for the ELMo models. Additional dependencies include `scipy`, `matplotlib`, `numpy`, `pyyaml`, `munch`. To build the negation database (this has already been done), you will also need `stanfordcorenlp`.
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
