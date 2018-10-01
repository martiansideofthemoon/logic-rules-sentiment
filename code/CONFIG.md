## Model Arguments

These model flags can be see in `config/arguments.py`. Here's a short description,

1. `--config-file` - The model configuration file that needs to be loaded, with respect to parent directory. These files are typically located in the `config` folder. The current configurations include `default.yml`, `logicnn.yml` and `elmo.yml`.
2. `--modify-config` - JSON string giving users flexibility to adjust configuration via command line flags.
3. `--thread-restrict` - Restrict the model to 2 CPU threads only.
4. `--data_dir` - Directory containing preprocessed data.
5. `--train_dir` - Directory to store checkpoints and result files.
6. `--best_dir` - Directory to store best checkpoints.
7. `--vocab_file` - File having list of words. Assumed to be inside `data_dir`.
8. `--w2v_file` - File having processed word2vec embeddings for dataset.
9. `--seed` - Random seed used by all libraries.
10. `--job_id` - Unique string identifer for the job.
11. `--load_id` - Option to load current model from other completed job.
12. `--no-cache` - Delete all saved files with supplied `job_id`. Does not do this in `test` or `analysis` mode.
13. `--eval_splits` - Comma separated list of all evaluation split names.
14. `--save-model` - Whether or not model parameters need to be saved after every epoch.
15. `--mode` - This can be `train`, `test` or `analysis`. All modes should be supplied the same configuration file and `job_id` via arguments.

## Model Configuration

These are YAML files located in the `config` folder.

1. `batch_size` - Batch size during training.
2. `eval_batch_size` - Batch size during evaluation.
3. `lr` - Learning rate.
4. `embedding_size` - Use `300` for `word2vec` in the `static` or `nonstatic` CNN modes.
5. `num_classes` - Number of output classes, stick with `2` for SST2.
6. `keep_prob` - `1 - dropout_rate`.
7. `clipped_norm` - Clipped norm for fully connected weight parameter in Kim et al. 2014.
8. `optimizer` - Only `adam` or `adadelta` supported.
9. `conv_filters` - List of filter sizes and number of channels in each filter.
10. `max_epochs` - Number of epochs the model is trained for.
11. `shuffle` - Shuffle training data from batch to batch.
12. `cnn_mode` - This can be `rand`, `static` or `nonstatic`. Refer to Kim et al. 2014 for a description of each of these modes.
13. `iterative` - Set `True` to use distillation in the loss function.
14. `l1_schedule` - Interpolation constant schedule between soft label and hard label loss. Currently supported schedules are `logicnn`, `constant`, `step` and `interpolate`.
15. `l1_val` - Interpolation value for hard label loss for the `interpolate` mode. This acts like the decay constant for the `logicnn` mode.
16. `elmo` - Set `True` to use ELMo word representations.