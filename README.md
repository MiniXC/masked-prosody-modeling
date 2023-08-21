[![tests](https://github.com/MiniXC/masked-prosody-modeling/actions/workflows/run_lint_and_test.yml/badge.svg)](https://github.com/MiniXC/ml-template/actions/workflows/run_lint_and_test.yml)
# Machine Leraning Template
Template for my machine learning projects. 

It's quite opinionated in the following (and probably more) ways:
- uses accelerate
- splits up config, model and scripts
- assumes one is always using a huggingface dataset (this can also be done using ``load_dataset('some_dataset.py')``)
- uses collators as the main way to process data that hasn't been preprocessed by the dataset
- uses separate configs for training (everything not shipped with the model), model and collator

## Architecture
The following updates automatically every time ``scripts/train.py`` is run.
<details>
<summary>Click to expand</summary>
<img src="./figures/model.png"></img>
</details>

## First Batch
The following updates automatically every time ``scripts/train.py`` is run.
<details>
<summary>Click to expand</summary>
<img src="./figures/first_batch.png"></img>
</details>