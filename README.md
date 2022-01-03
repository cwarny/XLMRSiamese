# XLMRSiamese

Extension of XLM-R with Siamese network and distance-based loss.

## Setup

* `source activate pytorch_p36`
* `pip install transformers`
* [optional] Install apex

## Data

* Prep data: `python dataprep.py --parallel-data-dir data/parallel --output-dir data/processed`

## Pre-train

* `python pretrain_distributed.py -i data/processed/train.tsv -o models --batch-size 16 --gpus 8`

## Train

* `python train.py`