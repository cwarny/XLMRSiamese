# XLMRSiamese

Extension of XLM-R with Siamese network and distance-based loss. See [experiments wiki](https://wiki.labcollab.net/confluence/display/Doppler/Siamese+XLM-R+for+multilingual+ZSL) for more info.

## Data

* Follow Kwaggle 2.0 instructions to get parallel corpora: https://kwaggle.aka.amazon.com/overview/onboarding
* Prep data: `python dataprep.py --parallel-data-dir data/parallel --output-dir data/processed`

## Pre-train

* `python pretrain.py`

## Train

* `python train.py`