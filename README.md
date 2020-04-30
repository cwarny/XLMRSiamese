# XLMRSiamese

Extension of XLM-R with Siamese network and distance-based loss.

## Documentation

Generated documentation for the latest released version can be accessed here:
https://devcentral.amazon.com/ac/brazil/package-master/package/go/documentation?name=XLMRSiamese&interface=1.0&versionSet=live

## Data

* Follow Kwaggle 2.0 instructions to get parallel corpora: https://kwaggle.aka.amazon.com/overview/onboarding
* Prep data: `python dataprep.py --parallel-data-dir data/parallel --output-dir data/processed`

## Pre-train

* `python pretrain.py`

## Train

* `python train.py`