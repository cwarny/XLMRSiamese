# XLMRSiamese

Extension of XLM-R with Siamese network and distance-based loss. See [experiments wiki](https://wiki.labcollab.net/confluence/display/Doppler/Siamese+XLM-R+for+multilingual+ZSL) for more info.

## Setup

* `source activate pytorch_p36`
* `pip install --no-index --find-links=file:/home/ec2-user/workspaces/hoverboard-workspaces/src/transformers transformers`
* `mkdir -p ~/.cache/torch; cp -r $WS/transformers_data ~/.cache/torch/transformers`
* `cd $WS/DeeperNlu; python setup.py install`

## Data

* Follow Kwaggle 2.0 instructions to get parallel corpora: https://kwaggle.aka.amazon.com/overview/onboarding
* Prep data: `python dataprep.py --parallel-data-dir data/parallel --output-dir data/processed`

## Pre-train

* `python pretrain.py`

## Train

* `python train.py`