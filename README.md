# XLMRSiamese

Extension of XLM-R with Siamese network and distance-based loss. See [experiments wiki](https://wiki.labcollab.net/confluence/display/Doppler/Siamese+XLM-R+for+multilingual+ZSL) for more info.

## Setup

* `source activate pytorch_p36`
* `pip install --no-index --find-links=file:/home/ec2-user/workspaces/hoverboard-workspaces/src/transformers transformers`
* `mkdir -p ~/.cache/torch; cp -r $WS/transformers_data ~/.cache/torch/transformers`
* `cd $WS/DeeperNlu; python setup.py install`
* [optional] `cd $WS/apex; pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`

## Data

* Follow Kwaggle 2.0 instructions to get parallel corpora: https://kwaggle.aka.amazon.com/overview/onboarding
* Prep data: `python dataprep.py --parallel-data-dir data/parallel --output-dir data/processed`

## Pre-train

* `python pretrain_distributed.py -i /efs-storage/data/processed/train.tsv -o /efs-storage/models --batch-size 16 --gpus 8`

## Train

* `python train.py`