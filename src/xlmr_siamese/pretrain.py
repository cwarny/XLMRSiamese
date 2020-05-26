import argparse
from pathlib import Path
from datetime import datetime
import json
from functools import partial
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import SequentialSampler

from transformers import XLMRobertaTokenizer

from deeper_nlu.data import TabularDataset, CategoryProcessor, MaskProcessor, TokenizeProcessor, DataBunch
from deeper_nlu.util import compose
from deeper_nlu.train import Learner, AvgStatsCallback, CudaCallback, SaveModel, EarlyStop
from deeper_nlu.metric import seq_acc, perplexity
from deeper_nlu.train import TripletLoss

from .loss import SiameseLoss
from .sampler import MultinomialSampler
from .callback import FetchHardestNegatives
from .model import XLMRobertaSiamese
from .util import mean_encoded_seq_batch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lambda-weight', type=float, default=.7)
    parser.add_argument('--alpha', type=float, default=.3)
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--encoder-name', type=str, default='xlm-roberta-large')
    parser.add_argument('--reload-from-files', action='store_true')
    parser.add_argument('-s', '--sample', type=int)
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()

args = parse_args()

if args.reload_from_files:
    path = Path(args.output)
    with open(path/'hp.json') as f: 
        args = argparse.Namespace(**{**args.__dict__, **json.load(f)})
    args.reload_from_files = True
else:
    root = Path(args.output)
    path = root/datetime.now().strftime('%Y%m%d')/'pretrain'
    path.mkdir(parents=True, exist_ok=True)
    with open(path/'hp.json', 'w') as f: 
        json.dump(args.__dict__, f, indent=4)


device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
device_count = (torch.cuda.device_count() or 1) if args.cuda else 1

tokenizer = XLMRobertaTokenizer.from_pretrained(args.encoder_name)

class CustomTokenizeProcessor(TokenizeProcessor):
    def process(self, items): 
        return self.tokenizer(items)['input_ids']

if args.sample:
    print('Training on a sample of %i examples from the training data.' % args.sample)
    # Train on a sample of the full training data.
    # This is for fast iteration or hyperparam tuning.
    import pandas as pd
    n = sum(1 for line in open(args.input)) # Get total number of training examples
    s = args.sample # Desired sample size
    skip = sorted(random.sample(range(n),n-s)) # Skip a random sample of n-s examples with pandas' read_csv method
    df = pd.read_csv(args.input, skiprows=skip, names=['source_text', 'source_lang', 'source_title', 'target_text', 'target_lang', 'target_title'], sep='\t')
    dataset = TabularDataset.from_df(df)
else:
    dataset = TabularDataset.from_csv(args.input, names=['source_text', 'source_lang', 'source_title', 'target_text', 'target_lang', 'target_title'])

# Processors are applied to fields in the tabular dataset
tok = CustomTokenizeProcessor(partial(tokenizer.batch_encode_plus, add_special_tokens=True, max_length=args.max_length, return_token_type_ids=False))
# The mask processor will return tuples of masked input (where 15% of tokens are either masked or swapped out) 
# and target output where unmasked tokens are ignored
mask = MaskProcessor(tokenizer.mask_token_id, tokenizer.vocab_size)
lang_vocab = list(set(dataset.df['source_lang'].unique().tolist()+dataset.df['target_lang'].tolist())) # Share vocab between source and target langs
cat = CategoryProcessor(vocab=lang_vocab, min_freq=0)

proc_x = {
    'source_text': [tok,mask], # tokenize and mask source text
    'target_text': [tok,mask], # tokenize and mask source text
    'source_lang': cat, # convert source langs to integers
    'target_lang': cat # convert target langs to integers
}

def pad_collate(samples, pad_idx=1, ignore_index=-100, dtype=torch.int64):
    '''
    Takes in a batch of input-target pairs and pads them appropriately.
    `samples` looks like [([(x_a,y_a),(x_p,y_p),x_al,x_pl],)]
    '''
    max_len = 0
    x_as, y_as, x_ps, y_ps = [], [], [], []
    for x,_ in samples:
        (x_a,y_a),(x_p,y_p),*_ = x
        max_len = max(max_len, len(x_a), len(x_p))
        x_as.append(x_a)
        y_as.append(y_a)
        x_ps.append(x_p)
        y_ps.append(y_p)
    xs = x_as + x_ps
    ys = y_as + y_ps
    bs = len(x)
    inps = torch.zeros(bs, max_len, dtype=dtype)+pad_idx
    outps = torch.zeros(bs, max_len, dtype=dtype)+ignore_index
    for i,(x,y) in enumerate(zip(xs,ys)):
        inps[i,:len(x)] = torch.tensor(x, dtype=dtype)
        outps[i,:len(y)] = torch.tensor(y, dtype=dtype)
    return inps, outps

if args.reload_from_files:
    # Reload from already processed data
    with open(path/'data.json') as i: 
        data = DataBunch.from_serializable( \
            json.load(i), \
            bs=args.batch_size/2, \
            collate_func=partial(pad_collate, pad_idx=tokenizer.pad_token_id), \
            train_sampler=partial(MultinomialSampler, alpha=args.alpha), valid_sampler=SequentialSampler \
        )
else:
    # This splits between train and valid sets
    # then processes the inputs
    # then creates data loaders and applies the samplers, 
    # collates the batches, and pads them appropriately.
    data = dataset \
        .split() \
        .label(proc_x=proc_x) \
        .to_databunch( \
            bs=args.batch_size/2, \
            collate_func=partial(pad_collate, pad_idx=tokenizer.pad_token_id), \
            train_sampler=partial(MultinomialSampler, alpha=args.alpha), valid_sampler=SequentialSampler \
        )

    with open(path/'data.json', 'w') as o: 
        json.dump(data.to_serializable(), o) # Save processed data

# Instantiate model
model = XLMRobertaSiamese(model_name=args.encoder_name, pad_ix=tokenizer.pad_token_id)
if args.cuda and device_count > 1: model = nn.DataParallel(model)

# Here we define `mlm_loss`, `_triplet_loss`, `_seq_acc`, and `_perplexity` as arbitrary metrics
# to track during training. 
# Tracking `mlm_loss` and `_triplet_loss` is to eyeball the range of 
# mlm loss v triplet loss in order to calibrate λ. 
# Tracking `_seq_acc` and `_perplexity` is to get a sense of the accuracy improvement of the model
# during training.

ce = nn.CrossEntropyLoss()
def mlm_loss(outp, target):
    logits = outp[0]
    vocab_size = logits.size(-1)
    return ce(logits.view(-1, vocab_size), target.view(-1))

triplet_loss = TripletLoss(margin=args.margin)
def _triplet_loss(outp, target):
    reps = outp[1]
    return triplet_loss(*reps)

def _seq_acc(outp, target):
    logits = outp[0]
    return seq_acc(logits, target)

def _perplexity(outp, target):
    logits = outp[0]
    return perplexity(logits, target)

# Here we define all the callbacks that will be called during the training loop
cbfs = [
    FetchHardestNegatives,
    partial(AvgStatsCallback, [_seq_acc, _perplexity, mlm_loss, _triplet_loss], path=path/'metrics.tsv'),
    partial(CudaCallback, device),
    partial(SaveModel, path/'model.pth'),
    EarlyStop
]

opt = partial(Adam, lr=args.learning_rate) # Good old Adam optimizer
loss_func = SiameseLoss(λ=args.lambda_weight, margin=args.margin) # Loss to backprop on
# Learner object, which implements the training loop and calls all the callbacks
learn = Learner(model, data, loss_func, lr=args.learning_rate, cb_funcs=cbfs, opt_func=opt)

print('Training model on {device_count} {device_type}(s)'.format(device_count=device_count, device_type=device))
learn.fit(args.num_epochs)