# General imports
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
from functools import partial
import random
from collections import defaultdict

# PyTorch imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import SequentialSampler
import torch.distributed as dist
import torch.multiprocessing as mp

# HuggingFace's transformers lib
from transformers import XLMRobertaTokenizer

# DeeperNLU dependencies
from deeper_nlu.data import TabularDataset, CategoryProcessor, MaskProcessor, TokenizeProcessor, DataBunch
from deeper_nlu.util import compose
from deeper_nlu.train import Learner, AvgStatsCallback, CudaCallback, SaveModel, EarlyStop

# Internal dependencies
from xlmr_siamese.sampler import MultinomialSampler
from xlmr_siamese.loss import SiameseLoss
from xlmr_siamese.model import XLMRobertaSiamese
from xlmr_siamese.util import mean_encoded_seq_batch
from xlmr_siamese.data import pad_collate
from xlmr_siamese.callback import FetchHardestNegatives, FP16Callback, DistributedTrainingCallback
from xlmr_siamese.metric import mlm_loss, _triplet_loss, _seq_acc, _perplexity

def parse_args():
    parser = argparse.ArgumentParser()
    
    # I/O
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-s', '--sample', type=int)
    
    # Hyperparams
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lambda-weight', type=float, default=.7)
    parser.add_argument('--alpha', type=float, default=.3)
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--encoder-name', type=str, default='xlm-roberta-large')
    
    # Distributed training
    parser.add_argument('--nodes', type=int, default=1, metavar='N')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--node-rank', type=int, default=0, help='Rank amongst the nodes')
    parser.add_argument('--master-addr', type=str, default='localhost')
    parser.add_argument('--master-port', type=str, default='12355')
    
    # Lower-precision training
    parser.add_argument('--fp16', action='store_true')
    
    return parser.parse_args()

class CustomTokenizeProcessor(TokenizeProcessor):
    def process(self, items): 
        return self.tokenizer(items)['input_ids']

def train(gpu, args):
    rank = args.node_rank*args.gpus + gpu
    print(f'Running training on rank {rank}.')
    if args.parallel:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group('nccl', world_size=args.world_size, rank=rank)

    torch.manual_seed(0)

    # create model and move it to GPU with id rank
    torch.cuda.set_device(gpu)
    model = XLMRobertaSiamese(model_name=args.encoder_name, pad_ix=args.tokenizer.pad_token_id).to(gpu)
    opt = partial(Adam, lr=args.learning_rate) # Good old Adam optimizer
    loss_func = SiameseLoss(λ=args.lambda_weight, margin=args.margin) # Loss to backprop on

    # This splits between train and valid sets
    # then processes the inputs
    # then creates data loaders and applies the samplers, 
    # collates the batches, and pads them appropriately.
    db = args.data.to_databunch(
        bs=args.batch_size//2,
        collate_func=partial(pad_collate, pad_idx=args.tokenizer.pad_token_id),
        train_sampler=partial(MultinomialSampler, alpha=args.alpha), 
        valid_sampler=SequentialSampler
    )

    # Here we define all the callbacks that will be called during the training loop
    cbfs = [
        FetchHardestNegatives,
        partial(CudaCallback, gpu)
    ]
    if args.fp16: cbfs.append(FP16Callback)
    if args.parallel: cbfs.append(partial(DistributedTrainingCallback, gpu))

    metrics = [
        _seq_acc, 
        _perplexity, 
        mlm_loss, 
        partial(_triplet_loss, margin=args.margin)
    ]
    
    if rank == 0:
        cbfs.extend([
            partial(AvgStatsCallback, metrics, path=args.path/'metrics.tsv'),
            partial(SaveModel, args.path/'model.pth'),
            EarlyStop
        ])
    
    learn = Learner(model, db, loss_func, lr=args.learning_rate, cb_funcs=cbfs, opt_func=opt)
    learn.fit(args.num_epochs)

    if args.parallel: dist.destroy_process_group()

def main():
    assert torch.cuda.is_available(), "Please run on CUDA-enabled hardware"

    args = parse_args()

    if args.gpus > 0:
        device_count = torch.cuda.device_count()
        assert args.gpus <= device_count, f"You tried to use {args.gpus} gpus but only {device_count} available"

    root = Path(args.output)
    path = root/datetime.now().strftime('%Y%m%d')/'pretrain'
    path.mkdir(parents=True, exist_ok=True)
    with open(path/'hp.json', 'w') as f: 
        json.dump(args.__dict__, f, indent=4)
    args.path = path

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.encoder_name)
    args.tokenizer = tokenizer

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

    data = dataset.split().label(proc_x=proc_x)
    args.data = data
    
    args.parallel = not (args.nodes <= 1 and args.gpus <= 1)
    if args.parallel:
        args.world_size = args.gpus*args.nodes
        mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
    else:
        train(0, args)

if __name__ == '__main__':
    main()