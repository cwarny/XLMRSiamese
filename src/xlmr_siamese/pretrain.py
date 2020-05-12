import argparse
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import XLMRobertaTokenizer
from deeper_nlu.data import TabularDataset, CategoryProcessor, MaskProcessor, TokenizeProcessor, DataBunch
from deeper_nlu.util import compose
from deeper_nlu.train import Learner, AvgStatsCallback, CudaCallback, SaveModel, EarlyStop, Callback
from deeper_nlu.metric import seq_acc, perplexity
from deeper_nlu.train import TripletLoss
from functools import partial
from model import XLMRobertaSiamese
from util import mean_encoded_seq_batch
import numpy as np
import random
from collections import defaultdict
from torch.utils.data import Sampler, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from loss import SiameseLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lambda-weight', type=float, default=.7)
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--pos-batch-size', type=int, default=16)
    parser.add_argument('--neg-batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--encoder-name', type=str, default='xlm-roberta-large')
    parser.add_argument('--reload-from-files', action='store_true')
    parser.add_argument('-s', '--sample', type=int)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_count = torch.cuda.device_count() or 1

tokenizer = XLMRobertaTokenizer.from_pretrained(args.encoder_name)

class CustomTokenizeProcessor(TokenizeProcessor):
    def process(self, items): 
        return self.tokenizer(items)['input_ids']

if args.sample:
    import pandas as pd
    n = sum(1 for line in open(args.input))
    s = args.sample
    skip = sorted(random.sample(range(1,n+1),n-s))
    df = pd.read_csv(args.input, skiprows=skip, names=['source_text', 'source_lang', 'source_title', 'target_text', 'target_lang', 'target_title'], sep='\t')
    dataset = TabularDataset.from_df(df)
else:
    dataset = TabularDataset.from_csv(args.input, names=['source_text', 'source_lang', 'source_title', 'target_text', 'target_lang', 'target_title'])

tok = CustomTokenizeProcessor(partial(tokenizer.batch_encode_plus, add_special_tokens=True, max_length=512, return_token_type_ids=False))
mask = MaskProcessor(tokenizer.mask_token_id, tokenizer.vocab_size)
lang_vocab = list(set(dataset.df['source_lang'].unique().tolist()+dataset.df['target_lang'].tolist())) # Share vocab between source and target langs
cat = CategoryProcessor(vocab=lang_vocab, min_freq=0)

proc_x = {
    'source_text': [tok,mask],
    'target_text': [tok,mask],
    'source_lang': cat,
    'target_lang': cat
}

def pad_collate(samples, pad_idx=1, ignore_index=-100, dtype=torch.int64):
    '''
    Takes in a batch of input-target pairs and pads them appropriately.
    `samples` looks like [([(x_a,y_a),(x_p,y_p),x_al,x_pl],)]
    '''
    max_source_len, max_target_len = 0, 0
    for x,_ in samples:
        (x_a,y_a),(x_p,y_p),*_ = x
        max_source_len = max(len(x_a), max_source_len)
        max_target_len = max(len(x_p), max_target_len)
    bs = len(samples)
    inps = list(map(lambda max_len: torch.zeros(bs, max_len, dtype=dtype)+pad_idx, (max_source_len, max_target_len)))
    outps = list(map(lambda max_len: torch.zeros(bs, max_len, dtype=dtype)+ignore_index, (max_source_len, max_target_len)))
    for i,(x,_) in enumerate(samples):
        (x_a,y_a),(x_p,y_p),*_ = x
        inps[0][i, :len(x_a)] = torch.tensor(x_a, dtype=dtype)
        inps[1][i, :len(x_p)] = torch.tensor(x_p, dtype=dtype)
        outps[0][i, :len(y_a)] = torch.tensor(y_a, dtype=dtype)
        outps[1][i, :len(y_p)] = torch.tensor(y_p, dtype=dtype)
    return inps, outps

class MultinomialSampler(Sampler):
    '''
    This sampler samples proportionally to the frequency of pos lang pairs
    while adjusting the multinomial distribution with the alpha param.
    '''
    def __init__(self, data, alpha=.3):
        partition = defaultdict(list)
        for i,(x,_) in enumerate(data): 
            *_,x_al,x_pl = x
            partition[(x_al,x_pl)].append(i) # partition by pos lang pairs
        keys = list(partition.keys())
        counts = np.array([len(partition[k]) for k in keys]) # get the freq for each lang pair
        ps = counts/counts.sum() # normalize counts
        ps_pow = ps**alpha # adjust the distribution
        qs = ps_pow/ps_pow.sum() # get the final weights for the multinomial
        sample_sizes = qs*len(data) # sample size for each lang pair
        idxs = []
        for k,sample_size in zip(keys, sample_sizes):
            # Within each partition, randomly pick the target number of samples
            idxs.extend(np.random.choice(partition[k], replace=True, size=int(sample_size)))
        random.shuffle(idxs) # Give it a good shuffle
        self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __iter__(self): return iter(self.idxs)

if args.reload_from_files:
    with open(path/'data.json') as i: 
        data = DataBunch.from_serializable( \
            json.load(i), \
            bs=args.pos_batch_size, \
            collate_func=partial(pad_collate, pad_idx=tokenizer.pad_token_id), \
            train_sampler=MultinomialSampler, valid_sampler=SequentialSampler \
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
            bs=args.pos_batch_size, \
            collate_func=partial(pad_collate, pad_idx=tokenizer.pad_token_id), \
            train_sampler=MultinomialSampler, valid_sampler=SequentialSampler \
        )

    with open(path/'data.json', 'w') as o: json.dump(data.to_serializable(), o)

# Instantiate model
model = XLMRobertaSiamese(model_name=args.encoder_name, pad_ix=tokenizer.pad_token_id)
if device_count > 1: model = nn.DataParallel(model)

class FetchHardestNegatives(Callback):
    '''
    This callback will be called at the beginning of each batch.
    For each anchor-pos pair, it will fetch `neg_batch_size` random sentences ("negative candidates"),
    find the negative candidates closest to each anchor ("hardest negatives")
    and line up those hardest negatives with the anchor-pos pairs to get the triples
    '''
    def __init__(self, x_p, y_p, lengths, neg_batch_size=16):
        self.x_p, self.y_p = x_p, y_p
        self.lengths = lengths
        self.neg_batch_size = neg_batch_size
        super().__init__()

    def begin_batch(self):
        xb_a,xb_p = self.xb
        # Get indices for negative candidates by randomly picking a subset of target seqs
        i = torch.randperm(self.x_p.size(0))[:self.neg_batch_size]
        # Grab neg candidates
        xb_nc = self.x_p[i] 
        yb_nc = self.y_p[i]
        max_len = self.lengths[i].max()
        # Trim unnecessary padding
        xb_nc = xb_nc[:,:max_len]
        yb_nc = yb_nc[:,:max_len]
        # Encode anchors and negs to then compute pairwise distances
        with torch.no_grad():
            xb_a_enc = mean_encoded_seq_batch(model.mlm.roberta(xb_a)[0], xb_a, ignore_index=tokenizer.pad_token_id)
            xb_nc_enc = mean_encoded_seq_batch(model.mlm.roberta(xb_nc)[0], xb_nc, ignore_index=tokenizer.pad_token_id)
        # Compute pairwise distances between every seq in `a_enc` and every seq in `nc_enc`, 
        # then find the negs closest to the anchors ("hardest negs")
        n_i = torch.cdist(xb_a_enc, xb_nc_enc).argmin(1) # n_i.size(0) == xb_a.size(0)
        xb_n = xb_nc[n_i]
        yb_n = yb_nc[n_i]
        self.run.xb = (xb_a,xb_p,xb_n)
        self.run.yb = (*self.yb,yb_n) # Add neg targets for MLM

X_p, Y_p = zip(*compose(dataset.df['target_text'], [tok,mask])) # idem for pos seqs
x_p = pad_sequence(list(map(torch.LongTensor, X_p)), batch_first=True, padding_value=tokenizer.pad_token_id)
y_p = pad_sequence(list(map(torch.LongTensor, Y_p)), batch_first=True, padding_value=-100)
lengths = torch.LongTensor(list(map(len, Y_p)))

ce = nn.CrossEntropyLoss()

def mlm_loss(outp, target):
    logits, reps = outp
    vocab_size = logits[0].size(-1)
    ce_loss = sum(map(lambda e: ce(e[0].view(-1, vocab_size), e[1].view(-1)), zip(logits,target)))/len(logits)
    return ce_loss

triplet_loss = TripletLoss(margin=args.margin)
def _triplet_loss(outp, target):
    logits, reps = outp
    return triplet_loss(*reps)

def _seq_acc(outp, target):
    logits, reps = outp
    return sum(map(lambda e: seq_acc(e[0], e[1]), zip(logits, target)))/len(logits)

def _perplexity(outp, target):
    logits, reps = outp
    return sum(map(lambda e: perplexity(e[0], e[1]), zip(logits, target)))/len(logits)

cbfs = [
    partial(FetchHardestNegatives, x_p, y_p, lengths, neg_batch_size=args.neg_batch_size),
    partial(AvgStatsCallback, [_seq_acc, _perplexity, mlm_loss, _triplet_loss], path=path/'metrics.tsv'),
    partial(CudaCallback, device),
    partial(SaveModel, path/'model.pth'),
    EarlyStop
]

opt = partial(Adam, lr=args.learning_rate)
loss_func = SiameseLoss(λ=args.lambda_weight, margin=args.margin)
learn = Learner(model, data, loss_func, lr=args.learning_rate, cb_funcs=cbfs, opt_func=opt)

print('Training model on {device_count} {device_type}(s)'.format(device_count=device_count, device_type=device))
learn.fit(args.num_epochs)