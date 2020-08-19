import argparse
import torch
import torch.nn as nn
from transformers import XLMRobertaForMaskedLM
from torch.optim import Adam
from deeper_nlu.train import TripletLoss

try:
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False

def is_apex_available(): return _has_apex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=str, choices=['cpu','cuda'], default='cuda')
    return parser.parse_args()

args = parse_args()

ce = nn.CrossEntropyLoss()

class SiameseLoss(nn.Module):
    def __init__(self, λ=.7, *args, **kwargs):
        super().__init__()
        self.λ = λ
        self.triplet_loss = TripletLoss(*args, **kwargs)
    
    def forward(self, outp, target):
        logits, reps = outp # `logits` is the logits for mlm, `reps` is the vector representation of the a,p,n seqs
        print('Batch size in loss: %i' % logits.size(0))
        vocab_size = logits.size(-1)
        ce_loss = ce(logits.view(-1, vocab_size), target.view(-1))
        return (1-self.λ)*ce_loss + self.λ*self.triplet_loss(*reps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
    def forward(self, x):
        print('Batch size in model: %i' % x.size(0))
        representations = self.mlm.roberta(x)[0]
        logits = self.mlm.lm_head(representations)
        representations = representations.mean(1)
        return logits, representations

vocab_size = 250002
device = torch.device(args.device)
device_count = torch.cuda.device_count()
model = Model()

opt = Adam(model.parameters(), lr=1e-3)
model.to(device)

if args.fp16:
    if not is_apex_available():
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, opt = amp.initialize(model, opt, opt_level='O1')

if args.parallel: model = nn.DataParallel(model)

loss_func = SiameseLoss(margin=1.)
if args.parallel: loss_func = nn.DataParallel(loss_func)

def get_batch():
    xb = torch.randint(0,vocab_size,(args.batch_size*device_count,args.max_length), dtype=torch.int64).to(device)
    yb = torch.randint(0,vocab_size,(args.batch_size*device_count,args.max_length), dtype=torch.int64).to(device)
    return xb,yb

xb,yb = get_batch()
logits,reps = model(xb)
print('Batch size outside: %i' % reps.size(0))
bs = reps.size(0)
a = reps[:bs//2] # anchors
p = reps[bs//2:] # positives
dist = torch.cdist(a,reps)
dist[range(bs//2),range(bs//2)] = float('inf') # remove self-pairs
dist[range(bs//2),range(bs//2,bs)] = float('inf') # remove positive pairs
i = dist.argmin(1)
n = reps[i]
outp = (logits,(a,p,n))
loss = loss_func(outp, yb)
if args.fp16:
    with amp.scale_loss(loss, opt) as scaled_loss: scaled_loss.backward(torch.ones(loss.size(0)).to(device))
else: loss.backward(torch.ones(loss.size(0)).to(device))
opt.step()
opt.zero_grad()