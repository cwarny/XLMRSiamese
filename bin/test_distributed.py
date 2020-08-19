import argparse
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import XLMRobertaForMaskedLM
from deeper_nlu.train import TripletLoss

try:
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False

def is_apex_available(): return _has_apex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1, metavar='N')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs per node')
    parser.add_argument('--node-rank', type=int, default=0, help='Rank amongst the nodes')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()

vocab_size = 250002

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlm = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-large')
    
    def forward(self, x):
        print('Batch size in model: %i' % x.size(0))
        print(x.device)
        representations = self.mlm.roberta(x)[0]
        logits = self.mlm.lm_head(representations)
        representations = representations.mean(1)
        return logits, representations
    
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
        print(logits.device)
        ce_loss = ce(logits.view(-1, vocab_size), target.view(-1))
        return (1-self.λ)*ce_loss + self.λ*self.triplet_loss(*reps)

def get_batch(bs, max_len, device):
    xb = torch.randint(0,vocab_size,(bs,max_len), dtype=torch.int64).to(device)
    yb = torch.randint(0,vocab_size,(bs,max_len), dtype=torch.int64).to(device)
    return xb,yb

def train(gpu, args):
    print(f'GPU: {gpu}')
    rank = args.node_rank*args.gpus + gpu
    print(f'Running training on rank {rank}.')
    if args.parallel:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    # create model and move it to GPU with id rank
    torch.cuda.set_device(gpu)
    model = Model().to(gpu)
    loss_func = SiameseLoss(margin=1.)
    opt = Adam(model.parameters(), lr=1e-3)
    if args.fp16:
        if not is_apex_available():
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, opt = amp.initialize(model, opt, opt_level='O2')
    if args.parallel:
        model = DDP(model, device_ids=[gpu])
    opt.zero_grad()
    xb,yb = get_batch(args.batch_size, args.max_length, gpu)
    print(xb.device)
    logits,reps = model(xb)
    bs = reps.size(0)
    a = reps[:bs//2] # anchors
    p = reps[bs//2:] # positives
    d = torch.cdist(a,reps)
    d[range(bs//2),range(bs//2)] = float('inf') # remove self-pairs
    d[range(bs//2),range(bs//2,bs)] = float('inf') # remove positive pairs
    i = d.argmin(1)
    n = reps[i]
    outp = (logits,(a,p,n))
    loss = loss_func(outp, yb)
    print(loss.shape)
    print(loss.device)
    print('Backward pass')
    if args.fp16:
        with amp.scale_loss(loss, opt) as scaled_loss: scaled_loss.backward()
    else:
        loss.backward()
    print('Update weights')
    opt.step()
    if gpu == 0: print('Training complete')
    dist.destroy_process_group()

def main():
    args = parse_args()
    args.world_size = args.gpus*args.nodes
    if args.parallel:
        mp.spawn(train, nprocs=args.gpus, args=(args,), join=True)
    else:
        train(0, args)

if __name__ == '__main__':
    main()