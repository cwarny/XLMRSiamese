import torch
import torch.nn as nn
from deeper_nlu.train import TripletLoss

ce = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=1)
mse = nn.MSELoss()

class SiameseLoss(nn.Module):
    def __init__(self, λ=.7, *args, **kwargs):
        super().__init__()
        self.λ = λ
        self.triplet_loss = TripletLoss(*args, **kwargs)
    
    def forward(self, outp, target):
        logits, reps = outp # `logits` is the logits for mlm, `reps` is the vector representation of the a,p,n seqs
        vocab_size = logits.size(-1)
        ce_loss = ce(logits.view(-1, vocab_size), target.view(-1))
        return (1-self.λ)*ce_loss + self.λ*self.triplet_loss(*reps)

def dual_loss(u, v, target):
    return mse(cos(u,v), target)

def mlm_dual_loss(outp, target, λ=.7):
    logits, reps = outp
    *labels,is_positive = target
    vocab_size = logits.size(-1)
    ce_loss = ce(logits.view(-1, vocab_size), labels.view(-1))
    return (1-λ)*ce_loss + λ*dual_loss(*reps, is_positive)