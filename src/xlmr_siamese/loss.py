import torch
import torch.nn as nn
from deeper_nlu.train import TripletLoss

ce = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=1)
mse = nn.MSELoss()

def dual_loss(outp, target):
    r1,r2 = outp
    return mse(cos(r1,r2), target)

class SiameseLoss(nn.Module):
    def __init__(self, λ=.7, *args, **kwargs):
        super().__init__()
        self.λ = λ
        self.triplet_loss = TripletLoss(*args, **kwargs)
    
    def forward(self, outp, target):
        logits, reps = outp # `logits` is the logits for mlm, `reps` is the vector representation of the a,p,n seqs
        vocab_size = logits[0].size(-1)
        ce_loss = sum(map(lambda e: ce(e[0].view(-1, vocab_size), e[1].view(-1)), zip(logits,target)))/len(logits)
        return (1-self.λ)*ce_loss + self.λ*self.triplet_loss(*reps)

# def loss(outp, target, λ=.5):
#     (p1, p2), r = outp # p is the logits for mlm, r is the vector representation of the seq
#     bs, sl, vocab_size = p1.shape
#     t, l1, l2 = target # t is whether this is a pos or neg pair
#     return (1-λ)/2*(ce(p1.view(-1, vocab_size), l1.view(-1)) + ce(p2.view(-1, vocab_size), l2.view(-1))).view(bs) + λ*dual_loss(r, t)

