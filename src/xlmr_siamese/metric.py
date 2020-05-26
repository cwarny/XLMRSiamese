import torch.nn as nn
from deeper_nlu.train import TripletLoss
from deeper_nlu.metric import seq_acc, perplexity

ce = nn.CrossEntropyLoss()
def mlm_loss(outp, target):
    logits = outp[0]
    vocab_size = logits.size(-1)
    return ce(logits.view(-1, vocab_size), target.view(-1))

def _triplet_loss(outp, target, *args, **kwargs):
    triplet_loss = TripletLoss(*args, **kwargs)
    reps = outp[1]
    return triplet_loss(*reps)

def _seq_acc(outp, target):
    logits = outp[0]
    return seq_acc(logits, target)

def _perplexity(outp, target):
    logits = outp[0]
    return perplexity(logits, target)