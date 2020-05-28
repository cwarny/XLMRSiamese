import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from deeper_nlu.train import Callback
from deeper_nlu.util import listify

from xlmr_siamese.sampler import DistributedSamplerWrapper

try:
    from apex import amp
    _has_apex = True
except ImportError:
    _has_apex = False

def is_apex_available(): return _has_apex

# A callback needs to overwrite methods that will be called at specific moments in the 
# training loop. In this case, the callback will be called after prediction (`after_pred`).
# The callback has access to any object defined in the training loop via `self.<object name>`.
# Here, after prediction, we have access to `self.pred`, i.e. the model output.
class FetchHardestNegatives(Callback):
    '''
    This callback fetches negative translations from within the batch
    by picking the target sequence closest to the anchor without
    being the actual translation.
    '''
    def after_pred(self):
        logits, reps = self.pred
        # Compute pairwise distances between every anchor seq and every positive seq.
        # We'll pick negatives based on that distance matrix.
        bs = reps.size(0)
        a = reps[:bs//2] # anchors
        p = reps[bs//2:] # positives
        dist = torch.cdist(a,reps)
        dist[range(bs//2),range(bs//2)] = float('inf') # remove self-pairs
        dist[range(bs//2),range(bs//2,bs)] = float('inf') # remove positive pairs
        # Pick as negatives the target seqs closest to the anchors ("hardest negs")
        # without them being the actual translations or themselves
        i = dist.argmin(1)
        n = reps[i]
        self.run.pred = (logits, (a,p,n))

def check_apex():
    if not is_apex_available():
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

class FP16Callback(Callback):
    _order = 0
    def begin_fit(self):
        check_apex()
        self.run.model, self.run.opt = amp.initialize(self.model, self.opt, opt_level='O2')
    
    def after_loss(self):
        check_apex()
        with amp.scale_loss(self.loss, self.opt) as scaled_loss: 
            self.run.loss = scaled_loss

class DistributedTrainingCallback(Callback):
    _order = 1
    def __init__(self, device_ids):
        self.device_ids = listify(device_ids)

    def begin_fit(self):
        self.run.model = DDP(self.model, device_ids=self.device_ids)
        self.run.data.train_dl = DataLoader(
            self.run.data.train_dl.dataset,
            batch_size=self.run.data.train_dl.batch_size,
            sampler=DistributedSamplerWrapper(self.run.data.train_dl.sampler),
            collate_fn=self.run.data.train_dl.collate_fn
        )
        self.run.data.valid_dl = DataLoader(
            self.run.data.valid_dl.dataset,
            batch_size=self.run.data.valid_dl.batch_size,
            sampler=DistributedSamplerWrapper(self.run.data.valid_dl.sampler),
            collate_fn=self.run.data.valid_dl.collate_fn
        )