from .data import DatasetFromSampler
from torch.utils.data import DistributedSampler, Sampler
from collections import defaultdict
from operator import itemgetter
import numpy as np
import random

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

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