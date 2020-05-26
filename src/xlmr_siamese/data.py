import torch
from torch.utils.data import Dataset

class DatasetFromSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

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