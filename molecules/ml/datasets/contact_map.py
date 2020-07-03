import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5

class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """
    def __init__(self, path, split_ptc=0.8, split='train', squeeze=False):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.

        squeeze : bool
            If True, data is reshaped to (H, W) else if False data
            is reshaped to (1, H, W).
        """
        if split not in ('train', 'valid'):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError('Parameter split_ptc must satisfy 0 <= split_ptc <= 1.')

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        h5_file = open_h5(path)
        # contact_maps dset has shape (N, W, H, 1)
        self.dset = h5_file['contact_maps']
 
        # train validation split index
        self.split_ind = int(split_ptc * len(self.dset))
        self.split = split

        if squeeze:
            self.shape = (self.dset.shape[1], self.dset.shape[2])
        else:
            self.shape = (1, self.dset.shape[1], self.dset.shape[2])

    def __len__(self):
        if self.split == 'train':
            return self.split_ind
        return len(self.dset) - self.split_ind

    def __getitem__(self, idx):
        import time; start = time.time()
        if self.split == 'valid':
            idx += self.split_ind
        return torch.from_numpy(np.array(self.dset[idx]) \
                 .reshape(self.shape)).to(torch.float32)
