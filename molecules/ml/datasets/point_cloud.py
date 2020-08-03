import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5

class PointCloudDataset(Dataset):
    """
    PyTorch Dataset class to load point clouds data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """
    def __init__(self, path, num_points, num_features, split_ptc=0.8,
                 split='train'):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        num_points : int
            Number of points per sample. Should be smaller or equal than the total number of points.

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.
        """
        if split not in ('train', 'valid'):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError('Parameter split_ptc must satisfy 0 <= split_ptc <= 1.')

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        h5_file = open_h5(path)

        # get point clouds
        self.dset = h5_file['point_clouds']
        self.len = len(self.dset)
 
        # train validation split index
        self.split_ind = int(split_ptc * self.len)
        self.split = split
        self.num_points = num_points
        self.num_features = num_features
        self.num_points_total = self.dset.shape[2]

        # sanity checks
        assert (self.num_points_total >= self.num_points)
        assert (self.dset.shape[1] == (3 + self.num_features))

        # rng
        self.rng = np.random.default_rng()

        # create temp buffer for IO
        self.token = np.zeros((1, 3 + self.num_features, self.num_points), dtype = np.float32)
        
    def __len__(self):
        if self.split == 'train':
            return self.split_ind
        return self.len - self.split_ind

    def __getitem__(self, idx):
        if self.split == 'valid':
            idx += self.split_ind

        if self.num_points < self.num_points_total:
            # select points to read
            indices = self.rng.choice(self.num_points_total, size = self.num_points,
                                      replace = False, shuffle = False)

            # read
            self.token[0, ...] = self.dset[idx, :, indices]
        else:
            self.dset.read_direct(self.token,
                                  np.s_[idx:idx+1, 0:(3 + self.num_features), 0:self.num_points],
                                  np.s_[0:1, 0:(3 + self.num_features), 0:self.num_points])
            
        return torch.squeeze(torch.as_tensor(self.token), dim=0)

