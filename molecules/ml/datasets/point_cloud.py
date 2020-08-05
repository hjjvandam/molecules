import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5


class PointCloudDataset(Dataset):
    """
    PyTorch Dataset class to load point clouds data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """
    def __init__(self, path, dataset_name, rmsd_name,
                 num_points, num_features, split_ptc=0.8,
                 split='train', normalize = True):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        dataset_name : str
            Name of the point cloud data in the HDF5 file.

        rmsd_name : str
            Name of the RMSD in the HDF5 file.  

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
        self.file_path = path
        self.h5_file = open_h5(self.file_path)

        # get point clouds
        self.dataset_name = dataset_name
        self.rmsd_name = rmsd_name
        self.dset = self.h5_file[self.dataset_name]
        self.len = self.dset.shape[0]
        
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

        # normalize input
        self.normalize = normalize
        if self.normalize:
            self.bias = self.dset[...].min(axis = (0,2))
            self.scale = 1. / (self.dset[...].max(axis = (0,2)) - self.bias)
            # broadcast shapes
            self.bias = np.tile(np.expand_dims(self.bias, axis = -1), (1, self.num_points))
            self.scale = np.tile(np.expand_dims(self.scale, axis = -1), (1, self.num_points))
        else:
            self.bias = np.zeros((3 + self.num_features, self.num_points), dtype = np.float32)
            self.scale = np.ones((3 + self.num_features, self.num_points), dtype = np.float32)

        # close and reopen later
        self.dset = None
        self.h5_file.close()
        self.init = False
            
    def __len__(self):
        if self.split == 'train':
            return self.split_ind
        return self.len - self.split_ind

    def __getitem__(self, idx):
        # init if necessary
        if not self.init:
            self.h5_file = open_h5(self.file_path)
            self.dset = self.h5_file[self.dataset_name]
            self.rmsd = self.h5_file[self.rmsd_name]
            self.init = True
        
        if self.split == 'valid':
            idx += self.split_ind

        self.dset.id.refresh()
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
            #self.token[0, ...] = self.dset[idx:idx+1, ...]
        
        # normalize
        result = (self.token[0, ...] - self.bias) * self.scale
        rmsd = self.rmsd[idx, 2]
                    
        return torch.tensor(result, requires_grad = False), torch.tensor(rmsd, requires_grad = False)

