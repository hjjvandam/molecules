import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5


class PointCloudInMemoryDataset(Dataset):
    """
    PyTorch Dataset class to load point clouds data. Reads the relevant chunk 
    of the HDF5 file into DRAM or GPU memory.
    
    """
    def __init__(self, path, dataset_name, rmsd_name,
                 fnc_name, num_points, num_features,
                 split_ptc=0.8, split = 'train', 
                 shard_id = 0, num_shards = 1, seed = 333,
                 normalize = 'box', cms_transform = False):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing data set.

        dataset_name : str
            Name of the point cloud data in the HDF5 file.

        rmsd_name : str
            Name of the RMSD in the HDF5 file.  

        fnc_name : str
            Name of the fraction of native contact dset in the HDF5 file.

        num_points : int
            Number of points per sample. Should be smaller or equal than the total number of points.

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.

        shard_id: int
            Specify the id of the dataset shard. Useful for DDP mode.
                 
        num_shards: int
            Specify the number of shards to generate.

        seed : int
            Seed for the RNG for the splitting. Make sure it is the same for all workers reading
            from the same file.
        """
        if split not in ('train', 'valid'):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError('Parameter split_ptc must satisfy 0 <= split_ptc <= 1.')

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        self.file_path = path
        self.dataset_name = dataset_name
        self.rmsd_name = rmsd_name
        self.fnc_name = fnc_name
        self.num_points = num_points
        self.num_features = num_features
        self.normalize = normalize
        self.cms_transform = cms_transform

        # get size of the dataset
        with open_h5(self.file_path, 'r', libver = 'latest', swmr = False) as f:
            
            # get dataset
            dset = f[self.dataset_name]
            rmsd = f[self.rmsd_name]
            fnc = f[self.fnc_name]

            # get lengths
            self.len = dset.shape[0]
            self.num_features_total = dset.shape[1]
            self.num_points_total = dset.shape[2]

            # do splitting
            self.split_ind = int(split_ptc * self.len)
            self.split = split
            split_rng = np.random.default_rng(seed)
            self.indices = split_rng.permutation(list(range(self.len)))
            if self.split == "train":
                self.indices = sorted(self.indices[:self.split_ind])
            else:
                self.indices = sorted(self.indices[self.split_ind:])

            # sanity checks
            assert (self.num_points_total >= self.num_points)
            assert (self.num_features_total == (3 + self.num_features))

            # store seed
            self.seed = seed
            
            # normalization involves all the data
            # cms transform if requested
            cms = 0.
            if self.cms_transform:
                # average over points
                cms = np.mean(dset[:, 0:3, :].astype(np.float64), axis = 2, keepdims = True).astype(np.float32)

            # normalize input
            self.bias = np.zeros((3 + self.num_features, self.num_points), dtype = np.float32)
            self.scale = np.ones((3 + self.num_features, self.num_points), dtype = np.float32)
            if self.normalize == 'box':
                self.bias[0:3, :] = (dset[:, 0:3, :] - cms).min()
                self.scale[0:3, :] = 1. / ((dset[:, 0:3, :] - cms).max() - self.bias)
                
            # load the relevant chunks into memory
            # compute shard sizes
            fullsize = len(self.indices)
            chunksize = fullsize // num_shards
            start = shard_id * chunksize
            end = start + chunksize
            
            # apply sharding
            self.indices = self.indices[start:end]
            
            # load the data
            # data
            data_array = dset[...].astype(np.float32)
            self.data_array = data_array[self.indices, ...]
            # rmsd
            rmsd_array = rmsd[...].astype(np.float32)
            self.rmsd_array = rmsd_array[self.indices]
            # fnc
            fnc_array = fnc[...].astype(np.float32)
            self.fnc_array = fnc_array[self.indices]
            
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        #get global index so that we can later relate it to the global file position
        index = self.indices[idx]
        
        #self.dset.id.refresh()
        if self.num_points < self.num_points_total:
            # select points to read
            point_indices = self.rng.choice(self.num_points_total, size = self.num_points,
                                            replace = False, shuffle = False)

            # read
            self.token = self.data_array[idx, :, point_indices]
        else:
            self.token = self.data_array[idx, ...]
        
        # cms subtract
        if self.cms_transform:
            self.token[0:3, :] -= np.mean(self.token[0:3, :], axis = -1, keepdims = True)

        if np.any(np.isnan(self.token)):
            raise ValueError("NaN encountered in input.")
            
        # normalize
        result = (self.token - self.bias) * self.scale
        rmsd = torch.tensor(self.rmsd_array[idx], requires_grad=False)
        fnc = torch.tensor(self.fnc_array[idx], requires_grad=False)
        return torch.tensor(result, requires_grad = False), rmsd, fnc, index

