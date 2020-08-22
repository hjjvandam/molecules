import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5
import time

class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """
    def __init__(self, path, dataset_name, rmsd_name,
                 shape, split_ptc=0.8, split='train', seed=333,
                 cm_format='sparse-concat'):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        dataset_name : str
            Path to contact maps in HDF5 file.

        rmsd_name : str
            Path to rmsd data in HDF5 file.

        shape : tuple
            Shape of contact matrices (H, W), may be (1, H, W).

        split_ptc : float
            Percentage of total data to be used as training set.

        split : str
            Either 'train' or 'valid', specifies whether this
            dataset returns train or validation data.

        cm_format : str
            If 'sparse-concat', process data as concatenated row,col indicies.
            If 'sparse-rowcol', process data as sparse row/col COO format.
            If 'full', process data is normal torch tensors (matrices).
            If none of the above, raise a ValueError.

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

        # more params
        self.cm_format = cm_format
        self.shape = shape
        
        # get lengths and paths
        with open_h5(self.file_path, 'r', libver = 'latest', swmr = False) as f:
            if self.cm_format == 'sparse-rowcol':
                self.len = len(f[self.dataset_name]['row'])
            elif self.cm_format == 'sparse-concat':
                self.len = len(f[self.dataset_name])
            elif self.cm_format == 'full':
                self.len = len(f[self.dataset_name])
    
        # do splitting
        self.split_ind = int(split_ptc * self.len)
        self.split = split
        split_rng = np.random.default_rng(seed)
        self.indices = split_rng.permutation(list(range(self.len)))
        if self.split == "train":
            self.indices = sorted(self.indices[:self.split_ind])
        else:
            self.indices = sorted(self.indices[self.split_ind:])

        # inited:
        self.initialized = False

        
    def _init_file(self):
        self.h5_file = open_h5(self.file_path, 'r', libver = 'latest', swmr = False)

        
    def _close_file(self):            
        # close file
        self.h5_file.close()

        
    def __len__(self):
        return len(self.indices)

    
    def __getitem__(self, idx):

        if not self.initialized:
            self._init_file()
            if self.cm_format == 'sparse-rowcol':
                self.row_dset = self.h5_file[self.dataset_name]['row']
                self.col_dset = self.h5_file[self.dataset_name]['col']
            elif self.cm_format == 'sparse-concat':
                self.dset = self.h5_file[self.dataset_name]
            else:
                self.dset = self.h5_file[self.dataset_name]
            self.rmsd_dset = self.h5_file[self.rmsd_name]
            self.initialized = True

        # get real index
        index = self.indices[idx]

        if self.cm_format == 'sparse-rowcol':
            indices = torch.from_numpy(np.vstack((self.row_dset[index],
                                                  self.col_dset[index]))).to(torch.long)
            values = torch.ones(indices.shape[1], dtype=torch.float32)
            # Set shape to the last 2 elements of self.shape.
            # Handles (1, W, H) and (W, H)
            data = torch.sparse.FloatTensor(indices, values,
                        self.shape[-2:]).to_dense()
        elif self.cm_format == 'sparse-concat':
            ind = self.dset[index, ...].reshape(2, -1)
            indices = torch.from_numpy(ind).to(torch.long)
            values = torch.ones(indices.shape[1], dtype=torch.float32)
            # Set shape to the last 2 elements of self.shape.
            # Handles (1, W, H) and (W, H)
            data = torch.sparse.FloatTensor(indices, values,
                        self.shape[-2:]).to_dense()
        elif self.cm_format == 'full':
            data = torch.Tensor(self.dset[index, ...])

        rmsd = self.rmsd_dset[index]

        return data, torch.tensor(rmsd, requires_grad=False), index
