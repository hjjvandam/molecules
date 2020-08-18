import torch
import numpy as np
from torch.utils.data import Dataset
from molecules.utils import open_h5

class ContactMapDataset(Dataset):
    """
    PyTorch Dataset class to load contact matrix data. Uses HDF5
    files and only reads into memory what is necessary for one batch.
    """
    def __init__(self, path, dataset_name, rmsd_name,
                 shape, split_ptc=0.8,
                 split='train', cm_format='sparse-concat'):
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
        """
        if split not in ('train', 'valid'):
            raise ValueError("Parameter split must be 'train' or 'valid'.")
        if split_ptc < 0 or split_ptc > 1:
            raise ValueError('Parameter split_ptc must satisfy 0 <= split_ptc <= 1.')

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        h5_file = open_h5(path)

        if cm_format == 'sparse-rowcol':
            group = h5_file[dataset_name]
            self.len = len(group.get('row'))
        elif cm_format == 'sparse-concat':
            self.dset = h5_file[dataset_name]
            self.len = len(self.dset)
        elif cm_format == 'full':
            # contact_maps dset has shape (N, W, H, 1)
            self.dset = h5_file[dataset_name]
            self.len = len(self.dset)
        else:
            raise ValueError(f'Invalid cm_format {cm_format}. Should be one of ' \
                              '[sparse-rowcol, sparse-concat, full].')

        # train validation split index
        self.split_ind = int(split_ptc * self.len)
        self.split = split
        self.cm_format = cm_format
        self.shape = shape
        self.not_init = True
        self.path = path
        self.dataset_name = dataset_name
        self.rmsd_name = rmsd_name


    def __len__(self):
        if self.split == 'train':
            return self.split_ind
        return self.len - self.split_ind

    def __getitem__(self, idx):
        if self.split == 'valid':
            idx += self.split_ind

        if self.not_init:
            h5_file = open_h5(self.path)
            if self.cm_format == 'sparse-rowcol':
                group = h5_file[self.dataset_name]
                self.row_dset = group.get('row')
                self.col_dset = group.get('col')
            elif self.cm_format == 'sparse-concat':
                self.dset = h5_file[self.dataset_name]
            elif self.cm_format == 'full':
                # contact_maps dset has shape (N, W, H, 1)
                self.dset = h5_file[self.dataset_name]

            self.rmsd = h5_file[self.rmsd_name]
            self.not_init = False

        if self.cm_format == 'sparse-rowcol':
            indices = torch.from_numpy(np.vstack((self.row_dset[idx],
                                                  self.col_dset[idx]))).to(torch.long)
            values = torch.ones(indices.shape[1], dtype=torch.float32)
            # Set shape to the last 2 elements of self.shape.
            # Handles (1, W, H) and (W, H)
            data = torch.sparse.FloatTensor(indices, values,
                        self.shape[-2:]).to_dense()
        elif self.cm_format == 'sparse-concat':
            indices = torch.from_numpy(self.dset[idx].reshape(2, -1) \
                           .astype('int16')).to(torch.long)
            values = torch.ones(indices.shape[1], dtype=torch.float32)
            # Set shape to the last 2 elements of self.shape.
            # Handles (1, W, H) and (W, H)
            data = torch.sparse.FloatTensor(indices, values,
                        self.shape[-2:]).to_dense()
        elif self.cm_format == 'full':
            data = torch.Tensor(self.dset[idx, ...])

        rmsd = self.rmsd[idx]

        return data.view(self.shape), torch.tensor(rmsd, requires_grad=False)
