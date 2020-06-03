import os
import pytest
import numpy as np
import torch
from torchsummary import summary
from molecules.utils import open_h5
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.vae import SymmetricVAEHyperparams
from molecules.ml.unsupervised.vae import VAE
from molecules.ml.hyperparams import OptimizerHyperparams

class TestVAE:

    # TODO: find elegant way to get the input_shape for the model initialization
    class ContactMap(Dataset):
        def __init__(self, path, split_ptc=0.8, split='train', squeeze=False):
            with open_h5(path) as input_file:
                # Access contact matrix data from h5 file
                data = np.array(input_file['contact_maps'])

            # 80-20 train validation split index
            split_ind = int(split_ptc * len(data))

            if split == 'train':
                self.data = data[:split_ind]
            elif split == 'valid':
                self.data = data[split_ind:]
            else:
                raise ValueError(f'Parameter split={split} is invalid.')

            # TODO: in future contact map Dataset, pass in device to precompute
            #       the operation


            # TODO: this reshape code may not be the best solution. revisit
            num_residues = self.data.shape[2]
            assert num_residues == 22

            if squeeze:
                shape = (-1, num_residues, num_residues)
            else:
                shape = (-1, 1, num_residues, num_residues)

            self.data = torch.from_numpy(self.data.reshape(shape)).to(torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    @classmethod
    def setup_class(self):
        self.epochs = 10
        self.batch_size = 100
        self.input_shape = (1, 22, 22) # Use FSPeptide sized contact maps
        self.checkpoint_dir = os.path.join('.', 'test', 'test_checkpoint')

        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': 10}

        hparams ={'filters': [64, 64, 64, 64],
                  'kernels': [3, 3, 3, 3],
                  'strides': [1, 2, 1, 1],
                  'affine_widths': [128],
                  'affine_dropouts': [0],
                  'latent_dim': 3}

        self.hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)
        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    def test_pytorch_cvae_real_data(self):

        path = './test/cvae_input.h5'

        train_loader = DataLoader(TestVAE.ContactMap(path, split='train'),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestVAE.ContactMap(path, split='valid'),
                                 batch_size=self.batch_size, shuffle=True)

        from molecules.utils.checkpoint import Checkpoint
        checkpoint = Checkpoint(directory=self.checkpoint_dir,
                                interval=0) # Once per 2 batches

        vae = VAE(self.input_shape, self.hparams, self.optimizer_hparams, checkpoint=checkpoint)

        print(vae)
        summary(vae.model, self.input_shape)

        from molecules.utils.callback import LossHistory

        loss_history = LossHistory()

        vae.train(train_loader, test_loader, self.epochs, callbacks=[loss_history])

    @classmethod
    def teardown_class(self):
        pass
