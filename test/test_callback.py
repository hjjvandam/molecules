import os
import glob
import shutil
import pytest
import numpy as np
import torch
from torchsummary import summary
from molecules.utils import open_h5
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.utils.callback import LossCallback, CheckpointCallback, Embedding2dCallback

class TestCallback:

    class DummyContactMap(Dataset):
        def __init__(self, input_shape, size=200):
            self.maps = np.eye(input_shape[-1]).reshape(input_shape)
            self.maps = np.array([self.maps for _ in range(size)])

            # Creates size identity matrices. Total shape: (size, input_shape)

        def __len__(self):
            return len(self.maps)

        def __getitem__(self, idx):
            return torch.from_numpy(self.maps[idx]).to(torch.float32)

    # TODO: find elegant way to get the input_shape for the model initialization
    class ContactMap(Dataset):
        def __init__(self, path, split_ptc=0.8, split='train', squeeze=False):
            with open_h5(path) as input_file:
                # Access contact matrix data from h5 file
                data = np.array(input_file['contact_maps'])[:1000]

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
        self.epochs = 2
        self.batch_size = 100
        self.input_shape = (1, 22, 22) # Use FSPeptide sized contact maps
        self.checkpoint_dir = os.path.join('.', 'test', 'test_checkpoint')

        # Optimal Fs-peptide params
        self.fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                                  'kernels': [5, 5, 5, 5],
                                  'strides': [1, 2, 1, 1],
                                  'affine_widths': [64],
                                  'affine_dropouts': [0],
                                  'latent_dim': 10}

        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

        path = './test/cvae_input.h5'

        self.train_loader = DataLoader(TestCallback.ContactMap(path, split='train'),
                                  batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TestCallback.ContactMap(path, split='valid'),
                                 batch_size=self.batch_size, shuffle=True)

        # Save checkpoint to test loading

        checkpoint_callback = CheckpointCallback(directory=self.checkpoint_dir,
                                                 interval=2)

        hparams = SymmetricVAEHyperparams(**self.fs_peptide_hparams)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams)

        vae.train(self.train_loader, self.test_loader, epochs=1,
                  callbacks=[checkpoint_callback])

        # Get checkpoint after 2nd epoch
        file = os.path.join(self.checkpoint_dir, '*')
        self.checkpoint_file = sorted(glob.glob(file))[-1]

    def test_pytorch_cvae_real_data(self):

        hparams = SymmetricVAEHyperparams(**self.fs_peptide_hparams)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams)

        print(vae)
        summary(vae.model, self.input_shape)

        loss_callback = LossCallback()
        checkpoint_callback = CheckpointCallback(directory=self.checkpoint_dir,
                                                 interval=2)
        embedding_callback = Embedding2dCallback(TestCallback.DummyContactMap(self.input_shape)[:])

        vae.train(self.train_loader, self.test_loader, self.epochs,
                  callbacks=[loss_callback, checkpoint_callback, embedding_callback])

        print(loss_callback.train_losses)
        print(loss_callback.valid_losses)

        #embedding_callback.save('./test/embed.pt')
        #loss_callback.save('./test/loss.pt')

    def test_load_checkpoint(self):

        hparams = SymmetricVAEHyperparams(**self.fs_peptide_hparams)
        checkpoint_callback = CheckpointCallback(directory=self.checkpoint_dir,
                                                 interval=2)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams)

        print('loading checkpoint to resume training')
        # Train for 2 more additional epochs i.e. epochs=4
        vae.train(self.train_loader, self.test_loader, epochs=4,
                  checkpoint=self.checkpoint_file, callbacks=[checkpoint_callback])
    @classmethod
    def teardown_class(self):
        # Remove checkpoint directory
        shutil.rmtree(self.checkpoint_dir)

