import os
import glob
import shutil
import pytest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
from molecules.ml.unsupervised.vae import VAE
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.utils.checkpoint import Checkpoint

class TestVAE:

    class DummyContactMap(Dataset):
            def __init__(self, input_shape, size=200):
                self.maps = np.eye(input_shape[-1]).reshape(input_shape)
                self.maps = np.array([self.maps for _ in range(size)])

                # Creates size identity matrices. Total shape: (size, input_shape)

            def __len__(self):
                return len(self.maps)

            def __getitem__(self, idx):
                return torch.from_numpy(self.maps[idx]).to(torch.float32)

    @classmethod
    def setup_class(self):
        self.epochs = 2
        self.batch_size = 50
        self.input_shape = (22, 22) # Use FSPeptide sized contact maps


        self.train_loader = DataLoader(TestVAE.DummyContactMap(self.input_shape),
                                  batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TestVAE.DummyContactMap(self.input_shape),
                                 batch_size=self.batch_size, shuffle=True)

        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

        self.checkpoint_dir = os.path.join('.', 'test', 'test_checkpoint')

        # Save checkpoint to test loading

        checkpoint = Checkpoint(directory=self.checkpoint_dir,
                                interval=0) # Once per epoch

        hparams = ResnetVAEHyperparams(self.input_shape, latent_dim=11)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams,
                  checkpoint=checkpoint)

        vae.train(self.train_loader, self.test_loader, epochs=2)

        # Get checkpoint after 2nd epoch
        file = os.path.join(self.checkpoint_dir, '*')
        self.checkpoint_file = sorted(glob.glob(file))[-1]

    def test_checkpoint_per_epoch(self):

        checkpoint = Checkpoint(directory=self.checkpoint_dir,
                                interval=0) # Once per epoch

        hparams = ResnetVAEHyperparams(self.input_shape, latent_dim=11)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams,
                  checkpoint=checkpoint)

        vae.train(self.train_loader, self.test_loader, self.epochs)

    def test_checkpoint_per_batch(self):

        hparams = ResnetVAEHyperparams(self.input_shape, latent_dim=11)

        checkpoint = Checkpoint(directory=self.checkpoint_dir,
                                interval=2) # Once per 2 batches

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams,
                  checkpoint=checkpoint)

        vae.train(self.train_loader, self.test_loader, self.epochs)

    def test_load_checkpoint(self):

        hparams = ResnetVAEHyperparams(self.input_shape, latent_dim=11)

        checkpoint = Checkpoint(directory=self.checkpoint_dir)

        vae = VAE(self.input_shape, hparams, self.optimizer_hparams,
                  checkpoint=checkpoint)

        # Train for 2 more additional epochs i.e. epochs=4
        vae.train(self.train_loader, self.test_loader, epochs=4,
                  checkpoint=self.checkpoint_file)

    @classmethod
    def teardown_class(self):
        # Remove checkpoint directory
        shutil.rmtree(self.checkpoint_dir)
