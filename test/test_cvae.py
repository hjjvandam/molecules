import pytest
import numpy as np
import torch
from torchsummary import summary
from molecules.utils import open_h5
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.vae import ConvVAEHyperparams
from molecules.ml.unsupervised.vae import CVAE
from molecules.ml.hyperparams import OptimizerHyperparams

class TestCVAE:

    class DummyContactMap(Dataset):
            def __init__(self, input_shape, size=1000):
                self.maps = np.eye(input_shape[-1]).reshape(input_shape)
                self.maps = np.array([self.maps for _ in range(size)])

                # Creates size identity matrices. Total shape: (size, input_shape)

            def __len__(self):
                return len(self.maps)

            def __getitem__(self, idx):
                return torch.from_numpy(self.maps[idx]).to(torch.float32)


    # TODO: find elegant way to get the input_shape for the model initialization
    class ContactMap(Dataset):
        def __init__(self, path, split_ptc=0.8, split='train'):
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

            self.data = torch.from_numpy(self.data.reshape(-1, 1, 22, 22)).to(torch.float32)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    @classmethod
    def setup_class(self):
        self.epochs = 10
        self.batch_size = 100
        self.input_shape = (1, 22, 22) # Use FSPeptide sized contact maps
        self.train_loader = DataLoader(TestCVAE.DummyContactMap(self.input_shape),
                                       batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TestCVAE.DummyContactMap(self.input_shape),
                                      batch_size=self.batch_size, shuffle=True)


        # Optimal Fs-peptide params
        hparams ={'filters': [100, 100, 100, 100],
                  'kernels': [5, 5, 5, 5],
                  'strides': [1, 2, 1, 1],
                  'affine_widths': [64],
                  'affine_dropouts': [0],
                  'latent_dim': 10
                 }

        self.hparams = ConvVAEHyperparams(**hparams)
        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    def notest_pytorch_cvae(self):
        cvae = CVAE(self.input_shape, self.hparams, self.optimizer_hparams)

        print(cvae)
        summary(cvae.cvae, self.input_shape)

        cvae.train(self.train_loader, self.test_loader, self.epochs)


    def test_padding(self):
        from molecules.ml.unsupervised.vae.vae import even_padding

        input_dim = 22
        kernel_size = 3

        assert even_padding(input_dim, kernel_size, stride=1) == 1
        assert even_padding(input_dim, kernel_size, stride=2) == 1


    def test_pytorch_cvae_real_data(self):

        path = './test/cvae_input.h5'

        train_loader = DataLoader(TestCVAE.ContactMap(path, split='train'),
                                  batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TestCVAE.ContactMap(path, split='valid'),
                                 batch_size=self.batch_size, shuffle=True)

        cvae = CVAE(self.input_shape, self.hparams, self.optimizer_hparams)

        print(cvae)
        summary(cvae.cvae, self.input_shape)

        cvae.train(train_loader, test_loader, self.epochs)

    @classmethod
    def teardown_class(self):
        pass
