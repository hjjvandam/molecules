import pytest
import numpy as np
from molecules.ml.unsupervised import EncoderHyperparams, DecoderHyperparams

# Keras imports
from keras.optimizers import RMSprop
from molecules.ml.unsupervised import (VAE, EncoderConvolution2D,
                                       DecoderConvolution2D)

# Pytorch imports
import torch
from molecules.utils import open_h5
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.conv_vae.pytorch_cvae import CVAE
from molecules.ml.hyperparams import OptimizerHyperparams
from torchsummary import summary

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

        # For Keras model
        self.dataset = TestCVAE.DummyContactMap(self.input_shape)

        # Optimal Fs-peptide params
        hparams ={'num_conv_layers': 4,
                  'filters': [100, 100, 100, 100],
                  'kernels': [5, 5, 5, 5],
                  'num_affine_layers': 1,
                  'affine_widths': [64],
                  'affine_dropouts': [0],
                  'latent_dim': 10
                 }

        strides = [1, 2, 1, 1]

        self.encoder_hparams = EncoderHyperparams(**hparams, strides=strides)
        self.decoder_hparams = DecoderHyperparams(**hparams, strides=list(reversed(strides)))
        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    def notest_keras_cvae(self):

        encoder = EncoderConvolution2D(input_shape=self.input_shape,
                                       hyperparameters=self.encoder_hparams)

        # Get shape attributes of the last encoder layer to define the decoder
        encode_conv_shape, num_conv_params = encoder.get_final_conv_params()

        decoder = DecoderConvolution2D(output_shape=self.input_shape,
                                       enc_conv_params=num_conv_params,
                                       enc_conv_shape=encode_conv_shape,
                                       hyperparameters=self.decoder_hparams)

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        cvae = VAE(input_shape=self.input_shape,
                   encoder=encoder,
                   decoder=decoder,
                   optimizer=optimizer)

        split = int(len(self.dataset) * 0.8)
        train, valid = self.dataset.maps[:split], self.dataset.maps[split:]

        print(type(train))
        print(train.shape)
        cvae.summary()

        cvae.train(data=train, validation_data=valid,
                   batch_size=self.batch_size, epochs=self.epochs)


    def notest_pytorch_cvae(self):
        cvae = CVAE(self.input_shape, self.encoder_hparams,
                    self.decoder_hparams, self.optimizer_hparams)

        print(cvae)
        summary(cvae.cvae, self.input_shape)


        cvae.train(self.train_loader, self.test_loader, self.epochs)


    def test_padding(self):
        from molecules.ml.unsupervised.conv_vae.pytorch_cvae.cvae import even_padding

        input_dim = 22
        kernel_size = 3

        assert even_padding(input_dim, kernel_size, stride=1) == 1
        assert even_padding(input_dim, kernel_size, stride=2) == 1


    def test_pytorch_cvae_real_data(self):

        path = './test/cvae_input.h5'

        self.train_loader = DataLoader(TestCVAE.ContactMap(path, split='train'),
                                       batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TestCVAE.ContactMap(path, split='valid'),
                                      batch_size=self.batch_size, shuffle=True)

        cvae = CVAE(self.input_shape, self.encoder_hparams,
                    self.decoder_hparams, self.optimizer_hparams)

        print(cvae)
        summary(cvae.cvae, self.input_shape)


        cvae.train(self.train_loader, self.test_loader, self.epochs)

    @classmethod
    def teardown_class(self):
        pass