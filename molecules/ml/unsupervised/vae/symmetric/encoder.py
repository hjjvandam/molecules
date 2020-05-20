import torch
from torch import nn
from math import isclose
from molecules.ml.unsupervised.vae.utils import *
from molecules.ml.unsupervised.vae.symmetric import SymmetricVAEHyperparams

class SymmetricEncoderConv2d(nn.Module):
    def __init__(self, input_shape, hparams):
        super(SymmetricEncoderConv2d, self).__init__()

        assert isinstance(hparams, SymmetricVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hparams
        self.encoder_dim = input_shape[1]

        self.encoder = nn.Sequential(*self._conv_layers(),
                                     nn.Flatten(),
                                     *self._affine_layers())

        self.mu = self._embedding_layer()
        self.logvar = self._embedding_layer()

        self.init_weights()

    def init_weights(self):
        self.encoder.apply(init_weights)
        init_weights(self.mu)
        init_weights(self.logvar)

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)[0]

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """

        conv2d_layers = []

        # Contact matrices have one channel
        in_channels = self.input_shape[0]

        for filter_, kernel, stride in zip(self.hparams.filters,
                                           self.hparams.kernels,
                                           self.hparams.strides):

            padding = even_padding(self.encoder_dim, kernel, stride)

            conv2d_layers.append(nn.Conv2d(in_channels=in_channels,
                                           out_channels=filter_,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=padding))

            conv2d_layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            in_channels = filter_

            self.encoder_dim = conv2d_output_dim(self.encoder_dim, kernel, stride, padding)

        return conv2d_layers

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        fc_layers : list
            Linear layers
        """

        fc_layers = []

        # First layer gets flattened convolutional output
        in_features = self.hparams.filters[-1] * self.encoder_dim**2

        for width, dropout in zip(self.hparams.affine_widths,
                                  self.hparams.affine_dropouts):

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(select_activation(self.hparams.activation))

            if not isclose(dropout, 0):
                fc_layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return fc_layers

    def _embedding_layer(self):
        return nn.Linear(in_features=self.hparams.affine_widths[-1],
                         out_features=self.hparams.latent_dim)
