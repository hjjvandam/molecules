import torch
from torch import nn
from math import isclose
from molecules.ml.unsupervised.vae.utils import (conv_output_dim, same_padding,
                                                 select_activation, init_weights)
from molecules.ml.unsupervised.vae.symmetric import SymmetricVAEHyperparams

def reversedzip(*iterables):
    """
    Yields the zip of iterables in reversed order.

    Example
    -------
    l1 = [1,2,3]
    l2 = ['a','b','c']
    l3 = [5,6,7]

    for tup in reversedzip(l1, l2, l3):
        print(tup)

    Outputs:
        (3, 'c', 7)
        (2, 'b', 6)
        (1, 'a', 5)

    """
    for tup in zip(*map(reversed, iterables)):
        yield tup

class SymmetricDecoderConv2d(nn.Module):
    def __init__(self, output_shape, hparams, encoder_dim):
        super(SymmetricDecoderConv2d, self).__init__()

        assert isinstance(hparams, SymmetricVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.encoder_dim = encoder_dim
        self.hparams = hparams

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers = nn.Sequential(*self._conv_layers())

        self.init_weights()

    def init_weights(self):
        self.affine_layers.apply(init_weights)
        self.conv_layers.apply(init_weights)

    def reshape(self, x):
        """
        Reshape flattened x as a tensor (channels, output, output)
        """
        new_shape = (-1, self.hparams.filters[-1],
                     self.encoder_dim, self.encoder_dim)
        return x.view(new_shape)

    def forward(self, x):
        x = self.affine_layers(x)
        x = self.reshape(x)
        return self.conv_layers(x)

    def decode(self, embedding):
        self.eval()
        with torch.no_grad():
            return self.forward(embedding)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        layers : list
            Convolution layers
        """
        layers = []

        in_channels = self.hparams.filters[-1]

        # Dimension of square matrix
        input_dim = self.output_shape[1]

        # Set last filter to be the number of channels in the reconstructed image.
        tmp = self.hparams.filters[0]
        self.hparams.filters[0] = self.output_shape[0]

        for filter_, kernel, stride in reversedzip(self.hparams.filters,
                                                   self.hparams.kernels,
                                                   self.hparams.strides):

            padding = same_padding(input_dim, kernel, stride)

            layers.append(nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=filter_,
                                             kernel_size=kernel,
                                             stride=stride,
                                             padding=padding,
                                             output_padding=1 if stride != 1 else 0))

            # TODO: revist output_padding, see github issue.
            #       This code may not generalize to other examples. Needs testing.

            layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            # Except for the last layer which is 1 (or output_shape channels)
            in_channels = filter_

            # Compute non-channel dimension given to next layer
            input_dim = conv_output_dim(input_dim, kernel, stride, padding, transpose=True)

        # Overwrite output activation
        layers[-1] = select_activation(self.hparams.output_activation)

        # Restore invariant state
        self.hparams.filters[0] = tmp

        return layers

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        layers : list
            Linear layers
        """

        layers = []

        in_features = self.hparams.latent_dim

        for width, dropout in reversedzip(self.hparams.affine_widths,
                                          self.hparams.affine_dropouts):

            layers.append(nn.Linear(in_features=in_features,
                                    out_features=width))

            layers.append(select_activation(self.hparams.activation))

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add last layer with dims to connect the last linear layer to
        # the first convolutional decoder layer
        layers.append(nn.Linear(in_features=self.hparams.affine_widths[0],
                                   out_features=self.hparams.filters[-1] * self.encoder_dim**2))
        layers.append(select_activation(self.hparams.activation))


        return layers