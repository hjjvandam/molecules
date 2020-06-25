import torch
from torch import nn
from math import isclose
from molecules.ml.unsupervised.vae.utils import (conv_output_shape, same_padding,
                                                 get_activation, init_weights, prod)
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
    def __init__(self, output_shape, hparams, encoder_shapes):
        super(SymmetricDecoderConv2d, self).__init__()

        assert isinstance(hparams, SymmetricVAEHyperparams)
        hparams.validate()

        self.output_shape = output_shape
        self.encoder_shapes = encoder_shapes
        self.hparams = hparams

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers, self.conv_acts = self._conv_layers()
        self.conv_output_sizes = list(reversed(self.encoder_shapes[:-1]))
        # Reshape flattened x as a tensor (channels, output1, output2)
        self.reshape = (-1, *self.encoder_shapes[-1])

        self.init_weights()

    def init_weights(self):
        self.affine_layers.apply(init_weights)
        self.conv_layers.apply(init_weights)

    def forward(self, x):
        x = self.affine_layers(x).view(self.reshape)
        batch_size = x.size()[0]
        for conv_t, act, output_size in \
            zip(self.conv_layers, self.conv_acts, self.conv_output_sizes):
            x = act(conv_t(x, output_size=(batch_size, *output_size)))
        return x

    def decode(self, embedding):
        self.eval()
        with torch.no_grad():
            return self(embedding)

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
        activations : list
            Activation functions
        """
        layers, activations = [], []

        act = get_activation(self.hparams.activation)

        # The first out_channels should be the second to last filter size
        tmp = self.hparams.filters.pop()

        # self.output_shape[0] Needs to be the last out_channels to match the input matrix
        for i, (filter_, kernel, stride) in enumerate(reversedzip((self.output_shape[0],
                                                                  *self.hparams.filters),
                                                                  self.hparams.kernels,
                                                                  self.hparams.strides)):
            shape = self.encoder_shapes[-1*i -1]

            # TODO: this is a quick fix but might not generalize to some architectures
            if stride == 1:
                padding = same_padding(shape[1:], kernel, stride)
            else:
                padding = tuple(int(dim % 2 == 0) for dim in self.encoder_shapes[-1*i -2][1:])

            layers.append(nn.ConvTranspose2d(in_channels=shape[0],
                                             out_channels=filter_,
                                             kernel_size=kernel,
                                             stride=stride,
                                             padding=padding))

            # TODO: revist padding, output_padding, see github issue.
            #       This code may not generalize to other examples. Needs testing.
            #       this also needs to be addressed in conv_output_dim

            activations.append(act)

        # Overwrite output activation
        activations[-1] = get_activation(self.hparams.output_activation)

        # Restore invariant state
        self.hparams.filters.append(tmp)

        return nn.ModuleList(layers), activations

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        layers : list
            Linear layers
        """

        layers = []

        act = get_activation(self.hparams.activation)

        in_features = self.hparams.latent_dim

        for width, dropout in reversedzip(self.hparams.affine_widths,
                                          self.hparams.affine_dropouts):

            layers.append(nn.Linear(in_features=in_features,
                                    out_features=width))

            layers.append(act)

            if not isclose(dropout, 0):
                layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add last layer with dims to connect the last linear layer to
        # the first convolutional decoder layer
        layers.append(nn.Linear(in_features=self.hparams.affine_widths[0],
                                out_features=prod(self.encoder_shapes[-1])))
        layers.append(act)

        return layers
