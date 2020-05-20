import torch
from torch import nn
from molecules.ml.unsupervised.vae.utils import (conv2d_output_dim, same_padding,
                                                 select_activation, init_weights)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
from molecules.ml.unsupervised.vae.resnet.residual_module import ResidualConv1d

class ResnetEncoder(nn.Module):
    def __init__(self, input_shape, hparams):
        super(ResnetEncoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hparams

        self.encoder = nn.Sequential(*self._encoder_layers())

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

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def _encoder_layers(self):
        layers = []

        # Contact matrices have one channel
        in_channels = self.input_shape[0]

        padding = same_padding(self.input_shape[1], kernel_size=5, stride=1)

        layers.append(nn.Conv1d(in_channels=in_channels,
                                out_channels=self.hparams.enc_filters,
                                kernel_size=5,
                                stride=1,
                                padding=padding))

        layers.append(select_activation(self.hparams.activation))

        return layers

    def _embedding_layer(self):
        # TODO: implement
        return nn.Linear(in_features=1,
                         out_features=1)
