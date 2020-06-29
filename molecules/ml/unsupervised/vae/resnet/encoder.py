import torch
from torch import nn
from molecules.ml.unsupervised.vae.utils import (conv_output_shape,
                                                 same_padding, get_activation,
                                                 init_weights, prod)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
from molecules.ml.unsupervised.vae.resnet.residual_module import ResidualConv1d


class ResnetEncoder(nn.Module):
    def __init__(self, input_shape, hparams):
        super(ResnetEncoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # input_shape is of dimension (N1, N2) where N1, N2 are the
        # number of residues, treat 1st dim of contact matrix as
        # channel

        self.input_shape = input_shape
        self.hparams = hparams

        self.encoder, output_shape = self._encoder_layers()

        self.mu = self._embedding_layer(output_shape)
        self.logvar = self._embedding_layer(output_shape)

        self.init_weights()

    def init_weights(self):
        self.encoder.apply(init_weights)
        init_weights(self.mu)
        init_weights(self.logvar)

    def forward(self, x):
        x = self.encoder(x)
        m,v = self.mu(x), self.logvar(x)
        print('encoder forward outshape: ', m.shape)
        return m, v

    def encode(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)[0]

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def _encoder_layers(self):
        layers = []

        # input_shape[0] is max_len
        # input_shape[1] is nchars

        padding = same_padding(self.input_shape[1], kernel_size=5, stride=1)

        layers.append(nn.Conv1d(in_channels=self.input_shape[0], # should be num_residues
                                out_channels=self.hparams.enc_filters,
                                kernel_size=5,
                                stride=1,
                                padding=padding))

        layers.append(get_activation(self.hparams.activation))

        res_input_shape = conv_output_shape(self.input_shape[1],
                                            kernel_size=5,
                                            stride=1,
                                            padding=padding,
                                            num_filters=self.hparams.enc_filters,
                                            dim=1)

        # Add residual layers
        for lidx in range(self.hparams.enc_reslayers):

            filters = self.hparams.enc_filters * self.hparams.enc_filter_growth_fac**lidx
            filters = round(filters) # To nearest int

            layers.append(ResidualConv1d(res_input_shape,
                                         filters,
                                         self.hparams.enc_kernel_size,
                                         self.hparams.activation,
                                         shrink=True))

            res_input_shape = layers[-1].output_shape

        return nn.Sequential(*layers, nn.Flatten()), prod(res_input_shape)

    def _embedding_layer(self, output_shape):
        return nn.Linear(in_features=output_shape,
                         out_features=self.hparams.latent_dim)
