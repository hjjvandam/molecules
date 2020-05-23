import torch
from torch import nn
from molecules.ml.unsupervised.vae.utils import (conv_output_dim, same_padding,
                                                 get_activation, init_weights)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
from molecules.ml.unsupervised.vae.resnet.residual_module import ResidualConv1d

class ResnetDecoder(nn.Module):
    def __init__(self, output_shape, hparams):
        super(ResnetDecoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.hparams = hparams

        self.decoder = self._decoder_layers()

        self.init_weights()

    def init_weights(self):
        self.decoder.apply(init_weights)

    def forward(self, x):
        return self.decoder(x.view(-1, 1, x.shape[1]))

    def decode(self, embedding):
        self.eval()
        with torch.no_grad():
            return self(embedding)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


    def _decoder_layers(self):

        layers = []

        res_input_shape = (1, self.hparams.latent_dim)

        for lidx in range(self.hparams.dec_reslayers):

            filters = self.hparams.dec_filters * self.hparams.dec_filter_growth_rate**lidx
            filters = round(filters)

            if self.hparams.shrink_rounds:
                self.hparams.shrink_rounds -= 1

            layers.append(ResidualConv1d(res_input_shape,
                                         filters,
                                         self.hparams.dec_kernel_size,
                                         self.hparams.activation,
                                         shrink=self.hparams.shrink_rounds))

            res_input_shape = layers[-1].output_shape

            if self.hparams.upsample_rounds:
                # TODO: consider upsample mode nearest neightbor etc.
                #       https://pytorch.org/docs/master/generated/torch.nn.Upsample.html
                layers.append(nn.Upsample(scale_factor=2))
                self.hparams.upsample_rounds -= 1

        padding = same_padding(res_input_shape[1],
                               self.hparams.dec_kernel_size,
                               stride=1)

        layers.append(nn.Conv1d(in_channels=res_input_shape[0],
                                out_channels=self.output_shape[1], # should be num_residues
                                kernel_size=self.hparams.dec_kernel_size,
                                stride=1,
                                padding=padding))

        layers.append(get_activation(self.hparams.output_activation))


        return nn.Sequential(*layers)
