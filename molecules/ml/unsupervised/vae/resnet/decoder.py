import torch
from torch import nn
from molecules.ml.unsupervised.utils import (conv_output_dim, same_padding,
                                             get_activation, init_weights, prod)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams
from molecules.ml.unsupervised.vae.resnet.residual_module import ResidualConv1d

class ResnetDecoder(nn.Module):
    def __init__(self, match_shape, output_shape, hparams):
        super(ResnetDecoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.match_shape = match_shape
        self.output_shape = output_shape
        self.hparams = hparams

        self.decoder = self._decoder_layers()

        self.init_weights()

    def init_weights(self):
        self.decoder.apply(init_weights)

    def forward(self, x):
        x = self.match_layer(x)
        x = x.reshape((x.shape[0], self.match_shape[1], self.match_shape[0]))
        #x = x.view(x.shape[0], x.shape[1], 1)
        return self.decoder(x)

    def decode(self, embedding):
        self.eval()
        with torch.no_grad():
            return self(embedding)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def _decoder_layers(self):

        # insert an FC layer to get the shapes matching
        if self.hparams.latent_dim != self.match_shape[0]:
            self.match_layer = nn.Sequential(nn.Linear(in_features = self.hparams.latent_dim,
                                                       out_features = prod(self.match_shape)),
                                             get_activation(self.hparams.output_activation))
        else:
            self.match_layer = nn.Identity()

        # we do a sneaky transposition here
        res_input_shape = (self.match_shape[1], self.match_shape[0])

        # construct decoder
        layers = []
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
                scale_factor = self.hparams.scale_factor
                # do some matching in case we have a weird scaling factor
                if res_input_shape[1] * scale_factor <= self.output_shape[1]:
                    layers.append(nn.Upsample(scale_factor=scale_factor))
                    res_input_shape = (res_input_shape[0], res_input_shape[1] * scale_factor)
                    self.hparams.upsample_rounds -= 1
                else:
                    # we are done here: upsample to output size and thats it
                    layers.append(nn.Upsample(size=self.output_shape[1]))
                    res_input_shape = (res_input_shape[0], self.output_shape[1])
                    self.hparams.upsample_rounds = 0
                    
        # add padding for last layer
        padding = same_padding(res_input_shape[1],
                               self.hparams.dec_kernel_size,
                               stride=1)

        layers.append(nn.Conv1d(in_channels=res_input_shape[0],
                                out_channels=self.output_shape[1], # should be num_residues i.e. nchars
                                kernel_size=self.hparams.dec_kernel_size,
                                stride=1,
                                padding=padding))

        layers.append(get_activation(self.hparams.output_activation))


        return nn.Sequential(*layers)
