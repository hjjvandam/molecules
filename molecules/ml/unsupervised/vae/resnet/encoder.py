import torch
from torch import nn
from molecules.ml.unsupervised.vae.utils import (conv2d_output_dim, same_padding,
                                                 select_activation, init_weights)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams

class ResnetEncoder(nn.Module):
    def __init__(self, input_shape, hparams):
        super(ResnetEncoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hparams

        self.encoder = nn.Sequential()

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

    def _embedding_layer(self):
        # TODO: implement
        return nn.Linear(in_features=1,
                         out_features=1)
