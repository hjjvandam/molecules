import torch
from torch import nn
from molecules.ml.unsupervised.vae.utils import (conv2d_output_dim, same_padding,
                                                 select_activation, init_weights)
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams

class ResnetDecoder(nn.Module):
    def __init__(self, output_shape, hparams):
        super(ResnetDecoder, self).__init__()

        assert isinstance(hparams, ResnetVAEHyperparams)
        hparams.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.hparams = hparams

        self.decoder = nn.Sequential()

        self.init_weights()

    def init_weights(self):
        self.decoder.apply(init_weights)

    def forward(self, x):
        return self.decoder(x)

    def decode(self, embedding):
        self.eval()
        with torch.no_grad():
            return self.forward(embedding)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
