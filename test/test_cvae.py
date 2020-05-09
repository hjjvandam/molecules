import pytest
import numpy as np
from torch.utils.data import Dataset, DataLoader
from molecules.ml.unsupervised.conv_vae.pytorch_cvae import CVAE
from molecules.ml.unsupervised import EncoderHyperparams, DecoderHyperparams
from molecules.ml.hyperparams import OptimizerHyperparams


class TestCVAE:

    class DummyContactMap(Dataset):
            def __init__(self, input_shape):
                # Use FSPeptide sized contact maps
                self.maps = np.random.randn(1000, *input_shape)

            def __len__(self):
                return len(self.maps)

            def __getitem__(self, idx):
                return self.maps[idx]

    @classmethod
    def setup_class(self):
        self.epochs = 2
        self.input_shape = (21, 21)
        self.train_loader = DataLoader(TestCVAE.DummyContactMap(self.input_shape),
                                                       batch_size=4, shuffle=True)
        self.test_loader = DataLoader(TestCVAE.DummyContactMap(self.input_shape),
                                                      batch_size=4, shuffle=True)

        hparams ={'num_conv_layers': 4,
                  'filters': [64, 64, 64, 64],
                  'kernels': [3, 3, 3, 3],
                  'strides': [1, 2, 1, 1],
                  'num_affine_layers': 1,
                  'affine_widths': [128],
                  'affine_dropouts': [0],
                  'latent_dim': 8
                 }

        self.encoder_hparams = EncoderHyperparams(**hparams)
        self.decoder_hparams = DecoderHyperparams(**hparams)
        self.optimizer_hparams = OptimizerHyperparams(name='RMSprop')


    def test_cvae(self):
        cvae = CVAE(self.input_shape, self.encoder_hparams,
                    self.decoder_hparams, self.optimizer_hparams)

        cvae.train(self.train_loader, self.test_loader, self.epochs)



    @classmethod
    def teardown_class(self):
        pass