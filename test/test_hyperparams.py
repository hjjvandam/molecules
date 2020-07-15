"""
Tests for `molecules` module.
"""
import os
import pytest
from molecules.ml.unsupervised.vae import SymmetricVAEHyperparams


class TestHyperParams:

    @classmethod
    def setup_class(self):
        self.encoder_fname = os.path.join('.', 'test', 'data', 'encoder-hparams.json')
        self.optimizer_fname = os.path.join('.', 'test', 'data', 'optimizer-hparams.json')

    def test_save_load_functions(self):
        # Set model hyperparameters for encoder
        hparam_options = {'filters': [64, 64, 64, 64],
                          'kernels': [3, 3, 3, 3],
                          'strides': [1, 2, 1, 1],
                          'affine_widths': [128],
                          'latent_dim': 3,
                          'affine_dropouts': [0]
                         }
        hparams = SymmetricVAEHyperparams(**hparam_options)

        # Save model hyperparameters to disk
        hparams.save(self.encoder_fname)

        # Check that 'encoder-hparams.pkl' is in ./test/data
        assert os.path.basename(self.encoder_fname) \
            in os.listdir(os.path.dirname(self.encoder_fname))

        # Load saved hyperparameters from disk
        loaded_hparams = SymmetricVAEHyperparams().load(self.encoder_fname)

        # Check that all attributes were read from disk correctly
        for key, val in hparam_options.items():
            assert getattr(loaded_hparams, key) == val

    def test_validators(self):
        # Set model hyperparameters for encoder and decoder
        hparam_options = {'filters': [64, 64, 64, 64],
                          'kernels': [3, 3, 3, 3],
                          'strides': [1, 2, 1, 1],
                          'affine_widths': [128],
                          'affine_dropouts': [0],
                          'latent_dim': 3
                         }

        hparams = SymmetricVAEHyperparams(**hparam_options)

        # Raises exception if invalid
        hparams.validate()

        # Invalidate state
        hparams.affine_dropouts.append(.5)

        # validate() should throw an ValueError
        try:
            hparams.validate()
        except ValueError:
            pass
        else:
            assert False

        # Invalidate inputs
        hparam_options['filters'].append(64)

        # Constructor should implicitly validate and throw ValueError
        try:
            hparams = SymmetricVAEHyperparams(**hparam_options)
        except ValueError:
            pass
        else:
            assert False

    def test_optimizer_hyperparams(self):
        from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layer = nn.Linear(5, 5)

            def forward(self, x):
                return self.layer(x)

        model = Model()
        name = 'RMSprop'
        hparams = {'lr': 0.9}

        optimizer_hparams = OptimizerHyperparams(name, hparams)

        optimizer_hparams.save(self.optimizer_fname)

        del optimizer_hparams
        import gc
        gc.collect()

        loaded_hparams = OptimizerHyperparams().load(self.optimizer_fname)

        optimizer = get_optimizer(model, loaded_hparams)

    @classmethod
    def teardown_class(self):
        # Delete file to clean testing directories
        os.remove(self.encoder_fname)
        os.remove(self.optimizer_fname)
