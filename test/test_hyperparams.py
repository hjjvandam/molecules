"""
Tests for `molecules` module.
"""
import os
import pytest
from molecules.ml.unsupervised import HyperparamsEncoder, HyperparamsDecoder


class TestHyperParams:

    @classmethod
    def setup_class(cls):
        cls.fname = os.path.join('.', 'test', 'data', 'encoder-hparams.pkl')

    def test_save_load_functions(self):
        # Set model hyperparameters for encoder
        hparam_options = {'num_conv_layers': 4,
                          'filters': [64, 64, 64, 64],
                          'kernels': [3, 3, 3, 3],
                          'strides': [1, 2, 1, 1],
                          'num_affine_layers': 1,
                          'affine_widths': [128],
                          'latent_dim': 3,
                          'affine_dropouts': [0]
                         }
        encoder_hparams = HyperparamsEncoder(**hparam_options)

        # Save model hyperparameters to disk
        encoder_hparams.save(self.fname)

        # Check that 'encoder-hparams.pkl' is in ./test/data
        assert os.path.basename(self.fname) \
            in os.listdir(os.path.dirname(self.fname))

        # Load saved hyperparameters from disk
        loaded_hparams = HyperparamsEncoder.load(self.fname)

        # Check that all attributes were read from disk correctly
        for key, val in hparam_options.items():
            assert getattr(loaded_hparams, key) == val

    def test_validators(self):
        # Set model hyperparameters for encoder and decoder
        shared_hparams = {'num_conv_layers': 4,
                          'filters': [64, 64, 64, 64],
                          'kernels': [3, 3, 3, 3],
                          'strides': [1, 2, 1, 1],
                          'num_affine_layers': 1,
                          'affine_widths': [128],
                          'latent_dim': 3
                         }

        affine_dropouts = [0]

        encoder_hparams = HyperparamsEncoder(affine_dropouts=affine_dropouts,
                                             **shared_hparams)
        decoder_hparams = HyperparamsDecoder(**shared_hparams)

        # Raises exception if invalid
        encoder_hparams.validate()
        decoder_hparams.validate()

        # Invalidate state
        encoder_hparams.num_conv_layers = 2

        # validate() should throw an Exception
        try:
            encoder_hparams.validate()
        except Exception:
            pass
        else:
            assert False

    @classmethod
    def teardown_class(cls):
        # Delete file to clean testing directories
        os.remove(cls.fname)
