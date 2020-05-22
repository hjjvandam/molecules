from molecules.ml.hyperparams import Hyperparams
from math import ceil, log, sqrt

class ResnetVAEHyperparams(Hyperparams):
    def __init__(self, input_shape, kernel_size=3, latent_dim=10, activation='ReLU'):

        # User defined hyperparams
        self.kernel_size = kernel_size
        self.activation = activation
        self.latent_dim = latent_dim

        # Below are architecture-specific derived parameters which
        # are not user-settable.

        # input_shape is of dimension (1, N*N) where N is
        # the number of residues
        num_residues = input_shape[1] #int(sqrt(input_shape[1]))

        # The number of starting filters we use for the first
        # Conv1D.  Subsequent number of filters are computed
        # from the growth factor, enc_filter_growth_fac
        self.enc_filters = num_residues

        # Calculate the number of layers required in the encoder.
        # This is a function of num_residues. We are computing the number
        # of times we need to divide num_residues by two before we get to one.
        # i.e., solving 2^x = num_residues for x.
        self.enc_reslayers = ceil(log(num_residues) / log(2))

        # Calculate the growth factor required to get to desired
        # hidden dim as a function of enc_reslayers and num_residues.
        # i.e., solving: num_residues * x^enc_reslayers = latent_dim; for x
        ratio = self.latent_dim / num_residues
        self.enc_filter_growth_fac = ratio**(1.0 / (self.enc_reslayers - 1))

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        pass