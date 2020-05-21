from molecules.ml.hyperparams import Hyperparams
from math import ceil, log

class ResnetVAEHyperparams(Hyperparams):
    def __init__(self, input_shape, kernel_size=3, latent_dim=10, activation='ReLU'):

        # User defined hyperparams
        self.kernel_size = kernel_size
        self.activation = activation
        self.latent_dim = latent_dim

        # Below are architecture-specific derived parameters which
        # are not user-settable.

        # the number of starting filters we use for the first
        # Conv1D.  Subsequent number of filters are computed
        # from the growth factor, enc_filter_growth_fac
        self.enc_filters = input_shape[1]


        # calculate the number of layers required in the encoder.
        # This is a function of MAX_LEN.  We are computing the number
        # of times we need to divide MAX_LEN by two before we get to one.
        # i.e., solving 2^x = MAX_LEN for x.
        self.enc_reslayers = ceil(log(input_shape[1]) / log(2))

        # calculate the growth factor required to get to desired
        # hidden dim as a function of enc_reslayers and NCHARS.
        # i.e., solving: NCHARS * x^enc_reslayers = latent_dim; for x
        ratio = self.latent_dim / input_shape[1]
        self.enc_filter_growth_fac = ratio**(1.0 / (self.enc_reslayers - 1))

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        pass