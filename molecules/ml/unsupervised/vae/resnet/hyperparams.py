from molecules.ml.hyperparams import Hyperparams
from math import ceil, log, sqrt

class ResnetVAEHyperparams(Hyperparams):
    def __init__(self, input_shape, enc_kernel_size=5,
                 latent_dim=150, activation='ReLU',
                 output_activation='Sigmoid',
                 dec_reslayers=3, dec_kernel_size=5,
                 dec_filters=1200, dec_filter_growth_rate=1.0):

        """
        Parameters
        ----------
        enc_kernel_size : int
            conv kernel size for encoder

        dec_reslayers : int
            number of residual layer blocks for decoder

        dec_filters : int
            number of conv filters for decoder

        dec_filter_growth_rate : float
            filter growth factor for decoder
        """

        # User defined hyperparams
        self.enc_kernel_size = enc_kernel_size
        self.activation = activation
        self.output_activation = output_activation
        self.latent_dim = latent_dim
        self.dec_reslayers = dec_reslayers
        self.dec_kernel_size = dec_kernel_size
        self.dec_filters = dec_filters
        self.dec_filter_growth_rate = dec_filter_growth_rate

        # Below are architecture-specific derived parameters which
        # are not user-settable.

        # input_shape is of dimension (N, N) where N is
        # the number of residues
        num_residues = input_shape[0]

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

        # Think about upsampling / downsampling in the decoder
        # Initialize these flags to 0
        self.upsample_rounds = 0
        self.shrink_rounds = 0

        # Assert that num_residues is a multiple of hidden_dim.
        # This is prevent the need for zero-padding.  We will
        # just use Upsampling.
        if self.latent_dim < num_residues:
            err_msg = 'num_residues must be a multiple of latent_dim'
            assert num_residues % self.latent_dim == 0, err_msg

            self.upsample_rounds = num_residues / self.latent_dim - 1

        # If we choose a larger hidden_dim, then we must be able
        # to get back to num_residues by using a strided conv. So we
        # must confirm that num_residues is a multiple of hidden_dim
        if self.latent_dim > num_residues:
            err_msg = 'latent_dim must be a multiple of num_residues'
            assert self.latent_dim % num_residues == 0, err_msg

            self.shrink_rounds = self.latent_dim / num_residues

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        pass