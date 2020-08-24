import itertools
from math import ceil, log, sqrt
from molecules.ml.hyperparams import Hyperparams

class ResnetVAEHyperparams(Hyperparams):
    def __init__(self, max_len, nchars, enc_kernel_size=5,
                 latent_dim=150, activation='ReLU',
                 output_activation='Sigmoid',
                 lambda_rec=1.,
                 enc_reslayers=None,
                 scale_factor=2,
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
        self.lambda_rec = lambda_rec
        self.latent_dim = latent_dim
        self.enc_reslayers = enc_reslayers
        self.scale_factor = scale_factor
        self.dec_reslayers = dec_reslayers
        self.dec_kernel_size = dec_kernel_size
        self.dec_filters = dec_filters
        self.dec_filter_growth_rate = dec_filter_growth_rate

        # Below are architecture-specific derived parameters which
        # are not user-settable.

        # TODO: rename these
        self.max_len = max_len
        self.nchars = nchars

        # The number of starting filters we use for the first
        # Conv1D.  Subsequent number of filters are computed
        # from the growth factor, enc_filter_growth_fac
        self.enc_filters = nchars

        # Calculate the number of layers required in the encoder.
        # This is a function of num_residues. We are computing the number
        # of times we need to divide num_residues by two before we get to one.
        # i.e., solving 2^x = num_residues for x.
        if self.enc_reslayers is None:
            self.enc_reslayers = ceil(log(max_len) / log(self.scale_factor))
        else:
            # once we pool down too much, then we can stop
            self.enc_reslayers = min([ceil(log(max_len) / log(self.scale_factor)), self.enc_reslayers])
            
        # Calculate the downsampling factor
        self.downsample_dim = max_len
        for i in range(self.enc_reslayers):
            prev_downsample_dim = self.downsample_dim
            self.downsample_dim = ceil(self.downsample_dim / self.scale_factor)
            if self.downsample_dim == self.latent_dim:
                break
            if self.downsample_dim < self.latent_dim:
                self.downsample_dim = prev_downsample_dim
                break
        
        ## compute dec layers based on this logic
        #for i in itertools.count():
        #    recon_dim = self.downsample_dim * (self.scale_factor ** i)
        #    print(recon_dim)
        #    if recon_dim >= self.max_len:
        #        self.dec_reslayers = i
        #        break
        #    #if recon_dim > self.max_len:
        #    #    raise ValueError(f'Unable to reconstruct downsample_dim {self.downsample_dim} ' \
        #    #                     f'to input size {self.max_len}. Must satisfy ' \
        #    #                     'downsample_dim * (scale_factor^k) == input_size for some ' \
        #    #                     'integer k.')
        
        # Calculate the growth factor required to get to desired
        # hidden dim as a function of enc_reslayers and num_residues.
        # i.e., solving: num_residues * x^enc_reslayers = latent_dim; for x
        ratio = self.downsample_dim / nchars
        self.enc_filter_growth_fac = ratio**(1.0 / (self.enc_reslayers - 1))

        # Think about upsampling / downsampling in the decoder
        # Initialize these flags to 0
        self.upsample_rounds = 0
        self.shrink_rounds = 0

        # Assert that num_residues is a multiple of hidden_dim.
        # This is prevent the need for zero-padding.  We will
        # just use Upsampling.
        if self.downsample_dim < max_len:
            err_msg = 'max_len must be a multiple of downsample_dim'
            assert max_len % self.downsample_dim == 0, err_msg
        # determine the number of upsample layers we need
        rec_size = self.downsample_dim
        for i in itertools.count():
            rec_size *= self.scale_factor
            if rec_size >= max_len:
                self.upsample_rounds = i+1
                break
        #self.upsample_rounds = ceil(max_len / self.downsample_dim - 1)

        # make sure we have at least as many reslayers in decoder as
        # upsample rounds
        self.dec_reslayers = max([self.dec_reslayers, self.upsample_rounds])

        # If we choose a larger hidden_dim, then we must be able
        # to get back to max_len by using a strided conv. So we
        # must confirm that max_len is a multiple of hidden_dim
        if self.downsample_dim > max_len:
            err_msg = 'downsample_dim must be a multiple of max_len'
            assert self.downsample_dim % max_len == 0, err_msg

            self.shrink_rounds = self.downsample_dim / max_len

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        pass

