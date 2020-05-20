from molecules.ml.hyperparams import Hyperparams

class ConvVAEHyperparams(Hyperparams):
    def __init__(self, filters=[64, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 2, 1], latent_dim=3,
                 affine_widths=[128], affine_dropouts=[0],
                 activation='ReLU', output_activation='Sigmoid'):

        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.latent_dim = latent_dim
        self.affine_widths = affine_widths
        self.affine_dropouts = affine_dropouts
        self.activation = activation
        self.output_activation = output_activation

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        num_conv_layers = len(self.filters)

        if any(num_conv_layers != len(param) for param in (self.kernels, self.strides)):
            raise ValueError('Number of filters, kernels and strides must be equal.')

        if len(self.affine_dropouts) != len(self.affine_widths):
            raise ValueError('Number of dropout parameters must equal the number of affine widths.')

        # Common convention: allows for filter center and for even padding
        if any(kernel % 2 == 0 for kernel in self.kernels):
            raise ValueError('Only odd valued kernel sizes allowed.')

        if any(p < 0 or p > 1 for p in self.affine_dropouts):
            raise ValueError('Dropout probabilities, p, must be 0 <= p <= 1.')
