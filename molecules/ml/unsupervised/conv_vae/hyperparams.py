from molecules.ml.hyperparams import HyperParams


class HyperParamsConvVAE(HyperParams):
    def __init__(self, num_conv_layers, filters, kernels,
                 strides, latent_dim, activation, num_affine_layers,
                 affine_widths):
        super().__init__()
        self.num_conv_layers = num_conv_layers 
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.latent_dim = latent_dim
        self.activation = activation
        self.num_affine_layers = num_affine_layers
        self.affine_widths = affine_widths

        self.__validate()

    def __validate(self):
        if len(self.filters) != self.num_conv_layers:
            raise Exception('number of filters must equal number of convolutional layers.')
        if len(self.kernels) != self.num_conv_layers:
            raise Exception('number of kernels must equal number of convolutional layers.')
        if len(self.strides) != self.num_conv_layers:
            raise Exception('number of strides must equal number of convolutional layers.')
        if len(self.affine_widths) != self.num_affine_layers:
            raise Exception('number of affine width parameters must equal the number of affine layers')


class HyperparamsEncoder(HyperParamsConvVAE):

    def __init__(self, num_conv_layers=3, filters=[64, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 2, 1], latent_dim=3, activation='relu', 
                 num_affine_layers=1, affine_widths=[128], affine_dropouts=[0]):

        super().__init__(num_conv_layers, filters, kernels, strides, 
                         latent_dim, activation, num_affine_layers,
                         affine_widths)

        self.affine_dropouts = affine_dropouts

        self.__validate()

    def __validate(self):
        if len(self.affine_dropouts) != self.num_affine_layers:
            raise Exception('number of dropout parameters must equal the number of affine layers')


class HyperparamsDecoder(HyperParamsConvVAE):

    def __init__(self, num_conv_layers=3, filters=[64, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 2, 1], latent_dim=3, activation='relu', 
                 num_affine_layers=1, affine_widths=[128],
                 output_activation='sigmoid'):

        super().__init__(num_conv_layers, filters, kernels, strides, 
                         latent_dim, activation, num_affine_layers,
                         affine_widths)

        self.output_activation = output_activation
