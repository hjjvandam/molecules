from torch import nn

def conv2d_num_params(f, c, num_filters):
    """
    Parameters
    ----------    
    f : int
        filter size

    c : int
        number of channels

    num_filters : int
        number of filters
    """
    # +1 to count the bias for each filter
    return (f*f*c + 1) * num_filters


def conv2d_output_size(n, f, stride, pad):
    """
    Parameters
    ----------
    n : int
        input size. may include padding
    
    f : int
        filter size

    stride : int
        stride length

    pad : int
        length of 0 pad

    """
    return (2*pad + n - f) // stride + 1


def conv2d_output_shape(n, f, stride, pad, num_filters):
    """
    Parameters
    ----------
    n : int
        input size
    
    f : int
        filter size

    stride : int
        stride length

    pad : int
        length of 0 pad

    num_filters : int
        number of filters


    Returns
    -------
    (N,N, num_filters) tuple

    """
    output_size = conv2d_output_size(n, f, stride, pad) 
    return (output_size, output_size, num_filters)

def same_padding(input_dim, f, stride):
    """
    Parameters
    ----------
    n : int
        input size
    
    f : int
        filter size

    stride : int
        stride length

    Effects
    -------
    In this case we want output_dim = input_dim

    input_dim = output_dim = (2*pad + input_dim - f) / stride + 1

    Solve for pad. 
    """
    return (input_dim * (stride - 1) - stride + f) // 2


class EncoderConvolution2D(nn.Module):
    def __init__(self, input_shape, hyperparameters=EncoderHyperparams()):
        super(EncoderConvolution2D, self).__init__()

        hyperparameters.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hyperparameters

        self.encoder = nn.Sequential(*self._conv_layers(),
                                     *self._affine_layers(),
                                     )

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []
        # for filter_, kernel, stride in zip(self.hparams.filters, 
        #                                    self.hparams.kernels, 
        #                                    self.hparams.strides):

        #     l = nn.Conv2d()
        #     conv2d_layers.append(l)

        conv2d_layers.append(nn.Conv2d(in_channels=1, 
                                       out_channels=self.hparams.filters[0],
                                       kernel_size=self.hparams.kernels[0],
                                       stride=self.hparams.strides[0],
                                       padding=same_padding(self.input_shape[0],
                                                            self.hparams.kernels[0],
                                                            self.hparams.strides[0])))

        # TODO: may only need to compute a single pad if we use same_padding 
        #       for each layer, otherwise use conv2d_output_size. Think more on this.

        conv2d_layers.append(nn.Conv2d(in_channels=self.hparams.kernels[0], 
                                       out_channels=self.hparams.filters[1],
                                       kernel_size=self.hparams.kernels[1],
                                       stride=self.hparams.strides[1],
                                       padding=same_padding(self.input_shape[0],
                                                            self.hparams.kernels[0],
                                                            self.hparams.strides[0])))

        return conv2d_layers


# Helpful links:
#   https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
#   https://pytorch.org/tutorials/beginner/saving_loading_models.html















class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x