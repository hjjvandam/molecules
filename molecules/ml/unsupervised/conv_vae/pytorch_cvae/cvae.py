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
                                     nn.Flatten(),
                                     *self._affine_layers())

        self.z_mean = Dense(nn.Linear(in_features=self.hparams.affine_widths[-1],
                                 out_features=self.hparams.latent_dim))
        self.z_logvar = Dense(nn.Linear(in_features=self.hparams.affine_widths[-1],
                                 out_features=self.hparams.latent_dim))

    def reparameterize(self):
        std = torch.exp(0.5*self.z_logvar)
        eps = torch.randn_like(std)
        return self.z_mean + eps*std

    def forward(self, x):
        x = self.encoder(x)
        x = self.reparameterize(x)
        return x

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []
        for i, (filter_, kernel, stride) in enumerate(zip(self.hparams.filters,
                                                          self.hparams.kernels,
                                                          self.hparams.strides)):

            l = nn.Conv2d(in_channels= 1 if not i else self.hparams.filters[i - 1],
                          out_channels=filter_,
                          kernel_size=kernel,
                          stride=stride,
                          padding=same_padding(self.input_shape[0], kernel, stride)
                          )
            conv2d_layers.append(l)

        return conv2d_layers

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        fc_layers : list
            Linear layers
        """

        fc_layers = []
        for i, (width, dropout) in enumerate(zip(self.hparams.affine_widths, self.hparams.affine_dropouts)):

            in_features = self.input_shape[0]**2 * self.hparams.filters[-1] if not i else self.hparams.affine_widths[i - 1]

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(nn.Dropout(p=dropout))

        return fc_layers


# TODO: add activation


class DecoderConvolution2D(nn.Module):
    def __init__(self, input_shape, hyperparameters=DecoderHyperparams()):
        super(DecoderConvolution2D, self).__init__()

        hyperparameters.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.hparams = hyperparameters

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers = nn.Sequential(*self._conv_layers()) 

    def reshape(self, x):
        new_shape = (*self.output_shape, self.hparams.affine_widths[-1] / self.output_shape[0]**2)
        return x.view(new_shape)

    def forward(self, x):
        x = self.affine_layers(x)
        x = self.reshape(x)
        x = self.conv_layers(x)
        return x

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []
        for i, (filter_, kernel, stride) in enumerate(zip(self.hparams.filters,
                                                          self.hparams.kernels,
                                                          self.hparams.strides)):

            l = nn.Conv2d(in_channels= 1 if not i else self.hparams.filters[i - 1],
                          out_channels=filter_,
                          kernel_size=kernel,
                          stride=stride,
                          padding=same_padding(self.input_shape[0], kernel, stride)
                          )
            conv2d_layers.append(l)

        return conv2d_layers

    def _affine_layers(self):
        """
        Compose affine layers.

        Returns
        -------
        fc_layers : list
            Linear layers
        """

        fc_layers = []
        for i, (width, dropout) in enumerate(zip(self.hparams.affine_widths, self.hparams.affine_dropouts)):

            in_features = self.hparams.latent_dim if not i else self.hparams.affine_widths[i - 1]

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(nn.Dropout(p=dropout))

        return fc_layers



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