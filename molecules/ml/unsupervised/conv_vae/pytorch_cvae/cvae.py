from torch import nn, optim
from torch.nn import functional as F

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


# TODO: conider moving this into the HyperParams class
#       could make base classes for ModelArchHyperParams
#       which handles layes and can return pytorch layer
#       types and activations. OptimizerHyperParams can
#       return different optimizers.
def select_activation(activation):
    """
    Parameters
    ----------
    activation : str
        type of activation e.g. 'ReLU', etc

    """
    activation = activation.lower()
    if activation is 'relu':
        return nn.ReLU()
    elif activation is 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation type: {activation}')


# TODO: consider making _conv_layers and _affine_layers helper functions


class EncoderConv2D(nn.Module):
    def __init__(self, input_shape, hparams):
        super(EncoderConvolution2D, self).__init__()

        hparams.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hparams

        self.encoder = nn.Sequential(*self._conv_layers(),
                                     nn.Flatten(),
                                     *self._affine_layers())

        self.z_mean = self._embedding_layer()
        self.z_logvar = self._embedding_layer()

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

        # Contact matrices have one channel
        in_channels = 1

        for filter_, kernel, stride in zip(self.hparams.filters,
                                           self.hparams.kernels,
                                           self.hparams.strides):

            padding = same_padding(self.input_shape[0], kernel, stride)

            conv2d_layers.append(nn.Conv2d(in_channels=in_channels,
                                           out_channels=filter_,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=padding))

            conv2d_layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            in_channels = filter_

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

        # First layer gets flattened convolutional output
        in_features = self.input_shape[0]**2 * self.hparams.filters[-1]

        for width, dropout in zip(self.hparams.affine_widths,
                                  self.hparams.affine_dropouts):

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(select_activation(self.hparams.activation))

            fc_layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return fc_layers

    def _embedding_layer(self):
        return Dense(nn.Linear(in_features=self.hparams.affine_widths[-1],
                               out_features=self.hparams.latent_dim))


class DecoderConv2D(nn.Module):
    def __init__(self, output_shape, hparams):
        super(DecoderConvolution2D, self).__init__()

        hparams.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.hparams = hparams

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers = nn.Sequential(*self._conv_layers())

    def reshape(self, x):
        """
        Reshape flattened x as a tensor (output, output, remainder)
        """
        new_shape = (*self.output_shape, self.hparams.affine_widths[-1] / self.output_shape[0]**2)
        return x.view(new_shape)

    def forward(self, x):
        x = self.affine_layers(x)
        x = self.reshape(x)
        x = self.conv_layers(x)
        return x


    # TODO: _conv_layers could have a bug in the in_channels. Needs testing.
    #       Check that the output dimension is correct

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []

        in_channels = self.hparams.affine_widths[-1] / self.output_shape[0]**2

        for filter_, kernel, stride in zip(self.hparams.filters,
                                           self.hparams.kernels,
                                           self.hparams.strides):

            padding = same_padding(self.input_shape[0], kernel, stride)

            conv2d_layers.append(nn.Conv2d(in_channels=in_channels,
                                           out_channels=filter_,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=padding))

            conv2d_layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            in_channels = filter_

        # Overwrite output activation
        conv2d_layers[-1] = select_activation(self.hparams.output_activation)

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

        in_features = self.hparams.latent_dim

        for width, dropout in zip(self.hparams.affine_widths,
                                  self.hparams.affine_dropouts):

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(select_activation(self.hparams.activation))

            fc_layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return fc_layers



# Helpful links:
#   https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
#   https://pytorch.org/tutorials/beginner/saving_loading_models.html



class CVAEModel(nn.Module):
    def __init__(self, input_shape, encoder_hparams, decoder_hparams):
        super(CVAE, self).__init__()

        self.encoder = EncoderConv2D(input_shape, encoder_hparams)
        self.decoder = DecoderConv2D(input_shape, decoder_hparams)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embed(self, data):
        return self.encoder(data)

    def generate(self, embedding):
        return self.decoder(embedding)

    def save_weights(self, enc_path, dec_path):
        torch.save(self.encoder, enc_path)
        torch.save(self.decoder, dec_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder = torch.load(enc_path)
        self.decoder = torch.load(dec_path)



class CVAE:
    def __init__(self, input_shape,
                 encoder_hparams=EncoderHyperparams(),
                 decoder_hparams=DecoderHyperparams(),
                 optimizer_hparams=OptimzerHyperparams(),
                 loss=None):

        optimizer_hparams.validate()

        self.optimizer_hparams = optimizer_hparams

        self.loss = self._loss if loss is None else loss

        self.cvae = CVAEModel(input_shape, encoder_hparams, decoder_hparams)

    def train(self, train_loader, batch_size, epochs=1, shuffle=False,
              valid_loader=None, checkpoint=None, callbacks=None):
        """
        Effects
        -------
        Train network

        Parameters
        ----------
        data : np.ndarray
            Input data

        batch_size : int
            Minibatch size

        epochs : int
            Number of epochs to train for

        shuffle : bool
            Whether to shuffle training data.

        validation_data : tuple, optional
            Tuple of np.ndarrays (X,y) representing validation data

        checkpoint : str
            Path to save model after each epoch. If none is provided,
            then checkpoints will not be saved.

        """
        self.cvae.train()
        # TODO: make optimizer_hparams.get_optimizer(model.parameters()) 
        #       function which returns the optimizer given the hparams
        optimizer = optim.RMSprop(self.cvae.parameters(), lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0)
        pass

    def embed(self, data):
        """
        Effects
        -------
        Embed a datapoint into the latent space.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray of embeddings.

        """
        return self.cvae.embed(data)

    def generate(self, embedding):
        """
        Effects
        -------
        Generate images from embeddings.

        Parameters
        ----------
        embedding : np.ndarray

        Returns
        -------
        generated image : np.nddary

        """
        return self.cvae.generate(embedding)

    def save_weights(self, enc_path, dec_path):
        """
        Effects
        -------
        Save model weights.

        Parameters
        ----------
        path : str
            Path to save the model weights.

        """
        self.cvae.save_weights(enc_path, dec_path)

    def load_weights(self, enc_path, dec_path):
        """
        Effects
        -------
        Load saved model weights.

        Parameters
        ----------
        path: str
            Path to saved model weights.

        """
        self.cvae.load_weights(enc_path, dec_path)

    def _loss(self, recon_x, x, mu, logvar):
        """
        Effects
        -------
        Reconstruction + KL divergence losses summed over all elements and batch

        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        """
        # TODO: fix view
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
