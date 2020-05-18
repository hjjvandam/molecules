import torch
from torch import nn, optim
from torch.nn import functional as F
from math import isclose
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer
from molecules.ml.unsupervised import EncoderHyperparams, DecoderHyperparams

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

def conv2d_output_dim(input_dim, kernel_size, stride, padding,
                      transpose=False):
    """
    Parameters
    ----------
    input_dim : int
        input size. may include padding
    
    kernel_size : int
        filter size

    stride : int
        stride length

    padding : int
        length of 0 pad

    """

    if transpose:
        return (input_dim - 1) * stride + kernel_size - 2*padding

    return (2*padding + input_dim - kernel_size) // stride + 1

def conv2d_output_shape(input_dim, kernel_size, stride, padding,
                        num_filters, transpose=False):
    """
    Parameters
    ----------
    input_dim : int
        input size
    
    kernel_size : int
        filter size

    stride : int
        stride length

    padding : int
        length of 0 pad

    num_filters : int
        number of filters


    Returns
    -------
    (N,N, num_filters) tuple

    """
    output_dim = conv2d_output_dim(input_dim, kernel_size, stride,
                                   padding, transpose)
    return (output_dim, output_dim, num_filters)

def same_padding(input_dim, kernel_size, stride):
    """
    Parameters
    ----------
    n : int
        input size
    
    kernel_size : int
        filter size

    stride : int
        stride length

    Effects
    -------
    In this case we want output_dim = input_dim

    input_dim = output_dim = (2*pad + input_dim - kernel_size) // stride + 1

    Solve for pad. 
    """
    return (input_dim * (stride - 1) - stride + kernel_size) // 2

def even_padding(input_dim, kernel_size, stride):
    """
    Implements Keras-like padding.
    If the stride is one then use same_padding.
    Otherwise, select the smallest pad such that the
    kernel_size fits evenly within the input_dim.
    """
    if stride == 1:
        return same_padding(input_dim, kernel_size, stride)

    # TODO: this could have bugs for stride != 1. Needs testing.
    return same_padding(input_dim // stride, kernel_size, 1)


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
    if activation is 'ReLU':
        return nn.ReLU()
    elif activation is 'Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation type: {activation}')


# TODO: consider making _conv_layers and _affine_layers helper functions


class EncoderConv2D(nn.Module):
    def __init__(self, input_shape, hparams):
        super(EncoderConv2D, self).__init__()

        hparams.validate()

        # Assume input is square matrix
        self.input_shape = input_shape
        self.hparams = hparams
        self.encoder_dim = input_shape[1]

        self.encoder = nn.Sequential(*self._conv_layers(),
                                     nn.Flatten(),
                                     *self._affine_layers())

        self.mu = self._embedding_layer()
        self.logvar = self._embedding_layer()

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

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
        in_channels = self.input_shape[0]

        for filter_, kernel, stride in zip(self.hparams.filters,
                                           self.hparams.kernels,
                                           self.hparams.strides):

            padding = even_padding(self.encoder_dim, kernel, stride)

            conv2d_layers.append(nn.Conv2d(in_channels=in_channels,
                                           out_channels=filter_,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=padding))

            conv2d_layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            in_channels = filter_

            self.encoder_dim = conv2d_output_dim(self.encoder_dim, kernel, stride, padding)

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
        in_features = self.hparams.filters[-1] * self.encoder_dim**2

        for width, dropout in zip(self.hparams.affine_widths,
                                  self.hparams.affine_dropouts):

            fc_layers.append(nn.Linear(in_features=in_features,
                                       out_features=width))

            fc_layers.append(select_activation(self.hparams.activation))

            if not isclose(dropout, 0):
                fc_layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        return fc_layers

    def _embedding_layer(self):
        return nn.Linear(in_features=self.hparams.affine_widths[-1],
                         out_features=self.hparams.latent_dim)


class DecoderConv2D(nn.Module):
    def __init__(self, output_shape, encoder_dim, hparams):
        super(DecoderConv2D, self).__init__()

        hparams.validate()

        # Assume input is square matrix
        self.output_shape = output_shape
        self.encoder_dim = encoder_dim
        self.hparams = hparams

        self.affine_layers = nn.Sequential(*self._affine_layers())
        self.conv_layers = nn.Sequential(*self._conv_layers())

    def reshape(self, x):
        """
        Reshape flattened x as a tensor (channels, output, output)
        """
        new_shape = (-1, self.hparams.filters[0],
                     self.encoder_dim, self.encoder_dim)
        return x.view(new_shape)

    def forward(self, x):
        x = self.affine_layers(x)
        x = self.reshape(x)
        return self.conv_layers(x)

    def _conv_layers(self):
        """
        Compose convolution layers.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []

        in_channels = self.hparams.filters[0]

        # Dimension of square matrix
        input_dim = self.output_shape[1]

        # Set last filter to be the number of channels in the reconstructed image.
        self.hparams.filters[-1] = self.output_shape[0]

        for filter_, kernel, stride in zip(self.hparams.filters,
                                           self.hparams.kernels,
                                           self.hparams.strides):

            padding = even_padding(input_dim, kernel, stride)

            conv2d_layers.append(nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=filter_,
                                                    kernel_size=kernel,
                                                    stride=stride,
                                                    padding=padding,
                                                    output_padding=1 if stride != 1 else 0))

            # TODO: revist output_padding. This code may not generalize to other examples. Needs testing.

            conv2d_layers.append(select_activation(self.hparams.activation))

            # Subsequent layers in_channels is the current layers number of filters
            # Except for the last layer which is 1 (or output_shape channels)
            in_channels = filter_

            # Compute non-channel dimension given to next layer
            input_dim = conv2d_output_dim(input_dim, kernel, stride, padding, transpose=True)

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

            if not isclose(dropout, 0):
                fc_layers.append(nn.Dropout(p=dropout))

            # Subsequent layers in_features is the current layers width
            in_features = width

        # Add last layer with dims to connect the last linear layer to
        # the first convolutional decoder layer
        fc_layers.append(nn.Linear(in_features=self.hparams.affine_widths[-1],
                                   out_features=self.hparams.filters[0] * self.encoder_dim**2))
        fc_layers.append(select_activation(self.hparams.activation))


        return fc_layers

# Helpful links:
#   https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
#   https://pytorch.org/tutorials/beginner/saving_loading_models.html



class CVAEModel(nn.Module):
    def __init__(self, input_shape, encoder_hparams, decoder_hparams, device):
        super(CVAEModel, self).__init__()

        # May not need these .to(device) since it is called in CVAE. Test this
        self.encoder = EncoderConv2D(input_shape, encoder_hparams).to(device)
        self.decoder = DecoderConv2D(input_shape, self.encoder.encoder_dim,
                                     decoder_hparams).to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        x = self.reparameterize(mu, logvar)
        x = self.decoder(x)
        # TODO: having issue with nans in BCE loss function. Try weight initialization.
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x, mu, logvar

    # TODO: wrap embed, generate in context manager with no_grad() ?

    def embed(self, data):
        # mu layer
        return self.encoder(data)[0]

    def generate(self, embedding):
        return self.decoder(embedding)

    def save_weights(self, enc_path, dec_path):
        torch.save(self.encoder, enc_path)
        torch.save(self.decoder, dec_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder = torch.load(enc_path)
        self.decoder = torch.load(dec_path)


class CVAE:
    # TODO: set weight initialization hparams
    def __init__(self, input_shape,
                 encoder_hparams=EncoderHyperparams(),
                 decoder_hparams=DecoderHyperparams(),
                 optimizer_hparams=OptimizerHyperparams(),
                 loss=None):

        optimizer_hparams.validate()

        self.input_shape = input_shape

        self.loss = self._loss if loss is None else loss

        # TODO: consider passing in device, or giving option to run cpu even if cuda is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cvae = CVAEModel(input_shape, encoder_hparams,
                              decoder_hparams, self.device).to(self.device)

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer = get_optimizer(self.cvae, optimizer_hparams)

    def __repr__(self):
        return str(self.cvae)

    def train(self, train_loader, test_loader, epochs):
        for epoch in range(1, epochs + 1):
            self._train(train_loader, epoch)
            self._test(test_loader)

    def _train(self, train_loader, epoch,
               checkpoint=None, callbacks=None,
               log_interval=2):
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
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.cvae(data)
            loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()


            # TODO: Handle logging/checkpoint
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))


    def _test(self, test_loader):
        self.cvae.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.cvae(data)
                test_loss += self.loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))


    def _loss(self, recon_x, x, mu, logvar):
        """
        Effects
        -------
        Reconstruction + KL divergence losses summed over all elements and batch

        See Appendix B from VAE paper:
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        """
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD


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
