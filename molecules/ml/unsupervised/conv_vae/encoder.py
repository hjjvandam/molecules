import gc
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Convolution2D

from .hyperparams import HyperparamsEncoder


class EncoderConvolution2D:

    def __init__(self, input_shape, hyperparameters=HyperparamsEncoder()):
        self.input_shape = input_shape
        self.input = Input(shape=input_shape)
        self.hparams = hyperparameters
        self.graph = self.create_graph(self.input)
        self.embedder = Model(self.input, self.z_mean)

    def __repr__(self):
        return '2D Convolutional Encoder.'

    def summary(self):
        print('Convolutional Encoder:')
        self.embedder.summary()

    def embed(self, data):
        """
        Embed a datapoint into the latent space.

        Parameters
        ----------
        data : np.ndarray

        Returns
        -------
        np.ndarray of embeddings.
        """
        return self.embedder.predict(data)

    def _conv_layers(self, x):
        """
        Compose convolution layers.

        Parameters
        ----------
        x : keras.layers.Input
            Shape of the image input.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """
        conv2d_layers = []
        for filter_, kernel, stride in zip(self.hparams.filters, 
                                           self.hparams.kernels, 
                                           self.hparams.strides):

            x = Convolution2D(filter_, kernel, strides=stride,
                              activation=self.hparams.activation,
                              padding='same')(x)
            conv2d_layers.append(x)

        del x
        gc.collect()

        return conv2d_layers

    def _affine_layers(self, x):
        """
        Compose fully connected layers.

        Parameters
        ----------
        x : tensorflow Tensor
            Flattened tensor from convolution layers.

        Returns
        -------
        fc_layers : list
            Fully connected layers for embedding.
        """

        fc_layers = []
        for width, dropout in zip(self.hparams.affine_widths, self.hparams.affine_dropouts):
            x = Dense(width, activation=self.hparams.activation)(Dropout(dropout)(x))
            fc_layers.append(x)

        del x
        gc.collect()

        self.z_mean = Dense(self.hparams.latent_dim)(fc_layers[-1])
        self.z_log_var = Dense(self.hparams.latent_dim)(fc_layers[-1])
        self.z = Lambda(self.sampling, output_shape=(self.hparams.latent_dim,))([self.z_mean, self.z_log_var])

        embed = self.z
        return embed

    def sampling(self, args):
        """
        Reparameterization trick by sampling fr an isotropic unit Gaussian.

        Parameters
        ----------
        encoder_output : tensor
            Mean and log of variance of Q(z|X)

        Returns
        -------
        z : tensor
            Sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def create_graph(self, input_):
        """Create keras model outside of class"""
        self.conv_layers = self._conv_layers(input_)
        flattened = Flatten()(self.conv_layers[-1])
        z = self._affine_layers(flattened)
        return z

    def get_final_conv_params(self):
        """Return the number of flattened parameters from final convolution layer."""
        input_ = np.ones((1, *self.input_shape))
        dummy = Model(self.input, self.conv_layers[-1])
        conv_shape = dummy.predict(input_).shape
        encode_conv_shape = conv_shape[1:]
        num_conv_params = np.prod(conv_shape)
        return encode_conv_shape, num_conv_params
