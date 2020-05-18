import gc
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Flatten, Dropout, Convolution2D
from .hyperparams import EncoderHyperparams


class EncoderConvolution2D:

    def __init__(self, input_shape, hyperparameters=EncoderHyperparams()):
        hyperparameters.validate()
        self.input_shape = input_shape
        self.hparams = hyperparameters
        self.input = Input(shape=self.input_shape, name='encoder_input')
        self.embedder = self._create_graph()

    def __repr__(self):
        return '2D Convolutional Encoder.'

    def summary(self):
        print(self)
        self.embedder.summary()

    def save_weights(self, path):
        self.embedder.save_weights(path)

    def load_weights(self, path):
        self.embedder.load_weights(path)

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
                              activation=self.hparams.activation.lower(),
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
            x = Dense(width, activation=self.hparams.activation.lower())(Dropout(dropout)(x))
            fc_layers.append(x)

        del x
        gc.collect()

        z_mean = Dense(self.hparams.latent_dim)(fc_layers[-1])
        z_log_var = Dense(self.hparams.latent_dim)(fc_layers[-1])
        z = Lambda(self.sampling, output_shape=(self.hparams.latent_dim,))([z_mean, z_log_var])

        return z_mean, z_log_var, z

    def sampling(self, args):
        """
        Reparameterization trick by sampling for an isotropic unit Gaussian.

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

    def _create_graph(self):
        """Create keras model outside of class"""
        conv_layers = self._conv_layers(self.input)
        flattened = Flatten()(conv_layers[-1])
        # Member vars z_mean, z_log_var used to calculate vae loss
        self.z_mean, self.z_log_var, z = self._affine_layers(flattened)
        embedder = Model(self.input, outputs=[self.z_mean, self.z_log_var, z], name='embedder')
        return embedder

    def get_final_conv_params(self):
        """Return the number of flattened parameters from final convolution layer."""
        dummy = Model(self.input, self._conv_layers(self.input)[-1])
        dummy_input = np.ones((1, *self.input_shape))
        conv_shape = dummy.predict(dummy_input).shape
        encode_conv_shape = conv_shape[1:]
        num_conv_params = np.prod(conv_shape)
        return encode_conv_shape, num_conv_params
