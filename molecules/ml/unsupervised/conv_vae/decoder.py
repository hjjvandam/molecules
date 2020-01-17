import gc
import warnings
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose

from .hyperparams import DecoderHyperparams

class DecoderConvolution2D:

    def __init__(self, output_shape, enc_conv_params, 
                 enc_conv_shape, hyperparameters=DecoderHyperparams()):
        hyperparameters.validate()
        self.output_shape = output_shape
        self.enc_conv_params = enc_conv_params
        self.enc_conv_shape = enc_conv_shape
        self.hparams = hyperparameters
        self.generator = self._create_graph()

    def __repr__(self):
        return '2D Convolutional Decoder.'

    def summary(self):
        print(self)
        return self.generator.summary()

    def save_weights(self, path):
        self.generator.save_weights(path)

    def load_weights(self, path):
        self.generator.load_weights(path)

    def generate(self, embedding):
        """
        Generate images from embeddings.

        Parameters
        ----------
        embedding : np.ndarray

        Returns
        -------
        generated image : np.nddary
        """
        self.generator.predict(embedding)

    def _affine_layers(self, x):
        """
        Compose fully connected layers.

        Parameters
        ----------
        x : tensor
            Input from latent dimension.

        Returns
        -------
        fc_layers : list
            Fully connected layers from embedding to convolution layers.
        """
        fc_layers = []
        for width in self.hparams.affine_widths:
            x = Dense(width, activation=self.hparams.activation)(x)
            fc_layers.append(x)

        # Since the networks are symmetric, we need a Dense layer to bridge fc layers and conv.
        x = Dense(self.enc_conv_params, activation=self.hparams.activation)(x)
        fc_layers.append(x)

        del x
        gc.collect()

        return fc_layers

    def _conv_layers(self, x):
        """
        Compose convolution layers.

        Parameters
        ----------
        x : tensorflow tensor
            Shape of the image input.

        Returns
        -------
        conv2d_layers : list
            Convolution layers
        """

        conv2d_layers = []
        for i in range(self.hparams.num_conv_layers - 1):
            x = Conv2DTranspose(self.hparams.filters[i],
                                self.hparams.kernels[i],
                                strides=self.hparams.strides[i],
                                activation=self.hparams.activation,
                                padding='same')(x)
            conv2d_layers.append(x)

        # Final output is special.
        x = Conv2DTranspose(self.output_shape[2],
                            self.hparams.kernels[-1],
                            strides=self.hparams.strides[-1],
                            activation=self.hparams.output_activation,
                            padding='same')(x)

        conv2d_layers.append(x)

        del x
        gc.collect()

        return conv2d_layers

    def _create_graph(self):
        input_ = Input(shape=(self.hparams.latent_dim,), name='decoder_input')
        affine_layers = self._affine_layers(input_)
        reshaped = Reshape(self.enc_conv_shape)(affine_layers[-1])
        conv_layers = self._conv_layers(reshaped)[-1]
        generator = Model(input_, conv_layers, name='generator')
        return generator
