from torch import nn
from molecules.ml.unsupervised.vae.utils import (same_padding,
                                                 conv_output_shape,
                                                 get_activation)

class ResidualConv1d(nn.Module):
    def __init__(self, input_shape, filters, kernel_size,
                 activation='ReLU', shrink=False, kfac=2):
        super(ResidualConv1d, self).__init__()

        self.input_shape = input_shape
        self.output_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.shrink = shrink
        self.kfac = kfac

        print('res ctor input_shape: ', self.input_shape)
        print('filters: ', self.filters)

        self.residual = self._residual_layers()
        padding = same_padding(self.input_shape[1], 1, 1)
        self.conv = nn.Conv1d(self.input_shape[0],
                              self.filters,
                              kernel_size=1,
                              stride=1,
                              padding=padding)

        self.temp = conv_output_shape(input_dim=self.input_shape[1],
                                kernel_size=1,
                                stride=1,
                                padding=padding,
                                num_filters=self.filters,
                                dim=1)

        print('res ctor shape: ', self.temp)

        self.activation_fnc = get_activation(self.activation)

        if self.shrink:
            self.shrink_layer, self.output_shape = self._shrink_layer()

    def forward(self, x):

        # TODO: should we use activation here?
        res = self.residual(x)
        print('res shape: ', res.size())
        conv = self.conv(x)
        print('conv shape: ', conv.size())
        x = self.activation_fnc(conv + res)

        if self.shrink:
            x = self.shrink_layer(x)

        print('end res: ', x.size())

        return x

    def _residual_layers(self):
        # TODO: check out SyncBatchNorm
        # TODO: could add activation for bottleneck layers
        # TODO: prefer wide layers and shallower autoencoder
        #       see https://arxiv.org/pdf/1605.07146.pdf

        assert len(self.input_shape) == 2

        layers = []

        # First add bottleneck layer

        # bottleneck_padding = same_padding(self.input_shape[1], kernel_size=1, stride=1)

        # layers.append(nn.Conv1d(in_channels=self.input_shape[0],
        #                         out_channels=self.filters,
        #                         kernel_size=1,
        #                         stride=1,
        #                         padding=bottleneck_padding))

        # shape = (self.filters, self.input_shape[1])

        shape = self.input_shape

        # Now add residual layers

        layers.append(nn.BatchNorm1d(num_features=shape[0]))

        layers.append(get_activation(self.activation))

        padding = same_padding(shape[1], self.kernel_size, stride=1)

        layers.append(nn.Conv1d(in_channels=shape[0],
                                out_channels=self.filters,
                                kernel_size=self.kernel_size,
                                stride=1,
                                padding=padding))

        shape = conv_output_shape(input_dim=shape[1],
                                  kernel_size=self.kernel_size,
                                  stride=1,
                                  padding=padding,
                                  num_filters=self.filters,
                                  dim=1)

        # Project back up (undo bottleneck)
        layers.append(nn.BatchNorm1d(num_features=shape[0]))

        layers.append(get_activation(self.activation))

        # TODO: in_channels=shape[0]
        # TODO: this does not appear to be in keras code (it uses self.kernel_size)
        # TODO: address above TODOs and test using self.kernel_size
        # layers.append(nn.Conv1d(in_channels=shape[0],
        #                         out_channels=self.input_shape[0],
        #                         kernel_size=1,
        #                         stride=1,
        #                         padding=bottleneck_padding))

        padding = same_padding(shape[1], self.kernel_size, stride=1)

        layers.append(nn.Conv1d(in_channels=shape[0],
                                out_channels=self.filters,
                                kernel_size=self.kernel_size,
                                stride=1,
                                padding=padding))


        return nn.Sequential(*layers)

    def _shrink_layer(self):

        # TODO: if this layer is added, there are 2 conv layers back to back
        #       without activation. The input to this layer is x + residual.
        #       Consider if it should be wrapped activation(x + residual).
        #       See forward function.

        padding = same_padding(self.temp[1], self.kfac, self.kfac)

        conv = nn.Conv1d(in_channels=self.temp[0],
                         out_channels=self.filters,
                         kernel_size=self.kfac,
                         stride=self.kfac,
                         padding=padding)

        #act = get_activation(self.activation)

        shape = conv_output_shape(input_dim=self.temp[1],
                                  kernel_size=self.kfac,
                                  stride=self.kfac,
                                  padding=padding,
                                  num_filters=self.filters,
                                  dim=1)

        # print('\nResidualConv1d::_shrink_layer\n',
        #       f'\t input_shape: {self.input_shape}\n',
        #       f'\t out_shape: {shape}\n',
        #       f'\t filters: {self.filters}\n',
        #       f'\t kernel_size: {self.kfac}\n',
        #       f'\t stride: {self.kfac}\n',
        #       f'\t padding: {padding}\n\n')

        return nn.Sequential(conv), shape
