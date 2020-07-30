import torch
import torch.nn as nn
from collections import OrderedDict
from hyperparams import AAE3dHyperparams
from molecules.ml.unsupervised.vae.utils import init_weights

class Generator(nn.Module):
    def __init__(self, num_points, hparams, device):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_generator_bias
        self.relu_slope = hparams.generator_relu_slope

        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # first layer
        layers = OrderedDict([('linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = hparams.generator_filters[0],
                                                    bias=self.use_bias)),
                              ('relu1', self.activation)])
        
        # intermediate layers
        for idx in range(1, len(hparams.generator_filters[1:])):
            layers.append(('linear{}'.format(idx+1), nn.Linear(in_features = hparams.generator_filters[idx - 1],
                                                               out_features = hparams.generator_filters[idx],
                                                               bias=self.use_bias)))
            layers.append(('relu{}'.format(idx+1), self.activation))

        # last layer
        layers.append(('linear{}'.format(idx+1), nn.Linear(in_features = hparams.generator_filters[-1],
                                                           out_features = self.num_points * 3 * self.num_features,
                                                           bias=self.use_bias)))

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights()
        
        #self.model = nn.Sequential(
        #    nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(in_features=1024, out_features=2048 * 3, bias=self.use_bias),
        #)
    def init_weights(self):
        self.model.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, input):
        output = self.model(input.squeeze())
        output = output.view(-1, 3*self.num_features, self.num_points)
        return output


class Discriminator(nn.Module):
    def __init__(self, hparams, device):
        super().__init__()

        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_discriminator_bias
        self.relu_slope = hparams.discriminator_relu_slope

        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # first layer
        layers = OrderedDict([('linear1', nn.Linear(in_features = self.z_size,
                                                    out_features = hparams.discriminator_filters[0],
                                                    bias = self.use_bias)),
                              ('relu1', self.activation)])

        # intermediate layers
        for idx in range(1, len(hparams.discriminator_filters[1:])):
            layers.append(('linear{}'.format(idx+1), nn.Linear(in_features = hparams.discriminator_filters[idx - 1],
                                                               out_features = hparams.discriminator_filters[idx],
                                                               bias = self.use_bias)))
            layers.append(('relu{}'.format(idx+1), self.activation))

        # final layer
        layers.append(('linear{}'.format(idx+1), nn.Linear(in_features = hparams.discriminator_filters[-1],
                                                           out_features = 1,
                                                           bias = self.use_bias)))

        # construct model
        self.model = nn.Sequential(layers)
        
        # init weights
        self.init_weights()
            
        #self.model = nn.Sequential(
        #
        #    nn.Linear(self.z_size, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 512, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(512, 128, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(128, 64, bias=True),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Linear(64, 1, bias=True)
        #)
        
    def init_weights(self):
        self.model.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        logit = self.model(x)
        return logit


class Encoder(nn.Module):
    def __init__(self, num_points, hparams, device):
        super().__init__()

        # copy some parameters
        self.num_points = num_points
        self.num_features = num_features
        self.z_size = hparams.latent_dim
        self.use_bias = hparams.use_encoder_bias
        self.relu_slope = hparams.encoder_relu_slope
        
        # select activation
        if self.relu_slope > 0.:
            self.activation = nn.LeakyReLU(negative_slope = self.relu_slope,
                                           inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        # first layer
        layers = OrderedDict([('conv1', nn.Conv1d(in_channels = 3 * self.num_features,
                                                  out_channels = hparams.encoder_filters[0],
                                                  kernel_size = 1,
                                                  bias = self.use_bias)),
                              ('relu1', self.activation)])

        # intermediate layers
        for idx in range(1, len(hparams.encoder_filters[1:-1])):
            layers.append(('conv{}'.format(idx+1), nn.Conv1d(in_channels = hparams.encoder_filters[idx - 1],
                                                             out_channels = hparams.encoder_filters[idx],
                                                             bias = self.use_bias)))
            layers.append(('relu{}'.format(idx+1), self.activation))

        # final layer
        layers.append(('linear{}'.format(idx+1), nn.Conv1d(in_channels = hparams.encoder_filters[-2],
                                                           out_channels = hparams.encoder_filters[-1],
                                                           bias = self.use_bias)))

        # construct model
        self.conv = nn.Sequential(layers)

        self.fc = nn.Sequential(
            nn.Linear(hparams.encoder_filters[-1],
                      hparams.encoder_filters[-2],
                      bias=True),
            self.activation
        )

        self.mu_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        self.std_layer = nn.Linear(hparams.encoder_filters[-2], self.z_size, bias=True)
        
        # init model
        self.init_weights()
        
        #self.conv = nn.Sequential(
        #    nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1,
        #              bias=self.use_bias),
        #    nn.ReLU(inplace=True),
        #
        #    nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1,
        #              bias=self.use_bias),
        #)

        #self.fc = nn.Sequential(
        #    nn.Linear(512, 256, bias=True),
        #    nn.ReLU(inplace=True)
        #)

        #self.mu_layer = nn.Linear(256, self.z_size, bias=True)
        #self.std_layer = nn.Linear(256, self.z_size, bias=True)
    
    def init_weights(self):
        self.conv.apply(init_weights)
        self.fc.apply(init_weights)
        self.mu_layer.apply(init_weights)
        self.std_layer.apply(init_weights)
        
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.conv(x)
        output2 = output.max(dim = 2)[0]
        logit = self.fc(output2)
        mu = self.mu_layer(logit)
        logvar = self.std_layer(logit)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class AAE3dModel(nn.Module):
    def __init__(self, num_points, hparams, device):
        super(AAE3dModel, self).__init__()

        # instantiate encoder, generator and discriminator
        self.encoder = Encoder(num_points, hparams, device)
        self.generator = Generator(num_points, hparams, device)
        self.discriminator = Discriminator(hparams, device)

        # map to device
        self.encoder = self.encoder.to(device)
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

    def forward(self, x):
        x, mu, logvar = self.encoder(x)
        x = self.generator(x)
        return x, mu, logvar
        
    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def generate(self, z):
        x = self.generator(z)
        return x
        
    def discriminate(self, z)
        p = self.discriminator(z)
        return p
    
    def save_weights(self, enc_path, gen_path, disc_path):
        self.encoder.save_weights(enc_path)
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.generator.load_weights(gen_path)
        self.discriminator.load_weights(disc_path)
        
class AAE3d(object):
    """
    Provides high level interface for training, testing and saving VAE
    models. Takes arbitrary encoder/decoder models specified by the choice
    of hyperparameters. Assumes the shape of the data is square.

    Attributes
    ----------
    model : torch.nn.Module (VAEModel)
        Underlying Pytorch model with encoder/decoder attributes.

    optimizer : torch.optim.Optimizer
        Pytorch optimizer used to train model.

    loss_func : function
        Loss function used to train model.

    Methods
    -------
    train(train_loader, valid_loader, epochs=1, checkpoint='', callbacks=[])
        Train model

    encode(x)
        Embed data into the latent space.

    decode(embedding)
        Generate matrices from embeddings.

    save_weights(enc_path, dec_path)
        Save encoder/decoder weights.

    load_weights(enc_path, dec_path)
        Load saved encoder/decoder weights.
    """
