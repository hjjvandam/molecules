import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
from .resnet import ResnetVAEHyperparams
from .symmetric import SymmetricVAEHyperparams
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer

__all__ = ['VAE']

Device = namedtuple('Device', ['encoder', 'decoder'])

class VAEModel(nn.Module):
    def __init__(self, input_shape, hparams, device):
        super(VAEModel, self).__init__()

        # Select encoder/decoder models by the type of the hparams
        if isinstance(hparams, SymmetricVAEHyperparams):
            from .symmetric import SymmetricEncoderConv2d, SymmetricDecoderConv2d
            self.encoder = SymmetricEncoderConv2d(input_shape, hparams)
            self.decoder = SymmetricDecoderConv2d(input_shape, hparams, self.encoder.shapes)

        elif isinstance(hparams, ResnetVAEHyperparams):
            from .resnet import ResnetEncoder, ResnetDecoder
            self.encoder = ResnetEncoder(input_shape, hparams)
            self.decoder = ResnetDecoder(input_shape, hparams)

        else:
            raise TypeError(f'Invalid hparams type: {type(hparams)}.')

        self.encoder.to(device.encoder)
        self.decoder.to(device.decoder)

        self.device = device

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # x should be placed on encoder gpu in the dataset class
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar).to(self.device.decoder)
        x = self.decoder(z).to(self.device.encoder)
        # TODO: see if we can remove this to speed things up
        #       or find an inplace way. Only necessary for bad
        #       hyperparam config such as optimizer learning rate
        #       being large.
        #x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x, z, mu, logvar

    def encode(self, x):
        # mu layer
        return self.encoder.encode(x)

    def decode(self, embedding):
        return self.decoder.decode(embedding)

    def save_weights(self, enc_path, dec_path):
        self.encoder.save_weights(enc_path)
        self.decoder.save_weights(dec_path)

    def load_weights(self, enc_path, dec_path):
        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)


def vae_loss(recon_x, x, mu, logvar):
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


# TODO: set weight initialization hparams
class VAE:
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

    def __init__(self, input_shape,
                 hparams=SymmetricVAEHyperparams(),
                 optimizer_hparams=OptimizerHyperparams(),
                 loss=None,
                 gpu=None,
                 verbose=True):
        """
        Parameters
        ----------
        input_shape : tuple
            shape of incomming data.
            Note: For use with SymmetricVAE use (1, num_residues, num_residues)
                  For use with ResnetVAE use (num_residues, num_residues)

        hparams : molecules.ml.hyperparams.Hyperparams
            Defines the model architecture hyperparameters. Currently implemented
            are SymmetricVAEHyperparams and ResnetVAEHyperparams.

        optimizer_hparams : molecules.ml.hyperparams.OptimizerHyperparams
            Defines the optimizer type and corresponding hyperparameters.

        loss: : function, optional
            Defines an optional loss function with inputs (recon_x, x, mu, logvar)
            and ouput torch loss.

        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively.

        verbose : bool
            True prints training and validation loss to stdout.
        """

        hparams.validate()
        optimizer_hparams.validate()

        self.verbose = verbose

        # Tuple of encoder, decoder device
        self.device = Device(*self._configure_device(gpu))

        self.model = VAEModel(input_shape, hparams, self.device)

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer = get_optimizer(self.model.parameters(), optimizer_hparams)

        self.loss_fnc = vae_loss if loss is None else loss

    def _configure_device(self, gpu):
        """
        Configures GPU/CPU device for training VAE. Allows encoder
        and decoder to be trained on seperate devices.

        Parameters
        ----------
        gpu : int, tuple, or None
            Encoder and decoder will train on ...
            If None, cuda GPU device if it is available, otherwise CPU.
            If int, the specified GPU.
            If tuple, the first and second GPUs respectively or None
            option if tuple contains None.

        Returns
        -------
        2-tuple of encoder_device, decoder_device
        """

        if gpu is None or (isinstance(gpu, tuple) and None in gpu):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return device, device
        if not torch.cuda.is_available():
            raise ValueError('Specified GPU training but CUDA is not available.')
        if isinstance(gpu, int):
            device = torch.device(f'cuda:{gpu}')
            return device, device
        if isinstance(gpu, tuple) and len(gpu) == 2:
            return torch.device(f'cuda:{gpu[0]}'), torch.device(f'cuda:{gpu[1]}')
        raise ValueError('Specified GPU device is invalid. Should be int, 2-tuple or None.')

    def __repr__(self):
        return str(self.model)

    def train(self, train_loader, valid_loader, epochs=1, checkpoint='',
              callbacks=[]):
        """
        Train model

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        epochs : int
            Number of epochs to train for

        checkpoint : str
            Path to checkpoint file to load and resume training
            from the epoch when the checkpoint was saved.

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.
        """

        if callbacks:
            logs = {'model': self.model, 'optimizer': self.optimizer}
        else:
            logs = {}

        start_epoch = 1

        if checkpoint:
            start_epoch += self._load_checkpoint(checkpoint)

        for callback in callbacks:
            callback.on_train_begin(logs)

        for epoch in range(start_epoch, epochs + 1):

            for callback in callbacks:
                callback.on_epoch_begin(epoch, logs)

            self._train(train_loader, epoch, callbacks, logs)
            self._validate(valid_loader, epoch, callbacks, logs)

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        for callback in callbacks:
            callback.on_train_end(logs)

    def _train(self, train_loader, epoch, callbacks, logs):
        """
        Train for 1 epoch

        Parameters
        ----------
        train_loader : torch.utils.data.dataloader.DataLoader
            Contains training data

        epoch : int
            Current epoch of training

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """

        self.model.train()
        train_loss = 0.
        for batch_idx, token in enumerate(train_loader):

            data, rmsd = token
            data = data.to(self.device[0])
            
            if self.verbose:
                start = time.time()

            if callbacks:
                pass # TODO: add more to logs

            for callback in callbacks:
                callback.on_batch_begin(batch_idx, epoch, logs)

            # TODO: Consider passing device argument into dataset class instead
            # data = data.to(self.device.encoder)
            self.optimizer.zero_grad()
            recon_batch, codes, mu, logvar = self.model(data)
            loss = self.loss_fnc(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item() / len(data)
            self.optimizer.step()

            if callbacks:
                logs['train_loss'] = loss.item() / len(data)
                logs['global_step'] = (epoch - 1) * len(train_loader) + batch_idx

            if self.verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}'.format(
                      epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                      100. * (batch_idx + 1) / len(train_loader),
                      loss.item() / len(data), time.time() - start))

            for callback in callbacks:
                callback.on_batch_end(batch_idx, epoch, logs)

        train_loss_ave = train_loss / float(batch_idx + 1)

        if callbacks:
            logs['train_loss_average'] = train_loss_ave

        if self.verbose:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_ave))

    def _validate(self, valid_loader, epoch, callbacks, logs):
        """
        Test model on validation set.

        Parameters
        ----------
        valid_loader : torch.utils.data.dataloader.DataLoader
            Contains validation data

        callbacks : list
            Contains molecules.utils.callback.Callback objects
            which are called during training.

        logs : dict
            Filled with data for callbacks
        """
        self.model.eval()
        valid_loss = 0
        for callback in callbacks:
            callback.on_validation_begin(epoch, logs)

        with torch.no_grad():
            for batch_idx, token in enumerate(valid_loader):
                data, rmsd = token
                data = data.to(self.device[0])
                
                # data = data.to(self.device)
                recon_batch, codes, mu, logvar = self.model(data)
                valid_loss += self.loss_fnc(recon_batch, data, mu, logvar).item() / len(data)

                for callback in callbacks:
                    callback.on_validation_batch_end(logs,
                                                     input = data.detach(),
                                                     rmsd = rmsd.detach(),
                                                     mu = mu.detach())
                
        valid_loss /= float(batch_idx + 1)

        for callback in callbacks:
            callback.on_validation_end(epoch, logs)
        
        if self.verbose:
            print('====> Validation loss: {:.4f}'.format(valid_loss))

    def _load_checkpoint(self, path):
        """
        Loads checkpoint file containing optimizer state and
        encoder/decoder weights.

        Parameters
        ----------
        path : str
            Path to checkpoint file

        Returns
        -------
        Epoch of training corresponding to the saved checkpoint.
        """

        cp = torch.load(path)
        self.model.encoder.load_state_dict(cp['encoder_state_dict'])
        self.model.decoder.load_state_dict(cp['decoder_state_dict'])
        self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        return cp['epoch']

    def encode(self, x):
        """
        Embed data into the latent space.

        Parameters
        ----------
        x : torch.Tensor
            Data to encode, could be a batch of data with dimension
            (batch_size, input_shape)

        Returns
        -------
        torch.Tensor of embeddings of shape (batch-size, latent_dim)

        """

        return self.model.encode(x)

    def decode(self, embedding):
        """
        Generate matrices from embeddings.

        Parameters
        ----------
        embedding : torch.Tensor
            Embedding data, could be a batch of data with dimension
            (batch-size, latent_dim)

        Returns
        -------
        torch.Tensor of generated matrices of shape (batch-size, input_shape)
        """

        return self.model.decode(embedding)

    def save_weights(self, enc_path, dec_path):
        """
        Save encoder/decoder weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights.

        dec_path : str
            Path to save the decoder weights.
        """

        self.model.save_weights(enc_path, dec_path)

    def load_weights(self, enc_path, dec_path):
        """
        Load saved encoder/decoder weights.

        Parameters
        ----------
        enc_path : str
            Path to save the encoder weights.

        dec_path : str
            Path to save the decoder weights.
        """

        self.model.load_weights(enc_path, dec_path)
