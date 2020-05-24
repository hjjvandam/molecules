import torch
from torch import nn, optim
from torch.nn import functional as F
from math import isclose
from molecules.ml.hyperparams import OptimizerHyperparams, get_optimizer
from molecules.ml.unsupervised.vae.symmetric import SymmetricVAEHyperparams
from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams

class VAEModel(nn.Module):
    def __init__(self, input_shape, hparams):
        super(VAEModel, self).__init__()

        if isinstance(hparams, SymmetricVAEHyperparams):
            from molecules.ml.unsupervised.vae.symmetric import (SymmetricEncoderConv2d,
                                                                 SymmetricDecoderConv2d)
            self.encoder = SymmetricEncoderConv2d(input_shape, hparams)
            self.decoder = SymmetricDecoderConv2d(input_shape, hparams, self.encoder.encoder_dim)

        elif isinstance(hparams, ResnetVAEHyperparams):
            from molecules.ml.unsupervised.vae.resnet import ResnetEncoder, ResnetDecoder
            self.encoder = ResnetEncoder(input_shape, hparams)
            self.decoder = ResnetDecoder(input_shape, hparams)

        else:
            raise TypeError(f'Invalid hparams type: {type(hparams)}.')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        x = self.reparameterize(mu, logvar)
        x = self.decoder(x)
        # TODO: see if we can remove this to speed things up
        #       or find an inplace way. Only necessary for bad
        #       hyperparam config such as optimizer learning rate
        #       being large.
        #x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x, mu, logvar

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


class VAE:
    # TODO: set weight initialization hparams
    def __init__(self, input_shape,
                 hparams=SymmetricVAEHyperparams(),
                 optimizer_hparams=OptimizerHyperparams(),
                 checkpoint=None,
                 loss=None,
                 cuda=True):

        hparams.validate()
        optimizer_hparams.validate()

        self.input_shape = input_shape
        self.checkpoint = checkpoint

        # TODO: consider passing in device (this will allow the ability to set the train/test
        #       data to cuda as well, since device will be a variable in the user space)
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

        self.model = VAEModel(input_shape, hparams).to(self.device)

        # TODO: consider making optimizer_hparams a member variable
        # RMSprop with lr=0.001, alpha=0.9, epsilon=1e-08, decay=0.0
        self.optimizer = get_optimizer(self.model, optimizer_hparams)

        self.loss = vae_loss if loss is None else loss

    def __repr__(self):
        return str(self.model)

    def train(self, train_loader, test_loader, epochs=1,
              checkpoint_path=None):

        if checkpoint_path is not None:
            # TODO: load checkpoint set epochs
            pass


        for epoch in range(1, epochs + 1):
            self._train(train_loader, epoch)
            self._test(test_loader)

    def _train(self, train_loader, epoch, log_interval=2):
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
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()


            # TODO: Handle logging/checkpoint
            #       https://pytorch.org/tutorials/beginner/saving_loading_models.html
            #       should also save optimizer state
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))

            if self.checkpoint and self.checkpoint.per_batch(batch_idx):
                self._save_checkpoint(epoch, train_loss)

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))

        if self.checkpoint and self.checkpoint.per_epoch():
            self._save_checkpoint(epoch, train_loss)


    def _test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def _save_checkpoint(self, epoch, loss):
        """
        Saves optimizer state, encoder weights, decoder weights.
        """

        self.checkpoint.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
            })

    def _load_checkpoint(self, path):
        # TODO: implement resume training from checkpoint
        pass

    def encode(self, x):
        """
        Effects
        -------
        Embed a datapoint into the latent space.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray of embeddings.

        """
        return self.model.encode(x)

    def decode(self, embedding):
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
        return self.model.decode(embedding)

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
        self.model.save_weights(enc_path, dec_path)

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
        self.model.load_weights(enc_path, dec_path)
