import json
from .callback import Callback

class LossCallback(Callback):
    def __init__(self, path, writer=None):
        """
        Parameters
        ----------
        path : str
            path to save loss history to

        writer : torch.utils.tensorboard.SummaryWriter
        """
        self.writer = writer
        self.path = path

    def on_train_begin(self, logs):
        self.epochs = []
        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch, logs):

        if self.writer is not None:
            self.writer.add_scalar('epoch training loss',
                                   logs['train_loss'],
                                   logs['global_step'])
            self.writer.add_scalar('epoch validation loss',
                                   logs['valid_loss'],
                                   logs['global_step'])

        self.epochs.append(epoch)
        self.train_losses.append(logs['train_loss'])
        self.valid_losses.append(logs['valid_loss'])

        self.save(self.path)

    def save(self, path):
        """
        Save train and validation loss from the end of each epoch.

        Parameters
        ----------
        path: str
            Path to save train and validation loss history
        """

        # Happens when loading from a checkpoint
        if self.epochs[0] != 1:
            with open(path) as f:
                data = json.load(f)
                if data:
                    # Prepend data from checkpointed model to the start of the
                    # current logs. This avoids needing to load the data every
                    # time the logs are saved.
                    data['epochs'].extend(self.epochs)
                    data['train_loss'].extend(self.train_losses)
                    data['valid_loss'].extend(self.valid_losses)
                    self.epochs = data['epochs']
                    self.train_losses = data['train_loss']
                    self.valid_losses = data['valid_loss']

        # Write history to disk
        with open(path, 'w') as f:
            json.dump({'train_loss': self.train_losses,
                       'valid_loss': self.valid_losses,
                       'epochs': self.epochs}, f)
