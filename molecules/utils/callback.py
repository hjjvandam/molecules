import os
import time
import torch

class Callback:
    def __init__(self): pass
    def on_train_begin(self, logs): pass
    def on_train_end(self, logs): pass
    def on_epoch_begin(self, epoch, logs): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_batch_begin(self, batch, epoch, logs): pass
    def on_batch_end(self, batch, epoch, logs): pass


# TODO: need way to share SummaryWriter among multiple callbacks for a model
#       could make writer global variable

class LossCallback(Callback):
    def on_train_begin(self, logs):
        #from torch.utils.tensorboard import SummaryWriter
        #self.writer = SummaryWriter()

        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch, logs):

        # self.writer.add_scalar('epoch training loss',
        #                        logs['train_loss'],
        #                        logs['global_step'])
        # self.writer.add_scalar('epoch validation loss',
        #                        logs['valid_loss'],
        #                        logs['global_step'])

        self.train_losses.append(logs['train_loss'])
        self.valid_losses.append(logs['valid_loss'])

    def save(self, path):
        """
        Save train and validation loss from the end of each epoch.

        Parameters
        ----------
        path: str
            Path to save train and validation loss history
        """
        torch.save({'loss': self.train_losses, 'valid': self.valid_losses}, path)

class CheckpointCallback(Callback):
    def __init__(self, interval=0,
                 directory=os.path.join('.', 'checkpoints')):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training and
        record model weight history.

        Parameters
        ----------
        directory : str
            Directory to store checkpoint files.
            Files are named 'epoch-{e}-%Y%m%d-%H%M%S.pt'

        interval : int
            Checkpoints model every interval batches, default is once per epoch.
        """

        if interval < 0:
            raise ValueError('Checkpoint interval must be non-negative')

        os.makedirs(directory, exist_ok=True)

        self.interval = interval
        self.directory = directory

    def on_batch_end(self, batch, epoch, logs):
        if self.interval and batch % self.interval == 0:
            self._save(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        if not self.interval:
            self._save(epoch, logs)

    def _save(self, epoch, logs):
        """Saves optimizer state and encoder/decoder weights."""

        checkpoint = {
            'encoder_state_dict': logs['model'].encoder.state_dict(),
            'decoder_state_dict': logs['model'].decoder.state_dict(),
            'optimizer_state_dict': logs['optimizer'].state_dict(),
            'epoch': epoch
            }

        time_stamp = time.strftime(f'epoch-{epoch}-%Y%m%d-%H%M%S.pt')
        path = os.path.join(self.directory, time_stamp)
        torch.save(checkpoint, path)


class EmbeddingCallback(Callback):
    """
    Saves embeddings of random samples.

    Parameters
    ----------
    data : torch.Tensor
        Dataset from which to sample for embeddings.

    """
    def __init__(self, data):
        self.data = data

    def on_train_begin(self, logs):
        self.embeddings = []
        self.data_index = []

    def on_epoch_end(self, epoch, logs):
        # TODO: may need to change the torch device
        idx = torch.randint(len(self.data), (1,))
        embedding = logs['model'].encode(self.data[idx])
        self.data_index.append(idx)
        self.embeddings.append(embedding)

    def save(self, path):
        """
        Save embeddings and index of associated data point.

        Parameters
        ----------
        path: str
            Path to save embeddings and indices

        """

        torch.save({'embeddings': self.embeddings, 'indices': self.data_index}, path)
