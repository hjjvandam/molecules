import os
import time
import torch
from .callback import Callback

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

