import os
import time
import torch

class Checkpoint:
    def __init__(self, interval=0,
                 directory=os.path.join('.', 'checkpoints')):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training or
        record model weight history.

        Note: This interface is not specifically designed to store
              loss or validation performance.

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

    def __bool__(self):
        """Check for existence against None type."""
        return True

    def per_epoch(self):
        """Return true if saving once per epoch."""
        return not self.interval

    def per_batch(self, batch_idx):
        """Return true if saving by batch_idx."""
        return self.interval and batch_idx % self.interval == 0

    def save(self, checkpoint, epoch):
        """
        Saves pt checkpoint file with the current time stamp
        and epoch of training.

        Parameters
        ----------
        checkpoint : dict
            Dictionary of objects to save

        epoch : int
            Current epoch of training
        """

        time_stamp = time.strftime(f'epoch-{epoch}-%Y%m%d-%H%M%S.pt')
        path = os.path.join(self.directory, time_stamp)
        torch.save(checkpoint, path)

    def load(self, path):
        """
        Parameters
        ----------
        path : str
            Path to checkpoint file

        Returns
        -------
        Checkpoint dictionary
        """

        return torch.load(path)
