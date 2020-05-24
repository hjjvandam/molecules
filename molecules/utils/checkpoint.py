import os
import time
import torch

class Checkpoint:
    def __init__(self, interval=0,
                 directory=os.path.join('.', 'checkpoints')):
        """
        Parameters
        ----------
        directory : str
            Directory to store checkpoint files.
            Files are named %Y%m%d-%H%M%S.pt

        interval : int
            Checkpoints model every interval batches,
            default is once per epoch.
        """

        if interval < 0:
            raise ValueError('Checkpoint interval must be non-negative')

        os.makedirs(directory, exist_ok=True)

        self.directory = directory
        self.interval = interval

    def __bool__(self):
        return True

    def per_epoch(self):
        """Return true if saving once per epoch"""
        return not self.interval

    def per_batch(self, batch_idx):
        """Return true if saving by batch_idx"""
        return self.interval and (batch_idx % self.interval) == 0

    def save(self, checkpoint):
        """
        Saves pt checkpoint file with the current time stamp.

        Parameters
        ----------
        checkpoint : dict
            dictionary of objects to save
        """

        time_stamp = time.strftime('%Y%m%d-%H%M%S.pt')
        path = os.path.join(self.directory, time_stamp)
        torch.save(checkpoint, path)

    def load(self, path):
        # TODO: implement
        pass
