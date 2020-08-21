import os
import time
import torch
from .callback import Callback
import torch.distributed as dist

class CheckpointCallback(Callback):
    def __init__(self, interval=0,
                 out_dir=os.path.join('.', 'checkpoints'),
                 mpi_comm = None):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training and
        record model weight history.

        Parameters
        ----------
        out_dir : str
            Directory to store checkpoint files.
            Files are named 'epoch-{e}-%Y%m%d-%H%M%S.pt'

        interval : int
            Checkpoints model every interval batches, default is once per epoch.
        """

        if interval < 0:
            raise ValueError('Checkpoint interval must be non-negative')

        self.comm = mpi_comm
        self.is_eval_node = True
        if (self.comm is not None) and (self.comm.Get_rank() != 0):
            self.is_eval_node = False

        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

        self.interval = interval
        self.out_dir = out_dir

    def on_batch_end(self, batch, epoch, logs):
        if self.is_eval_node and self.interval and (batch % self.interval == 0):
            self._save(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        if self.is_eval_node and not self.interval:
            self._save(epoch, logs)

    def _save(self, epoch, logs):
        """Saves optimizer state and encoder/decoder weights."""

        # create new dictionary
        checkpoint = {
            'epoch': epoch
            }

        # optimizer
        if "optimizer" in logs:
            checkpoint['optimizer_state_dict'] = logs['optimizer'].state_dict()

        if "optimizer_d" in logs:
            checkpoint['optimizer_d_state_dict'] = logs['optimizer_d'].state_dict()

        if "optimizer_eg" in logs:
            checkpoint['optimizer_eg_state_dict'] = logs['optimizer_eg'].state_dict()

        # model parameter
        if hasattr(logs['model'], 'encoder'):
            checkpoint['encoder_state_dict'] = logs['model'].encoder.state_dict()

        if hasattr(logs['model'], 'decoder'):
            checkpoint['decoder_state_dict'] = logs['model'].decoder.state_dict()

        if hasattr(logs['model'], 'generator'):
            checkpoint['generator_state_dict'] = logs['model'].generator.state_dict()

        if hasattr(logs['model'], 'discriminator'):
                checkpoint['discriminator_state_dict'] = logs['model'].discriminator.state_dict()

        time_stamp = time.strftime(f'epoch-{epoch}-%Y%m%d-%H%M%S.pt')
        path = os.path.join(self.out_dir, time_stamp)
        torch.save(checkpoint, path)
