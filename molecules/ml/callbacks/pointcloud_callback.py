import os
import time
import torch
import numpy as np
from .callback import Callback
import wandb

class PointCloud3dCallback(Callback):

    """
    Saves pointclouds to file or wandb
    """
    def __init__(self, out_dir,
                 sample_interval = 20,
                 writer = None,
                 wandb_config = None):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output plots.

        sample_interval : int
            Plots every sample_interval'th point in the data set

        writer : torch.utils.tensorboard.SummaryWriter
        """
        super(PointCloud3dCallback).__init__()

        os.makedirs(out_dir, exist_ok=True)

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.writer = writer
        self.wandb_config = wandb_config

    def on_epoch_end(self, epoch, logs):
        if wandb_config is not None:
            inp = loss["input_samples"]
            for idx in range(inp.shape[0]):
                print(inp.shape)
                if idx%self.sample_interval == 0:
                    wandb.log({"point_cloud_in": wandb.Object3D(inp[idx, ...]),
                               "epoch": epoch,
                               "sample": idx})

            tar = loss["reconstructed_samples"]
            for idx in range(tar.shape[0]):
                if idx%self.sample_interval == 0:
                    wandb.log({"point_cloud_out": wandb.Object3D(tar[idx, ...]),
                               "epoch": epoch,
                               "sample": idx})
