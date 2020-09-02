import os
import time
import torch
import numpy as np
from .callback import Callback
import wandb

def get_plot_object(name, array, epoch, sample):
    result = {
        name: wandb.Object3D(
            {
                "type": "lidar/beta",
                "points": array,
                "boxes": np.array(
                    [
                        {
                            "corners": [
                                [0,0,0],
                                [0,1,0],
                                [0,0,1],
                                [1,0,0],
                                [1,1,0],
                                [0,1,1],
                                [1,0,1],
                                [1,1,1]
                            ],
                            "label": "Box",
                            "color": [123,321,111]
                        }
                    ]
                ),
                "vectors": np.array(
                    [
                        {
                            "start": [0,0,0],
                            "end": [0.1,0.2,0.5]
                        }
                    ]
                )
            }
        ),
        "epoch": epoch,
        "sample": sample
    }

class PointCloud3dCallback(Callback):

    """
    Saves pointclouds to file or wandb
    """
    def __init__(self, out_dir,
                 sample_interval = 20,
                 wandb_config = None):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output plots.

        sample_interval : int
            Plots every sample_interval'th point in the data set
        """
        super(PointCloud3dCallback).__init__()

        os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.wandb_config = wandb_config

    def on_epoch_end(self, epoch, logs):
        if self.wandb_config is not None:
            
            # prepare plot data
            inp = np.transpose(logs["input_samples"], axes = (0,2,1))
            tar = np.transpose(logs["reconstructed_samples"], axes = (0,2,1))
            
            # plot inputs
            for idx in range(inp.shape[0]):
                if idx % self.sample_interval == 0:
                    wandb.log(get_plot_object("point_cloud_in", inp[idx, ...], epoch, idx), step = logs["global_step"])

            # plot target
            for idx in range(tar.shape[0]):
                if idx % self.sample_interval == 0:
                    wandb.log(get_plot_object("point_cloud_out", tar[idx, ...], epoch, idx), step = logs["global_step"])
                    #wandb.log({"point_cloud_out": wandb.Object3D(tar[idx, ...]),
                    #           "epoch": epoch,
                    #           "sample": idx})
