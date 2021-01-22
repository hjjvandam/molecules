import os
import time
import torch
import torch.distributed as dist
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from .callback import Callback
from PIL import Image
import wandb


class LatspaceStatisticsCallback(Callback):
    """
    Saves AE projections for mu and std of random samples.
    """
    def __init__(self, out_dir,
                 sample_interval = 20,
                 wandb_config = None,
                 mpi_comm = None):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output plots.
        sample_interval : int
            Plots every sample_interval'th point in the data set
        wandb_config : wandb configuration file
        """
        super().__init__(mpi_comm=mpi_comm)
            
        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.wandb_config = wandb_config
        
    def on_validation_begin(self, epoch, logs):
        self.sample_counter = 0
        self.mu = []
        self.std = []
        
        
    def on_validation_batch_end(self, logs, mu = None, logvar = None, **kwargs):
        if self.sample_interval == 0:
            return

        if (mu is None) or (logvar is None):
            pass
        
        # decide what to store
        for idx in range(0, len(mu)):
            if (self.sample_counter + idx) % self.sample_interval == 0:
                # use a singleton slice to keep dimensions intact
                self.mu.append(mu[idx:idx+1].detach().cpu().numpy())
                self.std.append(torch.exp(0.5*logvar[idx:idx+1].detach()).cpu().numpy())

        # increase sample counter
        self.sample_counter += len(mu)

        
    def on_validation_end(self, epoch, logs):
        # if the sample interval was too large, we should warn here and return
        if not self.mu or not self.std:
            print("Warning, not enough samples collected for tSNE, \
                  try to reduce sampling interval")
            return

        # prepare plot data 
        mu = np.concatenate(self.mu, axis = 0)
        std = np.concatenate(self.std, axis = 0)

        # communicate if necessary
        if (self.comm is not None):
            # gather data
            mu_gather = self.comm.gather(mu, root = 0)
            std_gather = self.comm.gather(std, root = 0)
            
            # concat
            if self.is_eval_node:
                mu = np.concatenate(mu_gather, axis = 0)
                std = np.concatenate(std_gather, axis = 0)
        
        # dist plots
        if self.is_eval_node and (self.sample_interval > 0):
            self.dist_plot(epoch, mu, std, logs)

        # we need to wait for it 
        if (self.comm is not None):
            self.comm.barrier()

        
    def dist_plot(self, epoch, mu, std, logs):

        # create plot grid
        ncols = 16
        nrows = mu.shape[1] // (ncols // 2)

        # create figure
        fig, axs = plt.subplots(figsize=(ncols*2, nrows*1.5),
                                nrows = nrows, ncols = ncols)

        titlestring = f'Latent Space Distributions after Epoch {epoch}'
        
        # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
        #       then run tSNE on outputs of PCA. This works for sparse matrices
        #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html

        for idx in range(mu.shape[1]):
        
            colid = idx % (ncols // 2)
            rowid = idx // (ncols // 2)
            
            # plot histograms:
            # mean
            ax = axs[rowid, 2 * colid + 0]
            ax.set_xlabel(r'$mu$') 
            ax.hist(mu[:,idx], bins='fd', density=True, color='dodgerblue', alpha=0.8)
            ## target
            #ax.axvline(x=0., color='dodgerblue', ls="--")
            # std
            ax = axs[rowid, 2 * colid + 1]
            ax.set_xlabel(r'$std$')
            ax.hist(std[:,idx], bins='fd', density=True, color='crimson', alpha=0.8)
            #ax.axvline(x=1., color='crimson', ls="--")
            

        # tight layout
        plt.tight_layout()

        # save figure
        time_stamp = time.strftime(f'latspace-step-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)

        # wandb logging
        if self.wandb_config is not None:
            img = Image.open(os.path.join(self.out_dir, time_stamp))
            wandb.log({"step latent space distributions": [wandb.Image(img, caption="Latent Space Distributions")]}, step = logs['global_step'])

        # close plot
        plt.close(fig)
