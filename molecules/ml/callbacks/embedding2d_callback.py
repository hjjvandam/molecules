import os
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from sklearn.manifold import TSNE
from .callback import Callback
from PIL import Image
import numba
import wandb
from molecules.utils import open_h5


class Embedding2dCallback(Callback):

    # Helper function. Returns tuple of min and max of 1d np.ndarray.
    @staticmethod
    @numba.jit
    def minmax(x):
        max_, min_ = x[0], x[0]
        for i in x[1:]:
            if i > max_:
                max_ = i
            elif i < min_:
                min_ = i
        return min_, max_
    """
    Saves VAE embeddings of random samples.
    """
    def __init__(self, out_dir,
                 path, rmsd_name,
                 projection_type = "2d",
                 sample_interval = 20,
                 writer = None,
                 wandb_config = None,
                 mpi_comm = None):
        """
        Parameters
        ----------
        path : str
            H5 File with rmsd data.
        
        rmsd_name : str
            Dataset name for rmsd data.
        projection_type: str
            Type of projection: 2D or 3D.
        out_dir : str
            Directory to store output plots.
        sample_interval : int
            Plots every sample_interval'th point in the data set
        writer : torch.utils.tensorboard.SummaryWriter
        wandb_config : wandb configuration file
        mpi_comm : mpi communicator for distributed training
        """
        self.comm = mpi_comm
        self.is_eval_node = True
        if (self.comm is not None) and (self.comm.Get_rank() != 0):
            self.is_eval_node = False

        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.projection_type = projection_type.lower()
        self.writer = writer
        self.wandb_config = wandb_config

        # needed for init plot
        if self.is_eval_node:
            self._init_plot(path, rmsd_name)

        
    def _init_plot(self, path, rmsd_name):
        # load all rmsd data
        with open_h5(path) as f:
            rmsd = f[rmsd_name][...]
        vmin, vmax = self.minmax(rmsd)

        # create colormaps
        cmi = plt.get_cmap('jet')
        cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        self.scalar_map.set_array(rmsd)

        # perplexities
        self.perplexities = [2, 5, 30, 50, 100]


    def on_validation_begin(self, epoch, logs):
        self.sample_counter = 0
        self.embeddings = []
        self.rmsd = []
        
        
    def on_validation_batch_end(self, logs, mu = None, rmsd = None, **kwargs):
        if self.sample_interval == 0:
            return

        if (mu is None) or (rmsd is None):
            pass
        
        # decide what to store
        for idx in range(0, len(mu)):
            if (self.sample_counter + idx) % self.sample_interval == 0:
                # use a singleton slice to keep dimensions intact
                self.embeddings.append(mu[idx:idx+1].detach().cpu().numpy())
                self.rmsd.append(rmsd[idx:idx+1].detach().cpu().numpy())

        # increase sample counter
        self.sample_counter += len(mu)

        
    def on_validation_end(self, epoch, logs):
        # if the sample interval was too large, we should warn here and return
        if not self.embeddings or not self.rmsd:
            print("Warning, not enough samples collected for tSNE, \
                  try to reduce sampling interval")
            return

        # prepare plot data 
        embeddings = np.concatenate(self.embeddings, axis = 0).astype(np.float32)
        rmsd = np.concatenate(self.rmsd, axis = 0).astype(np.float32)

        # communicate if necessary
        if (self.comm is not None):
            # gather data
            embeddings_gather = self.comm.gather(embeddings, root = 0)
            rmsd_gather = self.comm.gather(rmsd, root = 0)

            # concat
            if self.is_eval_node:
                embeddings = np.concatenate(embeddings_gather, axis = 0)
                rmsd = np.concatenate(rmsd_gather, axis = 0)
        
        # t-sne plots
        if self.is_eval_node and (self.sample_interval > 0):
            self.tsne_plot(epoch, embeddings, rmsd, logs)

        # we need to wait for it
        if (self.comm is not None):
            self.comm.barrier()

        
    def tsne_plot(self, epoch, embeddings, rmsd, logs):

        # create plot grid
        nrows = len(self.perplexities)
        ncols = 3 if (self.projection_type == "3d_project") else 1

        # create figure
        fig, axs = plt.subplots(figsize=(ncols * 4, nrows * 4),
                                nrows = nrows, ncols = ncols)

        # set up constants
        color = self.scalar_map.to_rgba(rmsd)
        titlestring = f'RMSD to reference state after epoch {epoch}'
        
        # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
        #       then run tSNE on outputs of PCA. This works for sparse matrices
        #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html

        for idr, perplexity in enumerate(self.perplexities):
        
            # Outputs 3D embeddings using all available processors
            tsne = TSNE(n_components = int(self.projection_type[0]), n_jobs=-1, perplexity = perplexity)

            # TODO: running on cpu as a numpy array may be an issue for large systems
            #       consider using pytorch tSNE implemenation. Drawback is that
            #       so far there are only 2D versions implemented.
            emb_trans = tsne.fit_transform(embeddings)

            # plot            
            if self.projection_type == "3d_project":
                z1, z2, z3 = emb_trans[:, 0], emb_trans[:, 1], emb_trans[:, 2]
                z1mm = self.minmax(z1)
                z2mm = self.minmax(z2)
                z3mm = self.minmax(z3)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                z3mm = (z3mm[0] * 0.95, z3mm[1] * 1.05)
                # x-y
                ax1 = axs[idr, 0]
                ax1.scatter(z1, z2, marker = '.', c = color)
                ax1.set_xlim(z1mm)
                ax1.set_ylim(z2mm)
                ax1.set_xlabel(r'$z_1$')
                ax1.set_ylabel(r'$z_2$')
                # x-z
                ax2 = axs[idr, 1]
                ax2.scatter(z1, z3, marker = '.', c = color)
                ax2.set_xlim(z1mm)
                ax2.set_ylim(z3mm)
                ax2.set_xlabel(r'$z_1$')
                ax2.set_ylabel(r'$z_3$')
                if idr == 0:
                    ax2.set_title(titlestring)
                # y-z
                ax3 = axs[idr, 2]
                ax3.scatter(z2, z3, marker = '.', c = color)
                ax3.set_xlim(z2mm)
                ax3.set_ylim(z3mm)
                ax3.set_xlabel(r'$z_2$')
                ax3.set_ylabel(r'$z_3$')
                # colorbar
                divider = make_axes_locatable(axs[idr, 2])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(self.scalar_map, ax = axs[idr, 2], cax = cax)
            
            else:
                ax = axs[idr]
                z1, z2 = emb_trans[:, 0], emb_trans[:, 1]
                ax.scatter(z1, z2, marker = '.', c = color)
                z1mm = self.minmax(z1)
                z2mm = self.minmax(z2)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                ax.set_xlim(z1mm)
                ax.set_ylim(z2mm)
                ax.set_xlabel(r'$z_1$')
                ax.set_ylabel(r'$z_2$')
                if idr == 0:
                    ax.set_title(titlestring)
                # colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(self.scalar_map, ax = axs, cax = cax)

        # tight layout
        plt.tight_layout()

        # save figure
        time_stamp = time.strftime(f'embeddings-step-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)

        # summary writer
        if self.writer is not None:
            self.writer.add_figure('epoch t-SNE embeddings', fig, epoch)

        # wandb logging
        if self.wandb_config is not None:
            img = Image.open(os.path.join(self.out_dir, time_stamp))
            wandb.log({"step t-SNE embeddings": [wandb.Image(img, caption="Latent Space Visualizations")]}, step = logs['global_step'])

        # close plot
        plt.close(fig)
