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


class EmbeddingCallback(Callback):

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
                 projection_type = "3d",
                 sample_interval = 20,
                 writer = None, wandb_config = None):
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
        """

        os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.projection_type = projection_type.lower()
        self.writer = writer
        self.wandb_config = wandb_config

        # needed for init plot
        self._init_plot(path, rmsd_name)

        
    def _init_plot(self, path, rmsd_name):
        # load all rmsd data
        f = open_h5(path)
        rmsd = f[rmsd_name][..., 2]
        vmin, vmax = self.minmax(rmsd)

        # create colormaps
        cmi = plt.get_cmap('jet')
        cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        self.scalar_map.set_array(rmsd)

        # perplexities
        self.perplexities = [2, 5, 30, 50, 100]

        
    def on_epoch_end(self, epoch, logs):
        # prepare plot data
        embeddings = logs["embeddings"][::self.sample_interval,...]
        rmsd = logs["rmsd"][::self.sample_interval,...]

        # t-sne plots
        self.tsne_plot(embeddings, rmsd, logs)

        
    def tsne_plot(self, embeddings, rmsd, logs):

        # create plot grid
        nrows = len(self.perplexities)
        ncols = 3 if (self.projection_type == "3d_project") else 1

        # create figure
        fig, axs = plt.subplots(figsize=(ncols * 4, nrows * 4),
                                nrows = nrows, ncols = ncols,
                                sharey = True)

        # set up constants
        color = self.scalar_map.to_rgba(rmsd)
        titlestring = f'RMSD to native state after step {logs["global_step"]}'
        
        # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
        #       then run tSNE on outputs of PCA. This works for sparse matrices
        #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html

        for idr, perplexity in enumerate(self.perplexities):
        
            # Outputs 3D embeddings using all available processors
            tsne = TSNE(n_components = int(self.projection_type[0]), n_jobs=-1, perplexity = perplexity)

            # TODO: running on cpu as a numpy array may be an issue for large systems
            #       consider using pytorch tSNE implemenation. Drawback is that
            #       so far there are only 2D versions implemented.
            embeddings = tsne.fit_transform(embeddings)

            # plot
            if self.projection_type == "3d":
                ax = axs[idr]
                z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
                ax.scatter3D(z1, z2, z3, marker = '.', c = color)
                ax.set_xlim3d(self.minmax(z1))
                ax.set_ylim3d(self.minmax(z2))
                ax.set_zlim3d(self.minmax(z3))
                ax.set_xlabel(r'$z_1$')
                ax.set_ylabel(r'$z_2$')
                ax.set_zlabel(r'$z_3$')
                if idr == 0:
                    ax.set_title(titlestring)
                fig.colorbar(self.scalar_map)
            
            elif self.projection_type == "3d_project":
                z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
                z1mm = self.minmax(z1)
                z2mm = self.minmax(z2)
                z3mm = self.minmax(z3)
                zmm = (min([z1mm[0], z2mm[0], z3mm[0]]), max([z1mm[1], z2mm[1], z3mm[1]]))
                # x-y
                ax1 = axs[idr, 0]
                ax1.scatter(z1, z2, marker = '.', c = color)
                ax1.set_xlim(zmm)
                ax1.set_ylim(zmm)
                ax1.set_xlabel(r'$z_1$')
                ax1.set_ylabel(r'$z_2$')
                # x-z
                ax2 = axs[idr, 1]
                ax2.scatter(z1, z3, marker = '.', c = color)
                ax2.set_xlim(zmm)
                ax2.set_ylim(zmm)
                ax2.set_xlabel(r'$z_1$')
                ax2.set_ylabel(r'$z_3$')
                if idr == 0:
                    ax2.set_title(titlestring)
                # y-z
                ax3 = axs[idr, 2]
                ax3.scatter(z2, z3, marker = '.', c = color)
                ax3.set_xlim(zmm)
                ax3.set_ylim(zmm)
                ax3.set_xlabel(r'$z_2$')
                ax3.set_ylabel(r'$z_3$')
                # colorbar
                divider = make_axes_locatable(axs[idr, 2])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(self.scalar_map, ax = axs[idr, 2], cax = cax)
            
            else:
                ax = axs[idr]
                z1, z2 = embeddings[:, 0], embeddings[:, 1]
                ax.scatter(z1, z2, marker = '.', c = color)
                ax.set_xlim(self.minmax(z1))
                ax.set_ylim(self.minmax(z2))
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
        time_stamp = time.strftime(f'step-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)

        # summary writer
        if self.writer is not None:
            self.writer.add_figure('step t-SNE embeddings', fig, logs['global_step'])

        # wandb logging
        if self.wandb_config is not None:
            img = Image.open(os.path.join(self.out_dir, time_stamp))
            wandb.log({"step t-SNE embeddings": [wandb.Image(img, caption="Latent Space Visualizations")]}, step = logs['global_step'])

        # close plot
        plt.close(fig)
