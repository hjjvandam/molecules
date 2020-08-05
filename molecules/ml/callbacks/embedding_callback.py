import os
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
                 sample_interval=20,
                 writer=None, wandb_config=None):
        """
        Parameters
        ----------
        path : str
            H5 File with rmsd data.
        
        rmsd_name : str
            Dataset name for rmsd data.

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

        
    def on_epoch_end(self, epoch, logs):
        # prepare plot data
        embeddings = logs["embeddings"][::self.sample_interval,...]
        rmsd = logs["rmsd"][::self.sample_interval,...]

        # t-sne plots
        self.tsne_plot(embeddings, rmsd, logs)

        
    def tsne_plot(self, embeddings, rmsd, logs):

        # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
        #       then run tSNE on outputs of PCA. This works for sparse matrices
        #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html

        # TODO: plot different charts using different perplexity values

        # Outputs 3D embeddings using all available processors
        tsne = TSNE(n_components=3, n_jobs=-1)

        # TODO: running on cpu as a numpy array may be an issue for large systems
        #       consider using pytorch tSNE implemenation. Drawback is that
        #       so far there are only 2D versions implemented.
        embeddings = tsne.fit_transform(embeddings)

        z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        color = self.scalar_map.to_rgba(rmsd)
        ax.scatter3D(z1, z2, z3, marker = '.', c = color)
        ax.set_xlim3d(self.minmax(z1))
        ax.set_ylim3d(self.minmax(z2))
        ax.set_zlim3d(self.minmax(z3))
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.set_zlabel(r'$z_3$')
        ax.set_title(f'RMSD to native state after epoch {logs["global_step"]}')
        fig.colorbar(self.scalar_map)

        # save figure
        time_stamp = time.strftime(f'epoch-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)

        # summary writer
        if self.writer is not None:
            self.writer.add_figure('epoch t-SNE embeddings', fig, logs['global_step'])

        # wandb logging
        if self.wandb_config is not None:
            img = Image.open(os.path.join(self.out_dir, time_stamp))
            wandb.log({"epoch t-SNE embeddings": [wandb.Image(img, caption="Latent Space Visualizations")]}, step = logs['global_step'])

        # close plot
        plt.close(fig)
