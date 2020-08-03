import os
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import TSNE
from .callback import Callback
from molecules.utils import open_h5
import numba
import wandb


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
                 sample_interval=20,
                 writer=None, wandb_config=None):
        """
        Parameters
        ----------
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


    def on_epoch_end(self, epoch, logs):
        # prepare plot data
        embeddings = logs["embeddings"][::self.sample_interval,...]

        # t-sne plots
        self.tsne_plot(embeddings)

        
    def tsne_plot(self, embeddings):

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

        self.ax.scatter3D(z1, z2, z3, marker='.', c=self.color)
        self.ax.set_xlim3d(self.minmax(z1))
        self.ax.set_ylim3d(self.minmax(z2))
        self.ax.set_zlim3d(self.minmax(z3))
        self.ax.set_xlabel(r'$z_1$')
        self.ax.set_ylabel(r'$z_2$')
        self.ax.set_zlabel(r'$z_3$')
        self.ax.set_title(f'RMSD to native state after epoch {logs["global_step"]}')
        time_stamp = time.strftime(f'epoch-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)
        if self.writer is not None:
            self.writer.add_figure('epoch t-SNE embeddings', self.fig, logs['global_step'])

        if self.wandb_config is not None:
            wandb.log({'epoch t-SNE embeddings': self.fig}, step = logs['global_step'])
        self.ax.clear()
