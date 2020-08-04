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
    def __init__(self, path, out_dir, squeeze,
                 sample_interval=20, batch_size=128,
                 writer=None):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.

        out_dir : str
            Directory to store output plots.

        squeeze : bool
            If True, data is reshaped to (H, W) else if False data
            is reshaped to (1, H, W).

        sample_interval : int
            Plots every sample_interval'th point in the data set

        batch_size : int
            Batch size to load raw contact matrices into memory.
            Batches are loaded into memory, encoded to a latent
            dimension and then collected in a np.ndarray. The
            np.ndarray is then passed to the TSNE algorithm.

            NOTE: Not a learning hyperparameter, simply needs to
                  be small enough to load batch into memory.

        writer : torch.utils.tensorboard.SummaryWriter
        """

        os.makedirs(out_dir, exist_ok=True)

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        h5_file = open_h5(path)

        self.dset = h5_file['contact_maps']
        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.batch_size = batch_size
        self.writer = writer
        # TODO: allow any input device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if squeeze:
            self.shape = (self.dset.shape[1], self.dset.shape[2])
        else:
            self.shape = (1, self.dset.shape[1], self.dset.shape[2])

        self._init_plot(h5_file)

    def batches(self):
        """
        Generator to return batches of contact map dset.
        Batches consist of every self.sample_interval'th point.
        NOTE: last batch may have diferent size.
        """
        start, step = 0, self.sample_interval * self.batch_size
        for idx in range(0, len(self.dset), step):
            yield torch.from_numpy(
                     np.array(self.dset[start: start + step: self.sample_interval]) \
                       .reshape(-1, *self.shape)).to(torch.float32).to(self.device)
            start += step

    def sample(self, h5_file):

        # TODO: compute native contacts

        rmsd_dset = h5_file['rmsd']
        rmsd = np.array(rmsd_dset[0: len(rmsd_dset): self.sample_interval][:, 2])
        return rmsd #, native_contacts

    def _init_plot(self, h5_file):

        rmsd = self.sample(h5_file)

        # TODO: Make rmsd_fig and nc_fig
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        vmin, vmax = self.minmax(rmsd)
        cmi = plt.get_cmap('jet')
        cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        scalar_map.set_array(rmsd)
        self.fig.colorbar(scalar_map)
        self.color = scalar_map.to_rgba(rmsd)

    def on_epoch_end(self, epoch, logs):
        self.tsne_plot(logs)

    def tsne_plot(self, logs):

        # Collect latent vectors for TSNE.
        # Process is batched in case dset does not fit into memory.
        embeddings = np.concatenate([logs['model'].encode(batch).cpu().numpy()
                                    for batch in self.batches()])

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
        self.ax.clear()
