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


class Embedding3dCallback(Callback):

    # Helper function. Returns tuple of min and max of 1d np.ndarray.
    # Min and max in single pass.
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
    def __init__(self, path, out_dir, shape, sparse=False,
                 sample_interval=20, batch_size=128,
                 gpu=None, writer=None):
        """
        Parameters
        ----------
        path : str
            Path to h5 file containing contact matrices.
        out_dir : str
            Directory to store output plots.
        shape : tuple
            Shape of contact matrices (H, W), may be (1, H, W).
        sparse : bool
            If True, process data as sparse row/col COO format. Data
            should not contain any values because they are all 1's and
            generated on the fly. If False, input data is normal tensor.
        sample_interval : int
            Plots every sample_interval'th point in the data set
        batch_size : int
            Batch size to load raw contact matrices into memory.
            Batches are loaded into memory, encoded to a latent
            dimension and then collected in a np.ndarray. The
            np.ndarray is then passed to the TSNE algorithm.
            NOTE: Not a learning hyperparameter, simply needs to
                  be small enough to load batch into memory.
        gpu : int, None
            If None, then data will be put onto the default GPU if CUDA
            is available and otherwise is put onto a CPU. If gpu is int
            type, then data is put onto the specified GPU.
        writer : torch.utils.tensorboard.SummaryWriter
        """

        os.makedirs(out_dir, exist_ok=True)

        # Open h5 file. Python's garbage collector closes the
        # file when class is destructed.
        h5_file = open_h5(path)

        if sparse:
            group = h5_file['contact_maps']
            self.row_dset = group.get('row')
            self.col_dset = group.get('col')
            self.len = len(self.row_dset)
        else:
            # contact_maps dset has shape (N, W, H, 1)
            self.dset = h5_file['contact_maps']
            self.len = len(self.dset)

        self.shape = shape
        self.sparse = sparse
        self.out_dir = out_dir
        self.sample_interval = sample_interval
        self.batch_size = batch_size
        self.writer = writer

        if gpu is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(f'cuda:{gpu}')

        self._init_plot(h5_file)

    def batches(self):
        """
        Generator to return batches of contact map dset.
        Batches consist of every self.sample_interval'th point.
        NOTE: last batch may have diferent size.
        """
        start, step = 0, self.sample_interval * self.batch_size
        for idx in range(0, self.len, step):
            if self.sparse:
                # Retrieve sparse row,col from disk and make dense in memory
                data = []
                for i in range(start, start + step, self.sample_interval):
                    if i >= self.len:
                        break
                    indices = torch.from_numpy(np.vstack((self.row_dset[i],
                                                          self.col_dset[i]))) \
                                          .to(self.device).to(torch.long)
                    values = torch.ones(indices.shape[1], dtype=torch.float32,
                                        device=self.device)
                    # Set shape to the last 2 elements of self.shape.
                    # Handles (1, W, H) and (W, H)
                    data.append(torch.sparse.FloatTensor(indices, values, self.shape[-2:]).to_dense())

                yield torch.stack(data).view(-1, *self.shape)

            else:
                yield torch.from_numpy(
                        np.array(self.dset[start: start + step: self.sample_interval]) \
                            .reshape(-1, *self.shape)).to(torch.float32).to(self.device)
            start += step

    def sample(self, h5_file):
        """
        Returns
        -------
        tuple : np.arrays of RMSD to native state and fraction of
                native contacts sampled every self.sample_interval'th
                frames of the MD trajectory.
        (rmsd array, fnc array) arrays of equal length.
        """
        return (np.array(dset[0: len(dset): self.sample_interval])
                for dset in (h5_file['rmsd'], h5_file['fnc']))

    def _init_plot(self, h5_file):

        rmsd, fnc = self.sample(h5_file)

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
        self.tsne_plot(epoch, logs)

    def tsne_plot(self, epoch, logs):

        # Collect latent vectors for TSNE.
        # Process is batched in case dset does not fit into memory.
        embeddings = np.concatenate([logs['model'].encode(batch).cpu().numpy()
                                    for batch in self.batches()])

        print('Number of embeddings: ', len(embeddings))
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
        self.ax.set_title(f'RMSD to reference state after epoch {epoch}')
        time_stamp = time.strftime(f'epoch-{logs["global_step"]}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(self.out_dir, time_stamp), dpi=300)
        if self.writer is not None:
            self.writer.add_figure('epoch t-SNE embeddings', self.fig, logs['global_step'])
        self.ax.clear()
