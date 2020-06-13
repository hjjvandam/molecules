import os
import time
import torch

class Callback:
    def __init__(self): pass
    def on_train_begin(self, logs): pass
    def on_train_end(self, logs): pass
    def on_epoch_begin(self, epoch, logs): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_batch_begin(self, batch, epoch, logs): pass
    def on_batch_end(self, batch, epoch, logs): pass


# TODO: need way to share SummaryWriter among multiple callbacks for a model
#       could make writer global variable

class LossCallback(Callback):
    def on_train_begin(self, logs):
        #from torch.utils.tensorboard import SummaryWriter
        #self.writer = SummaryWriter()

        self.train_losses = []
        self.valid_losses = []

    def on_epoch_end(self, epoch, logs):

        # self.writer.add_scalar('epoch training loss',
        #                        logs['train_loss'],
        #                        logs['global_step'])
        # self.writer.add_scalar('epoch validation loss',
        #                        logs['valid_loss'],
        #                        logs['global_step'])

        self.train_losses.append(logs['train_loss'])
        self.valid_losses.append(logs['valid_loss'])

    def save(self, path):
        """
        Save train and validation loss from the end of each epoch.

        Parameters
        ----------
        path: str
            Path to save train and validation loss history
        """
        torch.save({'loss': self.train_losses, 'valid': self.valid_losses}, path)

class CheckpointCallback(Callback):
    def __init__(self, interval=0,
                 directory=os.path.join('.', 'checkpoints')):
        """
        Checkpoint interface for saving dictionary objects to disk
        during training. Typically used to save model state_dict
        and optimizer state_dict in order to resume training and
        record model weight history.

        Parameters
        ----------
        directory : str
            Directory to store checkpoint files.
            Files are named 'epoch-{e}-%Y%m%d-%H%M%S.pt'

        interval : int
            Checkpoints model every interval batches, default is once per epoch.
        """

        if interval < 0:
            raise ValueError('Checkpoint interval must be non-negative')

        os.makedirs(directory, exist_ok=True)

        self.interval = interval
        self.directory = directory

    def on_batch_end(self, batch, epoch, logs):
        if self.interval and batch % self.interval == 0:
            self._save(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        if not self.interval:
            self._save(epoch, logs)

    def _save(self, epoch, logs):
        """Saves optimizer state and encoder/decoder weights."""

        checkpoint = {
            'encoder_state_dict': logs['model'].encoder.state_dict(),
            'decoder_state_dict': logs['model'].decoder.state_dict(),
            'optimizer_state_dict': logs['optimizer'].state_dict(),
            'epoch': epoch
            }

        time_stamp = time.strftime(f'epoch-{epoch}-%Y%m%d-%H%M%S.pt')
        path = os.path.join(self.directory, time_stamp)
        torch.save(checkpoint, path)

# TODO: add numba dependency to conda env
import numba
@numba.jit
def minmax(x):
    max_, min_ = x[0], x[0]
    for i in x[1:]:
        if i > max_:
            max_ = i
        elif i < min_:
            min_ = i
    return min_, max_

class EmbeddingCallback(Callback):
    """
    Saves embeddings of random samples.

    Parameters
    ----------
    data : torch.Tensor
        Dataset from which to sample for embeddings.

    """
    def __init__(self, data, rmsd=None):
        # TODO: put data to_device
        self.data = data
        self.i = 0

        if rmsd is not None:
            self._init_plot(rmsd)

    def _init_plot(self, rmsd):
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        cmi = plt.get_cmap('jet')
        cnorm = matplotlib.colors.Normalize(vmin=min(rmsd), vmax=max(rmsd))
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        scalar_map.set_array(rmsd)
        fig.colorbar(scalar_map)

        self.color = scalar_map.to_rgba(rmsd)

    def on_train_begin(self, logs):
        self.embeddings = []
        self.data_index = []

    def on_epoch_end(self, epoch, logs):
        # TODO: may need to change the torch device
        idx = torch.randint(len(self.data), (1,))
        embedding = logs['model'].encode(self.data[idx])
        self.data_index.append(idx)
        self.embeddings.append(embedding)

        if hasattr(self, 'ax'):
            self.tsne_plot(logs)

    def tsne_plot(self, logs):
        # TODO: factor out imports. put this callback in it's own file
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # Outputs 3D embeddings using all available processors
        tsne = TSNE(n_components=3, n_jobs=-1)
        embeddings = tsne.fit_transform(logs['model'].encode(self.data)) # TODO: convert to np?

        z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]

        self.ax.scatter3D(z1, z2, z3, marker='.', c=self.color)
        self.ax.set_xlim3d(minmax(z1))
        self.ax.set_ylim3d(minmax(z2))
        self.ax.set_zlim3d(minmax(z3))
        self.ax.set_xlabel(r'$z_1$')
        self.ax.set_ylabel(r'$z_2$')
        self.ax.set_zlabel(r'$z_3$')
        self.ax.set_title(f'RMSD to native state after epoch {self.i}')
        plt.savefig(f'./encoded_train-{self.i}.png', dpi=300)
        self.ax.clear()

        # TODO: add filename member var and remove self.i,
        #       format: filename-epoch-i-timestamp.png
        self.i += 1

    def save(self, path):
        """
        Save embeddings and index of associated data point.

        Parameters
        ----------
        path: str
            Path to save embeddings and indices

        """

        torch.save({'embeddings': self.embeddings, 'indices': self.data_index}, path)
