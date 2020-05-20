import numpy as np
from keras.callbacks import Callback


# TODO: remove these or migrate to pytorch

class EmbeddingCallback(Callback):
    """
    Saves embeddings of random samples.

    Parameters
    ----------
    data : np.ndarray
        Dataset from which to sample for embeddings.

    """
    def __init__(self, data, graph):
        self.data = data
        self.graph = graph

    def on_train_begin(self, logs={}):
        self.embeddings = []
        self.data_index = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        # TODO: test whether self.model can be used instead of storing self.graph.
        #       self.model is stored by keras in Callback
        idx = np.random.randint(0, len(self.data))
        embedding = self.graph.embed(self.data[idx - 1: idx])
        self.data_index.append(idx)
        self.embeddings.append(embedding)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

    def save(self, embed_path, idx_path):
        """
        Save embeddings and index of associated data point.

        Parameters
        ----------
        embed_path: str
            Path to save embedding

        idx_path : str
            Path to save embedding indices

        """
        np.save(embed_path, self.embeddings)
        np.save(idx_path, self.data_index)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def save(self, loss_path, val_loss_path):
        """
        Save train and validation loss from the end of each epoch.

        Parameters
        ----------
        loss_path: str
            Path to save loss

        val_loss_path : str
            Path to save validation loss

        """
        np.save(loss_path, self.losses)
        np.save(val_loss_path, self.val_losses)
