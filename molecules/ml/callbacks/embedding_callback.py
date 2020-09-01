import os
from .callback import Callback
from molecules.plot import plot_tsne
import concurrent.futures as cf


class EmbeddingCallback(Callback):
    """
    Saves t-SNE embedding plots.
    """
    def __init__(self, out_dir,
                 interval=1, colors=['rmsd'],
                 projection_type='2d',
                 target_perplexity=30,
                 perplexities=[5, 30, 50, 100, 200],
                 pca=True, pca_dim=50,
                 wandb_config=None,
                 mpi_comm=None):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output plots.
        interval : int
            Plots every interval epochs, default is once per epoch.
        wandb_config : wandb configuration file
        mpi_comm: mpi communicator
        """
        if interval < 1:
            raise ValueError('Plot interval must be int greater than 0')

        super().__init__(mpi_comm)

        if self.is_eval_node:
            os.makedirs(self.out_dir, exist_ok=True)

            self.interval = interval
            self.tsne_kwargs = {
                'out_dir': out_dir,
                'wandb_config': wandb_config, 
                'projection_type': projection_type,
                'target_perplexity': target_perplexity,
                'perplexities': perplexities,
                'pca': pca,
                'pca_dim': pca_dim
            }

            # Need for async plotting
            self.executor = cf.ThreadPoolExecutor(max_workers=1)
            self.future_tsne = None

    def on_epoch_end(self, epoch, logs):
        if self.is_eval_node and (epoch % self.interval == 0):

            # Wait for the old stuff to finish
            if self.future_tsne is not None:
                # TODO: may need to add timeout if there are errors
                #       and raise an exception if it fails
                cf.wait(self.future_tsne, timeout=None)

            # TODO: get embeddings_path from logs
            self.future_tsne = self.executor.submit(plot_tsne,
                                                    embedding_path=embedding_path,
                                                    global_step=logs['global_step'],
                                                    epoch=epoch,
                                                    **self.tsne_kwargs)
