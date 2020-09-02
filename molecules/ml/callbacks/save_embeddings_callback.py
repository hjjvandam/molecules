import os
import numpy as np
from .callback import Callback
from molecules.utils import open_h5


class SaveEmbeddingsCallback(Callback):

    """
    Saves embeddings
    """
    def __init__(self, out_dir,
                 interval=1,
                 sample_interval=20,
                 mpi_comm=None):
        """
        Parameters
        ----------
        out_dir : str
            Directory to store output embedding files.
        interval : int
            Plots every interval epochs, default is once per epoch.
        sample_interval : int
            Plots every sample_interval'th point in the data set
        mpi_comm : mpi communicator for distributed training
        """
        super().__init__(interval, mpi_comm)

        if self.is_eval_node:
            os.makedirs(out_dir, exist_ok=True)

        self.out_dir = out_dir
        self.sample_interval = sample_interval


    def on_validation_begin(self, epoch, logs):
        self.sample_counter = 0
        self.embeddings = []
        self.rmsd = []
        
        
    def on_validation_batch_end(self, batch, epoch, logs, mu=None, rmsd=None, **kwargs):
        if epoch % self.interval != 0:
            return
        if self.sample_interval == 0:
            return
        if (mu is None) or (rmsd is None):
            return
        
        # decide what to store
        for idx in range(len(mu)):
            if (self.sample_counter + idx) % self.sample_interval == 0:
                # use a singleton slice to keep dimensions intact
                self.embeddings.append(mu[idx:idx+1].detach().cpu().numpy())
                self.rmsd.append(rmsd[idx:idx+1].detach().cpu().numpy())

        # increase sample counter
        self.sample_counter += len(mu)

        
    def on_validation_end(self, epoch, logs):
        if epoch % self.interval != 0:
            return
        # if the sample interval was too large, we should warn here and return
        if not self.embeddings or not self.rmsd:
            print('Warning, not enough samples collected for tSNE, \
                  try to reduce sampling interval')
            return

        # prepare plot data 
        embeddings = np.concatenate(self.embeddings, axis=0).astype(np.float32)
        rmsd = np.concatenate(self.rmsd, axis=0).astype(np.float32)

        # communicate if necessary
        if self.comm is not None:
            # gather data
            embeddings_gather = self.comm.gather(embeddings, root=0)
            rmsd_gather = self.comm.gather(rmsd, root=0)

            # concat
            if self.is_eval_node:
                embeddings = np.concatenate(embeddings_gather, axis=0)
                rmsd = np.concatenate(rmsd_gather, axis=0)
        
        # Save embeddings to disk
        if self.is_eval_node and (self.sample_interval > 0):
            self.save_embeddings(epoch, embeddings, rmsd, logs)

        # All other nodes wait for node 0 to save
        if self.comm is not None:
            self.comm.barrier()


    def save_embeddings(self, epoch, embeddings, rmsd, logs):
        # Create embedding file path and store in logs for downstream callbacks
        time_stamp = time.strftime(f'embeddings-raw-step-{logs["global_step"]}-%Y%m%d-%H%M%S.h5')
        embeddings_path = os.path.join(self.out_dir, time_stamp)
        logs['embeddings_path'] = embeddings_path

        # Write embedding data to disk
        with open_h5(embeddings_path, 'w', swmr=False) as f:
            f['embeddings'] = embeddings[...]
            f['rmsd'] = rmsd[...]
