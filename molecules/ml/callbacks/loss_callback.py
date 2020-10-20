import json
import wandb
import numpy as np
from .callback import Callback

class LossCallback(Callback):
    
    def __init__(self, path,
                 interval=1,
                 wandb_config=None,
                 mpi_comm=None):
        """
        Parameters
        ----------
        path : str
            path to save loss history to
        interval : int
            Plots every interval epochs, default is once per epoch.
        wandb_config : wandb configuration file
        mpi_comm : mpi communicator for distributed training
        """
        super().__init__(interval, mpi_comm)

        self.path = path
        self.wandb_config = wandb_config
        
    def on_train_begin(self, logs):
        self.epochs = []
        self.train_losses = {}
        self.valid_losses = {}

        
    def on_epoch_end(self, epoch, logs):
        # Only store loss every self.interval epochs
        if epoch % self.interval != 0:
            return

        # epochs
        self.epochs.append(epoch)
    
        # train_losses
        for lossname in [x for x in logs if x.startswith("train_loss")]:

            # reduce losses
            if self.comm is not None:
                lossarr = np.array(logs[lossname], dtype = np.float32)
                logs[lossname] = np.asscalar(self.comm.allreduce(lossarr))
                logs[lossname] /= float(self.comm.Get_size())
            
            # manual logging
            if lossname in self.train_losses:
                self.train_losses[lossname].append(logs[lossname])
            else:
                self.train_losses[lossname] = [logs[lossname]]

            # wandb
            if self.wandb_config is not None:
                wandb.log({lossname: logs[lossname]}, step = logs["global_step"])
                    

        # validation losses
        for lossname in [x for x in logs if x.startswith("valid_loss")]:

            # reduce losses
            if self.comm is not None:
                lossarr = np.array(logs[lossname], dtype = np.float32)
                logs[lossname] = np.asscalar(self.comm.allreduce(lossarr))
                logs[lossname] /= float(self.comm.Get_size())
                
            # manual logging
            if lossname in self.valid_losses:
                self.valid_losses[lossname].append(logs[lossname])
            else:
                self.valid_losses[lossname] = [logs[lossname]]

            # wandb
            if self.wandb_config is not None:
                wandb.log({lossname: logs[lossname]}, step = logs["global_step"])

                
        # save to json for manual logging
        if self.is_eval_node:
            self.save(self.path)


    def save(self, path):
        """
        Save train and validation loss from the end of each epoch.

        Parameters
        ----------
        path: str
            Path to save train and validation loss history
        """

        # Happens when loading from a checkpoint
        if self.epochs[0] != 1:
            with open(path) as f:
                data = json.load(f)
                if data:
                    # Prepend data from checkpointed model to the start of the
                    # current logs. This avoids needing to load the data every
                    # time the logs are saved.
                    # epochs
                    data['epochs'].extend(self.epochs)
                    self.epochs = data['epochs']

                    # train losses
                    for lossname in [x for x in data if x.startswith("train_loss")]:
                        data[lossname].extend(self.train_losses[lossname])
                        self.train_losses[lossname] = data[lossname]

                    # valid losses
                    for lossname in [x for x in data if x.startswith("valid_loss")]:
                        data[lossname].extend(self.valid_losses[lossname])
                        self.valid_losses[lossname] = data[lossname]

        # Write history to disk
        with open(path, 'w') as f:
            # construct json dump file
            jsondict = {}
            for lossname in [x for x in self.train_losses if x.startswith("train_loss")]:
                jsondict[lossname] = self.train_losses[lossname]
            for lossname in [x for x in self.valid_losses if x.startswith("valid_loss")]:
                jsondict[lossname] = self.valid_losses[lossname]
            jsondict['epochs'] = self.epochs
            json.dump(jsondict, f)
