import json
from .callback import Callback

class LossCallback(Callback):
    def __init__(self, path, writer = None):
        """
        Parameters
        ----------
        path : str
            path to save loss history to

        writer : torch.utils.tensorboard.SummaryWriter
        """
        self.writer = writer
        self.path = path

    def on_train_begin(self, logs):
        self.epochs = []
        self.train_losses = {}
        self.valid_losses = {}

    def on_epoch_end(self, epoch, logs):

        if self.writer is not None:
            for lossname in [x for x in logs if x.startswith("train_loss")]:
                self.writer.add_scalar('epoch ' + lossname,
                                       logs[lossname],
                                       logs['global_step'])

            for lossname in [x for x in logs if x.startswith("validation_loss")]:
                self.writer.add_scalar('epoch ' + lossname,
                                       logs[lossname],
                                       logs['global_step'])

        self.epochs.append(epoch)
        for lossname in [x for x in logs if x.startswith("train_loss")]:
            if lossname in self.train_losses:
                self.train_losses[lossname].append(logs[lossname])
            else:
                self.train_losses[lossname] = [logs[lossname]]
                
        for lossname in [x for x in logs if x.startswith("valid_loss")]:
            if lossname in self.valid_losses:
                self.valid_losses[lossname].append(logs[lossname])
            else:
                self.valid_losses[lossname] = [logs[lossname]]

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
            for lossname in [x for x in self.train_losses if x.startswith("validation_loss")]:
                jsondict[lossname] = self.valid_losses[lossname]
            jsondict['epochs'] = self.epochs
            json.dump(jsondict, f)
