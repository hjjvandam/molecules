
class Callback:
    def __init__(self): pass
    def on_train_begin(self, logs): pass
    def on_train_end(self, logs): pass
    def on_epoch_begin(self, epoch, logs): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_batch_begin(self, batch, epoch, logs): pass
    def on_batch_end(self, batch, epoch, logs): pass


class LossHistory(Callback):
    def on_train_begin(self, logs):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter()

        # self.train_losses = []
        # self.valid_losses = []
        # self.steps = []

    def on_epoch_end(self, epoch, logs):

        self.writer.add_scalar('epoch training loss',
                               logs['train_loss'],
                               logs['global_step'])
        self.writer.add_scalar('epoch validation loss',
                               logs['valid_loss'],
                               logs['global_step'])


        # self.train_losses.append(logs['train_loss'])
        # self.valid_losses.append(logs['valid_loss'])
        # self.steps.append(logs['global_step'])
