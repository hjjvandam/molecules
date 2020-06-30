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

#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
