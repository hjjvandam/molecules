class Callback:
    def __init__(self): pass
    def on_train_begin(self, logs): pass
    def on_train_end(self, logs): pass
    def on_epoch_begin(self, epoch, logs): pass
    def on_epoch_end(self, epoch, logs): pass
    def on_batch_begin(self, batch, epoch, logs): pass
    def on_batch_end(self, batch, epoch, logs): pass
    # validation
    def on_validation_begin(self, epoch, logs): pass
    def on_validation_end(self, epoch, logs): pass
    def on_validation_batch_begin(self, logs, **kwargs): pass
    def on_validation_batch_end(self, logs, **kwargs): pass
