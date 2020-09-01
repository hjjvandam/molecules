class Callback:
    def __init__(self, mpi_comm=None):
        """
        Parameters
        ----------
        mpi_com : mpi communicator optional
        """
        self.comm = mpi_comm
        self.is_eval_node = True
        if (self.comm is not None) and (self.comm.Get_rank() != 0):
            self.is_eval_node = False

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
