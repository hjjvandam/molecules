from molecules.ml.hyperparams import Hyperparams
from torch import optim

class OptimizerHyperparams(Hyperparams):
    def __init__(self, name, hparams={}):
        """
        Parameters
        ----------
        name : str
            Name of Pytorch optimizer

        hparams : dict
            Dictionary of parameters to be passed to optimizer.
            If none are passed, uses default.

        """
        self.name = name
        self.hparams = hparams

        super().__init__()

    def validate(self):
        names = {'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam',
                 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD'}
        if self.name not in names:
            raise Exception(f'Invalid optimizer name: {self.name}.\n'
                            f'Please choose from {names}.\nSee PyTorch docs.')

def get_optimizer(model, hparams):
    """
    Parameters
    ----------
    model_parameters : torch.nn.Module
        PyTorch model

    hparams : OptimizerHyperparams
        Hyperparameters specifying the optimizer

    """

    try:

        if hparams.name is 'Adadelta':
            return optim.Adadelta(model.parameters(), **hparams.hparams)

        elif hparams.name is 'Adagrad':
            return optim.Adagrad(model.parameters(), **hparams.hparams)

        elif hparams.name is 'Adam':
            return optim.Adam(model.parameters(), **hparams.hparams)

        elif hparams.name is 'AdamW':
            return optim.AdamW(model.parameters(), **hparams.hparams)

        elif hparams.name is 'SparseAdam':
            return optim.SparseAdam(model.parameters(), **hparams.hparams)

        elif hparams.name is 'Adamax':
            return optim.Adamax(model.parameters(), **hparams.hparams)

        elif hparams.name is 'ASGD':
            return optim.ASGD(model.parameters(), **hparams.hparams)

        elif hparams.name is 'LBFGS':
            return optim.LBFGS(model.parameters(), **hparams.hparams)

        elif hparams.name is 'RMSprop':
            return optim.RMSprop(model.parameters(), **hparams.hparams)

        elif hparams.name is 'Rprop':
            return optim.Rprop(model.parameters(), **hparams.hparams)

        elif hparams.name is 'SGD':
            return optim.SGD(model.parameters(), **hparams.hparams)

    except TypeError as e:
        raise Exception(f'Invalid parameter in hparams: {hparams.hparams}'
                        f' for optimizer {hparams.name}.\nSee PyTorch docs.')
