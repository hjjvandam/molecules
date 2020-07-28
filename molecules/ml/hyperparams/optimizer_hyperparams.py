from molecules.ml.hyperparams import Hyperparams
from torch import optim

class OptimizerHyperparams(Hyperparams):
    def __init__(self, name='RMSprop', hparams={}):
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

    # TODO: could be useful to define bounds for each type of hparams
    #       to assist in the bayesian optization.

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

        if hparams.name == 'Adadelta':
            return optim.Adadelta(model.parameters(), **hparams.hparams)

        elif hparams.name == 'Adagrad':
            return optim.Adagrad(model.parameters(), **hparams.hparams)

        elif hparams.name == 'Adam':
            return optim.Adam(model.parameters(), **hparams.hparams)

        elif hparams.name == 'AdamW':
            return optim.AdamW(model.parameters(), **hparams.hparams)

        elif hparams.name == 'SparseAdam':
            return optim.SparseAdam(model.parameters(), **hparams.hparams)

        elif hparams.name == 'Adamax':
            return optim.Adamax(model.parameters(), **hparams.hparams)

        elif hparams.name == 'ASGD':
            return optim.ASGD(model.parameters(), **hparams.hparams)

        elif hparams.name == 'LBFGS':
            return optim.LBFGS(model.parameters(), **hparams.hparams)

        elif hparams.name == 'RMSprop':
            return optim.RMSprop(model.parameters(), **hparams.hparams)

        elif hparams.name == 'Rprop':
            return optim.Rprop(model.parameters(), **hparams.hparams)

        elif hparams.name == 'SGD':
            return optim.SGD(model.parameters(), **hparams.hparams)

    except TypeError as e:
        raise Exception(f'Invalid parameter in hparams: {hparams.hparams}'
                        f' for optimizer {hparams.name}.\nSee PyTorch docs.')
