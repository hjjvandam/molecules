import pickle
from abc import ABCMeta, abstractmethod

class Hyperparams(metaclass=ABCMeta):
    """Abstract interface for defining hyperparameters"""
    def __init__(self):
        self.validate()

    def __repr__(self):
        return f'{self.__class__.__name__}\n' + \
        ''.join(f'{attr}: {value}\n' for attr, value in self.__dict__.items())

    @abstractmethod
    def validate(self):
        raise NotImplementedError('Must implement validate().')

    def save(self, path):
        """Write HyperParams object to disk."""
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def load(path):
        """Load HyperParams object from disk."""
        with open(path, 'rb') as file:
            hparams = pickle.load(file)
        hparams.validate()
        return hparams
