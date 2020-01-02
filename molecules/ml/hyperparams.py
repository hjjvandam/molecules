import pickle
from abc import ABCMeta, abstractmethod

class HyperParams(metaclass=ABCMeta):
    """Abstract interface for defining hyperparameters"""
    def __init__(self):
        pass

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
