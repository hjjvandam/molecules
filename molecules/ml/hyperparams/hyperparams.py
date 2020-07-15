import json
from abc import ABCMeta, abstractmethod

class Hyperparams(metaclass=ABCMeta):
    """Abstract interface for defining hyperparameters"""
    def __init__(self):
        self.validate()
        if 'hparam_type' in self.__dict__:
            raise ValueError("'hparam_type' is not allowed to be used as "
                             "a member variable name since it is used "
                             "to save models to a file.")

    def __repr__(self):
        return f'{self.__class__.__name__}\n' + \
        ''.join(f'{attr}: {value}\n' for attr, value in self.__dict__.items())

    @abstractmethod
    def validate(self):
        raise NotImplementedError('Must implement validate().')

    def save(self, path):
        """Write HyperParams object to disk."""
        with open(path, 'w') as file:
            payload = self.__dict__
            payload['hparam_type'] = self.__class__.__name__
            json.dump(self.__dict__, file)

    def load(self, path):
        """Load HyperParams object from disk."""
        with open(path, 'r') as file:
            hparams = json.load(file)
            hparams.pop('hparam_type')
            self.__dict__ = hparams
        self.validate()
        return self
