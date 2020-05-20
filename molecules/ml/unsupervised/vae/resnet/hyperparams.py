from molecules.ml.hyperparams import Hyperparams

# TODO: may want seperate hyperparams classes for encoder/decoder

class ResnetVAEHyperparams(Hyperparams):
    def __init__(self):

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        pass