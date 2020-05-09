from molecules.ml.hyperparams.hyperparams import Hyperparams

class OptimizerHyperparams(HyperParams):
	def __init__(self):
		super().__init__()

	def validate(self):
		pass

	def get_optimizer(self, model_parameters):
		pass