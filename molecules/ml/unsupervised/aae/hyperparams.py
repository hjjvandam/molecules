from molecules.ml.hyperparams import Hyperparams

class 3dAAEHyperparams(Hyperparams):
    def __init__(self,
                 num_features = 1,
                 encoder_filters=[64, 128, 256, 256, 512],
                 generator_filters=[64, 128, 512, 1024],
                 discriminator_filters=[512, 512, 128, 64],
                 latent_dim=256,
                 encoder_relu_slope = 0.,
                 generator_relu_slope = 0.,
                 discriminator_relu_slope = 0.;
                 use_encoder_bias = True,
                 use_generator_bias = True,
                 use_discriminator_bias = True,
                 output_activation='Sigmoid'):

        self.num_features = num_features
        self.encoder_filters = encoder_filters
        self.deecoder_filters = deecoder_filters
        self.latent_dim = latent_dim
        self.relu_slope = relu_slope
        self.use_encoder_bias = use_encoder_bias
        self.use_generator_bias =use_generator_bias
        self.use_discriminator_bias =use_discriminator_bias
        self.output_activation = output_activation

        # Placed after member vars are declared so that base class can validate
        super().__init__()

    def validate(self):
        return
