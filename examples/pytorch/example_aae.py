import os
import click
from os.path import join
from torchsummary import summary
from torch.utils.data import DataLoader
#from molecules.ml.datasets import Point3dDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import LossCallback, CheckpointCallback, EmbeddingCallback
from molecules.ml.unsupervised.aae import AAE3d, AAE3dHyperparams


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True,
              help='Model ID in for file naming')

@click.option('-np', '--num_points', required=True, type=int,
              help='number of input points')

@click.option('-nf', '--num_features', default=1, type=int,
              help='number of features per point in addition to 3D coordinates')

@click.option('-E', '--encoder_gpu', default=None, type=int,
              help='Encoder GPU id')

@click.option('-G', '--generator_gpu', default=None, type=int,
              help='Generator GPU id')

@click.option('-D', '--discriminator_gpu', default=None, type=int,
              help='Discriminator GPU id')

@click.option('-e', '--epochs', default=10, type=int,
              help='Number of epochs to train for')

@click.option('-b', '--batch_size', default=128, type=int,
              help='Batch size for training')

@click.option('-d', '--latent_dim', default=256, type=int,
              help='Number of dimensions in latent space')

@click.option('-lrec', '--lambda_rec', default=1., type=float,
              help='Reconstruction loss weight')

@click.option('-lgp', '--lambda_gp', default=10., type=float,
              help='Gradient penalty weight')

def main(input_path, out_path, model_id, num_points, num_features, encoder_gpu,
         generator_gpu, discriminator_gpu, epochs, batch_size, latent_dim, lambda_rec, lambda_gp):
    """Example for training Fs-peptide with AAE3d."""

    # HP
    # model
    aae_hparams = {
        "num_features": num_features,
        "latent_dim": latent_dim,
        "lambda_rec": lambda_rec,
        "lambda_gp": lambda_gp
        }
    hparams = AAE3dHyperparams(**aae_hparams)

    # optimizers
    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    aae = AAE3d(num_points, num_features, batch_size, hparams, optimizer_hparams,
              gpu=(encoder_gpu, generator_gpu, discriminator_gpu))

    # Diplay model
    print(aae)
    
    ## Only print summary when encoder_gpu is None or 0
    #summary(vae.model, input_shape)

    ## Load training and validation data
    #train_loader = DataLoader(ContactMapDataset(input_path,
    #                                            input_shape,
    #                                            split='train',
    #                                            sparse=sparse,
    #                                            gpu=encoder_gpu),
    #                          batch_size=batch_size, shuffle=True)
    #valid_loader = DataLoader(ContactMapDataset(input_path,
    #                                            input_shape,
    #                                            split='valid',
    #                                            sparse=sparse,
    #                                            gpu=encoder_gpu),
    #                          batch_size=batch_size, shuffle=True)

    ## For ease of training multiple models
    #model_path = join(out_path, f'model-{model_id}')

    ## Optional callbacks
    #from torch.utils.tensorboard import SummaryWriter
    #writer = SummaryWriter()
    #loss_callback = LossCallback(join(model_path, 'loss.json'), writer)
    #checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'))
    ##embedding_callback = EmbeddingCallback(input_path,
    ##                                       input_shape,
    ##                                       out_dir=join(model_path, 'embedddings'),
    ##                                       writer=writer)

    ## Train model with callbacks
    #vae.train(train_loader, valid_loader, epochs,
    #          callbacks=[loss_callback, checkpoint_callback])# embedding_callback])

    ## Save loss history to disk.
    #loss_callback.save(join(model_path, 'loss.json'))

    ## Save hparams to disk
    #hparams.save(join(model_path, 'model-hparams.json'))
    #optimizer_hparams.save(join(model_path, 'optimizer-hparams.json'))

    ## Save final model weights to disk
    #vae.save_weights(join(model_path, 'encoder-weights.pt'),
    #                 join(model_path, 'decoder-weights.pt'))

    # Output directory structure
    #  out_path
    # ├── model_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── decoder-weights.pt
    # │   ├── encoder-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.pkl
    # │   └── optimizer-hparams.pkl

if __name__ == '__main__':
    main()
