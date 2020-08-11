import os
import click
from os.path import join
from torchsummary import summary
from torch.utils.data import DataLoader
from molecules.ml.datasets import PointCloudDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import LossCallback, CheckpointCallback, Embedding3dCallback, PointCloud3dCallback
from molecules.ml.unsupervised.point_autoencoder import AAE3d, AAE3dHyperparams


def parse_dict(ctx, param, value):
    if value is not None:
        token = value.split(",")
        result = {}
        for item in token:
            k, v = item.split("=")
            result[k] = v
        return result

@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-dn', '--dataset_name', required=True, type=str,
              help='Name of the dataset in the HDF5 file.')

@click.option('-rn', '--rmsd_name', required=True, type=str,
              help='Name of the RMSD data in the HDF5 file.')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True, type=str,
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

@click.option('-opt', '--optimizer', callback=parse_dict,
              help='Optimizer parameters')

@click.option('-d', '--latent_dim', default=256, type=int,
              help='Number of dimensions in latent space')

@click.option('-lw', '--loss_weights', callback=parse_dict,
              help='Loss parameters')

@click.option('-ndw', '--num_data_workers', default=1, type=int,
              help='Number of workers for data loading')

@click.option('-wp', '--wandb_project_name', default=None, type=str,
              help='Project name for wandb logging')

def main(input_path, dataset_name, rmsd_name, out_path, model_id,
         num_points, num_features,
         encoder_gpu, generator_gpu, discriminator_gpu,
         epochs, batch_size, optimizer, latent_dim,
         loss_weights, num_data_workers,
         wandb_project_name):
    """Example for training Fs-peptide with AAE3d."""

    # HP
    # model
    aae_hparams = {
        "num_features": num_features,
        "latent_dim": latent_dim,
        "noise_std": 0.2,
        "lambda_rec": float(loss_weights["lambda_rec"]),
        "lambda_gp": float(loss_weights["lambda_gp"]),
        "lambda_adv": float(loss_weights["lambda_adv"])
        }
    hparams = AAE3dHyperparams(**aae_hparams)
    
    # optimizers
    optimizer_hparams = OptimizerHyperparams(name = optimizer["name"],
                                             hparams={'lr':float(optimizer["lr"])})

    aae = AAE3d(num_points, num_features, batch_size, hparams, optimizer_hparams,
              gpu=(encoder_gpu, generator_gpu, discriminator_gpu))

    # Diplay model
    print(aae)
    
    # Only print summary when encoder_gpu is None or 0
    summary(aae.model, (3 + num_features, num_points))

    # Load training and validation data
    train_dataset = PointCloudDataset(input_path,
                                      dataset_name,
                                      rmsd_name,
                                      num_points,
                                      num_features,
                                      split='train',
                                      normalize='box',
                                      cms_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers = num_data_workers)

    valid_dataset = PointCloudDataset(input_path,
                                      dataset_name,
                                      rmsd_name,
                                      num_points,
                                      num_features,
                                      split='valid',
                                      normalize='box',
                                      cms_transform=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True, num_workers = num_data_workers)

    print(f"Having {len(train_dataset)} training and {len(valid_dataset)} validation samples.")
    
    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_id}')

    # do we want wandb
    wandb_config = None
    if wandb_project_name is not None:
        import wandb
        wandb.init(project = wandb_project_name,
                   name = model_id,
                   id = model_id,
                   resume = False)
        wandb_config = wandb.config

        # log HP
        wandb_config.num_points = num_points
        wandb_config.num_features = num_features
        wandb_config.latent_dim = latent_dim
        wandb_config.lambda_rec = hparams.lambda_rec
        wandb_config.lambda_gp = hparams.lambda_gp
        # noise
        wandb_config.noise_std = hparams.noise_std

        # optimizer
        wandb_config.optimizer_name = optimizer_hparams.name
        for param in optimizer_hparams.hparams:
            wandb_config["optimizer_" + param] = optimizer_hparams.hparams[param]
        
        # watch model
        wandb.watch(aae.model)
    
    # Optional callbacks
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    loss_callback = LossCallback(join(model_path, 'loss.json'), writer, wandb_config)
    checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'))    
    embedding_callback = Embedding3dCallback(out_dir = join(model_path, 'embedddings'),
                                             path = input_path,
                                             rmsd_name = rmsd_name,
                                             projection_type = "3d_project",
                                             sample_interval = len(valid_dataset) // 1000,
                                             writer = writer,
                                             wandb_config = wandb_config)

    # Train model with callbacks
    aae.train(train_loader, valid_loader, epochs,
              callbacks = [loss_callback, checkpoint_callback, embedding_callback])

    # Save loss history to disk.
    loss_callback.save(join(model_path, 'loss.json'))

    # Save hparams to disk
    hparams.save(join(model_path, 'model-hparams.json'))
    optimizer_hparams.save(join(model_path, 'optimizer-hparams.json'))

    # Save final model weights to disk
    aae.save_weights(join(model_path, 'encoder-weights.pt'),
                     join(model_path, 'generator-weights.pt'),
                     join(model_path, 'discriminator-weights.pt'))

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
