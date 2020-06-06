import click
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from molecules.utils import open_h5
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams, ResnetVAEHyperparams

import torch
import numpy as np
# TODO: this class is temporary. Will eventually be added to the package.
class ContactMap(Dataset):
    def __init__(self, path, split_ptc=0.8, split='train', squeeze=False):
        with open_h5(path) as input_file:
            # Access contact matrix data from h5 file
            data = np.array(input_file['contact_maps'])

        # 80-20 train validation split index
        split_ind = int(split_ptc * len(data))

        if split == 'train':
            self.data = data[:split_ind]
        elif split == 'valid':
            self.data = data[split_ind:]
        else:
            raise ValueError(f'Parameter split={split} is invalid.')

        # TODO: this reshape code may not be the best solution. revisit
        num_residues = self.data.shape[2]
        assert num_residues == 22

        if squeeze:
            shape = (-1, num_residues, num_residues)
        else:
            shape = (-1, 1, num_residues, num_residues)

        self.data = torch.from_numpy(self.data.reshape(shape)).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True,
              help='Model ID in for file naming')

@click.option('-g', '--gpu', default=0, type=int,
              help='GPU id')

@click.option('-e', '--epochs', default=100, type=int,
              help='Number of epochs to train for')

@click.option('-b', '--batch_size', default=512, type=int,
              help='Batch size for training')

@click.option('-t', '--model_type', default='resnet',
              help='Model architecture option: [resnet, symmetric]')

@click.option('-d', '--latent_dim', default=10, type=int,
              help='Number of dimensions in latent space')

def main(input_path, out_path, model_id, gpu, epochs, batch_size, model_type, latent_dim):
    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    if model_type is 'symmetric':

        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': latent_dim}

        input_shape = (1, 22, 22)
        squeeze = False
        hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type is 'resnet':
        input_shape = (22, 22)
        squeeze = True
        hparams = ResnetVAEHyperparams(input_shape, latent_dim=11)

    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    vae = VAE(input_shape, hparams, optimizer_hparams)

    # Diplay model
    print(vae)
    summary(vae.model, input_shape)


    train_loader = DataLoader(ContactMap(input_path, split='train', squeeze=squeeze),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ContactMap(input_path, split='valid', squeeze=squeeze),
                             batch_size=batch_size, shuffle=True)


    vae.train(train_loader, test_loader, epochs)


if __name__ == '__main__':
    main()

