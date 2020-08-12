import os
import click
from os.path import join
from torchsummary import summary
from torch.utils.data import DataLoader
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import LossCallback, CheckpointCallback, EmbeddingCallback
from molecules.ml.unsupervised.vae import VAE, SymmetricVAEHyperparams, ResnetVAEHyperparams


@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', '--out', 'out_path', required=True,
              type=click.Path(exists=True),
              help='Output directory for model data')

@click.option('-m', '--model_id', required=True,
              help='Model ID in for file naming')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-c', '--checkpoint',
             type=click.Path(exists=True),
             help='Model checkpoint file to resume training. ' \
                  'Checkpoint files saved as .pt by CheckpointCallback.')

@click.option('-s', '--sparse', is_flag=True,
              help='Specifiy whether input matrices are sparse format')

@click.option('-E', '--encoder_gpu', default=None, type=int,
              help='Encoder GPU id')

@click.option('-D', '--decoder_gpu', default=None, type=int,
              help='Decoder GPU id')

@click.option('-e', '--epochs', default=10, type=int,
              help='Number of epochs to train for')

@click.option('-b', '--batch_size', default=128, type=int,
              help='Batch size for training')

@click.option('-t', '--model_type', default='resnet',
              help='Model architecture option: [resnet, symmetric]')

@click.option('-d', '--latent_dim', default=10, type=int,
              help='Number of dimensions in latent space')

@click.option('-S', '--sample_interval', default=20, type=int,
              help="For embedding plots. Plots every sample_interval'th point")

def main(input_path, out_path, checkpoint, model_id, dim1, dim2, encoder_gpu, sparse,
         decoder_gpu, epochs, batch_size, model_type, latent_dim,
         sample_interval):
    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    assert model_type in ['symmetric', 'resnet']

    # Add incoming devices to visible devices, or default to gpu:0
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    if None not in (encoder_gpu, decoder_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join({str(encoder_gpu),
                                                       str(decoder_gpu)})
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    print('CUDA devices: ', os.environ['CUDA_VISIBLE_DEVICES'])
    # Note: See SymmetricVAEHyperparams, ResnetVAEHyperparams class definitions
    #       for hyperparameter options. 

    if model_type == 'symmetric':
        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': latent_dim}

        input_shape = (1, dim1, dim2)
        hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type == 'resnet':

        resnet_hparams = {'max_len': dim1,
                          'nchars': dim2,
                          'latent_dim': latent_dim,
                          'dec_filters': dim1}

        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams(**resnet_hparams)

    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu=(encoder_gpu, decoder_gpu))

    # Diplay model
    print(vae)
    # Only print summary when encoder_gpu is None or 0
    summary(vae.model, input_shape)

    # Load training and validation data
    train_loader = DataLoader(ContactMapDataset(input_path,
                                                input_shape,
                                                split='train',
                                                sparse=sparse,
                                                gpu=encoder_gpu),
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ContactMapDataset(input_path,
                                                input_shape,
                                                split='valid',
                                                sparse=sparse,
                                                gpu=encoder_gpu),
                              batch_size=batch_size, shuffle=True)

    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_id}')

    # Optional callbacks
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    loss_callback = LossCallback(join(model_path, 'loss.json'), writer)
    checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'))
    embedding_callback = EmbeddingCallback(input_path,
                                           join(model_path, 'embedddings'),
                                           input_shape,
                                           sparse=sparse,
                                           writer=writer,
                                           sample_interval=sample_interval,
                                           batch_size=batch_size,
                                           gpu=encoder_gpu)

    # Train model with callbacks
    vae.train(train_loader, valid_loader, epochs,
              checkpoint=checkpoint if checkpoint is not None else '',
              callbacks=[loss_callback, checkpoint_callback, embedding_callback])

    # Save loss history to disk.
    loss_callback.save(join(model_path, 'loss.json'))

    # Save hparams to disk
    hparams.save(join(model_path, 'model-hparams.json'))
    optimizer_hparams.save(join(model_path, 'optimizer-hparams.json'))

    # Save final model weights to disk
    vae.save_weights(join(model_path, 'encoder-weights.pt'),
                     join(model_path, 'decoder-weights.pt'))

    # Output directory structure
    #  out_path
    # ├── model_path
    # │   ├── checkpoint
    # │   │   ├── epoch-1-20200606-125334.pt
    # │   │   └── epoch-2-20200606-125338.pt
    # │   ├── decoder-weights.pt
    # │   ├── encoder-weights.pt
    # │   ├── loss.json
    # │   ├── model-hparams.json
    # │   └── optimizer-hparams.json

if __name__ == '__main__':
    main()