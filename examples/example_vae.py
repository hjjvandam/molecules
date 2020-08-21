import os
import click
from os.path import join

# torch stuff
from torchsummary import summary
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

# mpi4py
import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

# molecules stuff
from molecules.ml.datasets import ContactMapDataset
from molecules.ml.hyperparams import OptimizerHyperparams
from molecules.ml.callbacks import (LossCallback, CheckpointCallback,
                                    Embedding2dCallback, Embedding3dCallback)
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

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Format of contact map files. Options ' \
                   '[full, sparse-concat, sparse-rowcol]')

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

@click.option('-wp', '--wandb_project_name', default=None, type=str,
              help='Project name for wandb logging')

@click.option('--local_rank', default=None, type=int,
              help='Local rank on the machine, required for DDP')

@click.option('-a', '--amp', is_flag=True,
              help='Specify if we want to enable automatic mixed precision (AMP)')

@click.option('--distributed', is_flag=True,
              help='Enable distributed training')

def main(input_path, out_path, checkpoint, model_id, dim1, dim2, cm_format, encoder_gpu,
         decoder_gpu, epochs, batch_size, model_type, latent_dim,
         sample_interval, wandb_project_name, local_rank, amp, distributed):

    """Example for training Fs-peptide with either Symmetric or Resnet VAE."""

    # do some scaffolding for DDP
    comm_rank = 0
    comm_size = 1
    comm_local_rank = 0
    comm = None
    if distributed and dist.is_available():
        # init mpi4py:
        MPI.Init_thread()

        # get communicator: duplicate from comm world
        comm = MPI.COMM_WORLD.Dup()

        # now match ranks between the mpi comm and the nccl comm
        os.environ["WORLD_SIZE"] = str(comm.Get_size())
        os.environ["RANK"] = str(comm.Get_rank())

        # init torch distributed
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        comm_rank = dist.get_rank()
        comm_size = dist.get_world_size()
        if local_rank is not None:
            comm_local_rank = local_rank
        else:
            comm_local_rank = int(os.getenv("LOCAL_RANK", 0))


    assert model_type in ['symmetric', 'resnet']

    # Note: See SymmetricVAEHyperparams, ResnetVAEHyperparams class definitions
    #       for hyperparameter options. 

    if model_type == 'symmetric':
        # Optimal Fs-peptide params
        fs_peptide_hparams ={'filters': [100, 100, 100, 100],
                             'kernels': [5, 5, 5, 5],
                             'strides': [1, 2, 1, 1],
                             'affine_widths': [64],
                             'affine_dropouts': [0],
                             'latent_dim': latent_dim,
                             'output_activation': 'None'}

        input_shape = (1, dim1, dim2)
        hparams = SymmetricVAEHyperparams(**fs_peptide_hparams)

    elif model_type == 'resnet':

        resnet_hparams = {'max_len': dim1,
                          'nchars': dim2,
                          'latent_dim': latent_dim,
                          'dec_filters': dim1,
                          'output_activation': 'None'}

        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams(**resnet_hparams)

    optimizer_hparams = OptimizerHyperparams(name='RMSprop', hparams={'lr':0.00001})

    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu=(encoder_gpu, decoder_gpu), enable_amp = amp)

    if comm_size > 1:
        if (encoder_gpu == decoder_gpu):
            devid = torch.device(f'cuda:{encoder_gpu}')
            vae.model = DDP(vae.model, device_ids = [devid], output_device = devid)
        else:
            vae.model = DDP(vae.model, device_ids = None, output_device = None)

    if comm_rank == 0:
        # Diplay model
        print(vae)
        # Only print summary when encoder_gpu is None or 0
        #summary(vae.model, input_shape)

    # Load training and validation data
    # training
    train_dataset = ContactMapDataset(input_path,
                                      'contact_map',
                                      'rmsd',
                                      input_shape,
                                      split='train',
                                      cm_format=cm_format)

    # split across nodes
    if comm_size > 1:
        chunksize = len(train_dataset) // comm_size
        train_dataset = Subset(train_dataset,
                               list(range(chunksize * comm_rank, chunksize * (comm_rank + 1))))
    
    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size,
                              drop_last = True,
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 1)

    # validation
    valid_dataset = ContactMapDataset(input_path,
                                      'contact_map',
                                      'rmsd',
                                      input_shape,
                                      split='valid',
                                      cm_format=cm_format)

    # split across nodes
    if comm_size > 1:
        chunksize = len(valid_dataset) // comm_size
        valid_dataset = Subset(valid_dataset,
                               list(range(chunksize * comm_rank, chunksize * (comm_rank + 1))))
    
    valid_loader = DataLoader(valid_dataset,
                              batch_size = batch_size,
                              drop_last = True,
                              shuffle = True,
                              pin_memory = True,
                              num_workers = 1)

    # For ease of training multiple models
    model_path = join(out_path, f'model-{model_id}')

    # do we want wandb
    wandb_config = None
    if (comm_rank == 0) and (wandb_project_name is not None):
        import wandb
        wandb.init(project = wandb_project_name,
                   name = model_id,
                   id = model_id,
                   resume = False)
        wandb_config = wandb.config
        
        # log HP
        wandb_config.dim1 = dim1
        wandb_config.dim2 = dim2
        wandb_config.latent_dim = latent_dim
        
        # optimizer
        wandb_config.optimizer_name = optimizer_hparams.name
        for param in optimizer_hparams.hparams:
            wandb_config['optimizer_' + param] = optimizer_hparams.hparams[param]
            
        # watch model
        wandb.watch(vae.model)
    
    # Optional callbacks
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter() if (comm_rank == 0) else None
    loss_callback = LossCallback(join(model_path, 'loss.json'),
                                 writer = writer,
                                 wandb_config = wandb_config,
                                 mpi_comm = comm)
    
    checkpoint_callback = CheckpointCallback(out_dir=join(model_path, 'checkpoint'),
                                             mpi_comm = comm)
    
   # embedding3d_callback = Embedding3dCallback(input_path,
   #                                            join(model_path, 'embedddings'),
   #                                            input_shape,
   #                                            cm_format = cm_format,
   #                                            writer = writer,
   #                                            sample_interval = sample_interval,
   #                                            batch_size = batch_size,
   #                                            mpi_comm = comm,
   #                                            device = torch.device(f'cuda:{encoder_gpu}'))

    embedding2d_callback = Embedding2dCallback(out_dir = join(model_path, 'embedddings'),
                                               path = input_path,
                                               rmsd_name = 'rmsd',
                                               projection_type = '3d_project',
                                               sample_interval = sample_interval,
                                               writer = writer,
                                               wandb_config = wandb_config,
                                               mpi_comm = comm)

    # Train model with callbacks
    callbacks = [loss_callback, checkpoint_callback,
                 embedding2d_callback] #, embedding3d_callback]

    # create model
    vae.train(train_loader, valid_loader, epochs,
              checkpoint=checkpoint if checkpoint is not None else '',
              callbacks=callbacks)

    if comm_rank == 0:
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
