import os
from os.path import join
import json
import click
import numpy as np
from glob import glob
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from molecules.ml.unsupervised.cluster import optics_clustering
from molecules.ml.unsupervised import (VAE, EncoderConvolution2D, 
                                       DecoderConvolution2D,
                                       EncoderHyperparams,
                                       DecoderHyperparams)

def parse_list(ctx, param, value):
    """Parse click CLI list "item1,item2,item3..."."""
    if value is not None:
        return value.split(',')

def model_selection(model_paths, num_select=1):
    """
    Parameters
    ----------
    model_paths : list
        list of paths pointing to a directory structured like below.

    num_select : int
        Number of best models to select. NOT IMPLEMENTED

    Note:
    model_path directory structure
        ├── model_path
        │   ├── checkpoint
        │   │   ├── epoch-1-20200606-125334.pt
        │   │   └── epoch-2-20200606-125338.pt
        │   ├── decoder-weights.pt
        │   ├── encoder-weights.pt
        │   ├── loss.json
        │   ├── model-hparams.json
        │   └── optimizer-hparams.json
        |   |__ wandb_cache/
    """

    # Best model checkpoint file for each HPO run and
    # associated validation loss 
    best_checkpoints, best_valid_losses, hparams = [], [], []

    # Loop over all model paths in hyperparameter optimization search.
    for model_path in model_paths:
        
        with open(join(model_path, 'loss.json')) as loss_file:
            loss_log = json.load(loss_file)
        valid_loss, epochs = loss_log['valid_loss'], loss_log['epochs']
        # Need to index into epoch, since run could have been resumed from
        # a checkpoint, meaning that epochs does not start at 1.
        best_ind = np.argmin(valid_loss)
        best_valid_loss = valid_loss[best_ind]
        best_epoch = epochs[best_ind]

        # Format: epoch-1-20200606-125334.pt
        for checkpoint in glob(join(model_path, 'checkpoint')):
            if checkpoint.split('-')[1] == best_epoch:
                best_checkpoint = checkpoint
                break

        # Colect model checkpoints associated with minimal validation loss
        best_checkpoints.append(best_checkpoint)
        best_valid_losses.append(best_valid_loss)
        hparams.append(join(model_path, 'model-hparams.json'))

    best_model = np.argmax(best_valid_losses)
    best_checkpoint = best_checkpoints[best_model]
    best_hparams = hparams[best_model]

    return best_hparams, best_checkpoint


def generate_embeddings(hparams_path, checkpoint_path, dim1, dim2,
                        device, input_path, cm_format, batch_size):

    # TODO: make model_type str and handle variable to correct encoder and Dataset

    # Initialize encoder model
    input_shape = (dim1, dim1)
    hparams = ResnetVAEHyperparams.load(hparams_path)
    encoder = ResnetEncoder(input_shape, hparams)

    # Put encoder on specified CPU/GPU
    encoder.to(device)

    # Load encoder checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # Clean checkpoint up since it contains encoder, decoder and optimizer weights
    del checkpoint; import gc; gc.collect()


    dataset = ContactMapDataset(input_path,
                                'contact_map',
                                'rmsd', 'fnc',
                                input_shape,
                                split='train',
                                split_ptc=1.,
                                cm_format=cm_format)
    
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0)

    # Collect embeddings and associated index into simulation trajectory
    embeddings, indices = [], []
    for data, rmsd, fnc, index in data_loader:
        embeddings.append(encoder.encode(data).cpu().numpy())
        indices.append(index)

    embeddings = np.concatenate(embeddings)
    indices = np.concatenate(indices)

    return embeddings, indices

def perform_clustering(eps_path, encoder_weight_path, cm_embeddings, min_samples, eps):
    # TODO: if we decide on OPTICS clustering, then remove all eps code


    # If previous epsilon values have been calculated, load the record from disk.
    # eps_record stores a dictionary from cvae_weight path which uniquely identifies
    # a model, to the epsilon value previously calculated for that model.
    with open(eps_path) as file:
        try:
            eps_record = json.load(file)
        except json.decoder.JSONDecodeError:
            eps_record = {}

    best_eps = eps_record.get(encoder_weight_path)

    # If eps_record contains previous eps value then use it, otherwise use default.
    if best_eps:
        eps = best_eps

    eps, outlier_inds, labels = dbscan_clustering(cm_embeddings, eps, min_samples)
    eps_record[encoder_weight_path] = eps

    # Save the eps for next round of the pipeline
    with open(eps_path, 'w') as file:
        json.dump(eps_record, file)

    return outlier_inds, labels

def write_rewarded_pdbs(rewarded_inds, sim_path, pdb_out_path):
    # Get list of simulation trajectory files (Assume all are equal length (ns))
    traj_fnames = sorted(glob(os.path.join(sim_path, 'output-*.dcd')))

    # Get list of simulation PDB files
    pdb_fnames = sorted(glob(os.path.join(sim_path, 'input-*.pdb')))

    # Get total number of simulations
    sim_count = len(traj_fnames)

    # Get simulation indices and frame number coresponding to outliers
    reward_locs = map(lambda outlier: divmod(outlier, sim_count), rewarded_inds)

    # For documentation on mda.Writer methods see:
    #   https://www.mdanalysis.org/mdanalysis/documentation_pages/coordinates/PDB.html
    #   https://www.mdanalysis.org/mdanalysis/_modules/MDAnalysis/coordinates/PDB.html#PDBWriter._update_frame

    for frame, sim_id in reward_locs:
        pdb_fname = os.path.join(pdb_out_path, f'outlier-sim-{sim_id}-frame-{frame}.pdb')
        u = mda.Universe(pdb_fnames[sim_id], traj_fnames[sim_id])
        with mda.Writer(pdb_fname) as writer:
            # Write a single coordinate set to a PDB file
            writer._update_frame(u)
            writer._write_timestep(u.trajectory[frame])


@click.command()
@click.option('-s', '--sim_path', required=True,
              type=click.Path(exists=True),
              help='OpenMM simulation path containing *.dcd and *.pdb files')

@click.option('-p', '--pdb_out_path', required=True,
              type=click.Path(exists=True),
              help='Path to folder to write output PDB files to.')

@click.option('-d', '--data_path', required=True,
              type=click.Path(exists=True),
              help='Preprocessed data h5 file path')

@click.option('-m', '--model_paths', required=True,
              callback=parse_list,
              help='List of trained model paths "path1,path2,...".')

@click.option('-M', '--min_samples', default=10, type=int,
              help='Value of min_samples in the OPTICS algorithm')

@click.option('-D', '--device', default='cpu',
              help='PyTorch formatted device str. cpu, cuda, cuda:0, etc')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Format of contact map files. Options ' \
                   '[full, sparse-concat, sparse-rowcol]')

@click.option('-b', '--batch_size', default=128, type=int,
              help='Batch size for forward pass. Limited by available ' \
                   ' memory and size of input example. Does not affect output.' \
                   ' The larger the batch size, the faster the computation.')

def main(sim_path, pdb_out_path, data_path, model_paths, min_samples,
         device, dim1, dim2, cm_format, batch_size):
    
    best_hparams, best_checkpoint = model_selection(model_paths, num_select=1)

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings, indices = generate_embeddings(best_hparams, best_checkpoint, dim1, dim2,
                                              device, data_path, cm_format, batch_size)

    # Performs DBSCAN clustering on embeddings
    #outlier_inds, labels = perform_clustering(eps_path, best_checkpoint,
    #                                          embeddings, min_samples, eps)

    # Performs OPTICS clustering on embeddings
    outlier_inds, labels = optics_clustering(embeddings, min_samples=min_samples)

    # Map shuffled indices back to in-order MD frames
    simulation_inds = indices[outlier_inds]

    # Write rewarded PDB files to shared path
    write_rewarded_pdbs(simulation_inds, sim_path, pdb_out_path)

if __name__ == '__main__':
    main()