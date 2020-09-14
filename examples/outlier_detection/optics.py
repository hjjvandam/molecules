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
    best_checkpoints, best_valid_losses, hparams, opt_hparams = [], [], [], []

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
        opt_hparams.append(join(model_path, 'optimizer-hparams.json'))

    best_model = np.argmax(best_valid_losses)
    best_checkpoint = best_checkpoints[best_model]
    best_hparams = hparams[best_model]
    best_opt_hparams = opt_hparams[best_model]

    return best_hparams, best_opt_hparams, best_checkpoint


def generate_embeddings(hparams_path, opt_hparams_path, checkpoint_path, input_shape,
                        encoder_gpu, decoder_gpu, input_path, cm_format, batch_size):
    hparams = ResnetVAEHyperparams.load(hparams_path)
    optimizer_hparams = OptimizerHyperparams.load(opt_hparams_path)

    vae = VAE(input_shape, hparams, optimizer_hparams,
              gpu=(encoder_gpu, decoder_gpu), enable_amp=False)

    vae._load_checkpoint(checkpoint_path)

    dataset = ContactMapDataset(input_path,
                                'contact_map',
                                'rmsd', 'fnc',
                                input_shape,
                                split='train',
                                split_ptc=1.,
                                shuffle=False,
                                cm_format=cm_format)
    
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=0)

    embeddings = []
    for data in data_loader:
        embeddings.append(vae.encode(data).cpu().numpy())

    embeddings = np.concatenate(embeddings)

    return embeddings

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

def write_rewarded_pdbs(rewarded_inds, sim_path, shared_path):
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
        pdb_fname = os.path.join(shared_path, f'outlier-{sim_id}-{frame}.pdb')
        u = mda.Universe(pdb_fnames[sim_id], traj_fnames[sim_id])
        with mda.Writer(pdb_fname) as writer:
            # Write a single coordinate set to a PDB file
            writer._update_frame(u)
            writer._write_timestep(u.trajectory[frame])


@click.command()
@click.option('-i', '--sim_path', required=True,
              type=click.Path(exists=True),
              help='OpenMM simulation path containing *.dcd and *.pdb files')

@click.option('-s', '--shared_path', required=True,
              type=click.Path(exists=True),
              help='Path to folder shared between outlier and MD stages.')

@click.option('-d', '--cm_path', required=True,
              type=click.Path(exists=True),
              help='Preprocessed cvae-input h5 file path')

@click.option('-c', '--cvae_path', required=True,
              type=click.Path(exists=True),
              help='CVAE model directory path')

@click.option('-m', '--min_samples', default=10, type=int,
              callback=validate_positive,
              help='Value of min_samples in the DBSCAN algorithm')

@click.option('-g', '--gpu', default=0, type=int,
              callback=validate_positive,
              help='GPU id')

def main(sim_path, shared_path, cm_path, cvae_path, min_samples, gpu):
    
    best_hparams, best_opt_hparams, best_checkpoint = model_selection(model_paths, num_select=1)

    # Generate embeddings for all contact matrices produced during MD stage
    cm_embeddings = generate_embeddings(best_hparams, best_opt_hparams, best_checkpoint, ...)

    # Performs DBSCAN clustering on embeddings
    #outlier_inds, labels = perform_clustering(eps_path, encoder_weight_path,
    #                                          cm_embeddings, min_samples, eps)

    # Performs OPTICS clustering on embeddings
    outlier_inds, labels = optics_clustering(cm_embeddings, min_samples)


    # Write rewarded PDB files to shared path
    write_rewarded_pdbs(outlier_inds, sim_path, shared_path)

if __name__ == '__main__':
    main()