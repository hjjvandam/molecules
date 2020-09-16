import os
import json
import click
import shutil
import torch
import numpy as np
from glob import glob
from os.path import join
#import MDAnalysis as mda
from torch.utils.data import DataLoader
from sklearn.neighbors import LocalOutlierFactor
from molecules.ml.unsupervised.cluster import optics_clustering

# plotting
import matplotlib.pyplot as plt

# Helper function for LocalOutlierFactor
def topk(a, k):
    """
    Parameters
    ----------
    a : np.ndarray
        array of dim (N,)
    k : int
        specifies which element to partition upon
    Returns
    -------
    np.ndarray of length k containing indices of input array a
    coresponding to the k smallest values in a.
    """
    return np.argpartition(a, k)[:k]

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
        best_epoch = str(epochs[best_ind])
        
        checkpoint_dir = join(model_path, 'checkpoint')
        # Format: epoch-1-20200606-125334.pt
        for checkpoint in os.listdir(checkpoint_dir):
            if checkpoint.split('-')[1] == best_epoch:
                best_checkpoint = checkpoint
                break

        # Colect model checkpoints associated with minimal validation loss
        best_checkpoints.append(join(checkpoint_dir, best_checkpoint))
        best_valid_losses.append(best_valid_loss)
        hparams.append(join(model_path, 'model-hparams.json'))

    best_model = np.argmin(best_valid_losses)
    best_checkpoint = best_checkpoints[best_model]
    best_hparams = hparams[best_model]

    return best_hparams, best_checkpoint


def generate_embeddings(model_type, hparams_path, checkpoint_path, dim1, dim2,
                        device, input_path, cm_format, batch_size):

    if model_type == 'vae-resnet':
        from molecules.ml.datasets import ContactMapDataset
        from molecules.ml.unsupervised.vae.resnet import ResnetVAEHyperparams, ResnetEncoder
        # Initialize encoder model
        input_shape = (dim1, dim1)
        hparams = ResnetVAEHyperparams().load(hparams_path)
        encoder = ResnetEncoder(input_shape, hparams)

        dataset = ContactMapDataset(input_path,
                                    'contact_map',
                                    'rmsd', 'fnc',
                                    input_shape,
                                    split='train',
                                    split_ptc=1.,
                                    cm_format=cm_format)
    elif model_type == 'aae':
       from molecules.ml.datasets import PointCloudDataset
       from molecules.ml.unsupervised.point_autoencoder import AAE3dHyperparams
       from molecules.ml.unsupervised.point_autoencoder.aae import Encoder

       hparams = AAE3dHyperparams().load(hparams_path)
       encoder = Encoder(dim1, hparams.num_features, hparams)

       dataset = PointCloudDataset(input_path,
                                   'point_cloud',
                                   'rmsd',
                                   'fnc',
                                   dim1,
                                   hparams.num_features,
                                   split='train',
                                   split_ptc=1.,
                                   normalize='box',
                                   cms_transform=False)


    # Put encoder on specified CPU/GPU
    encoder.to(device)

    # Load encoder checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # Clean checkpoint up since it contains encoder, decoder and optimizer weights
    del checkpoint; import gc; gc.collect()

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             drop_last=False,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=4)

    # Collect embeddings and associated index into simulation trajectory
    embeddings, indices = [], []
    for i, (data, rmsd, fnc, index) in enumerate(data_loader):
        data = data.to(device)
        embeddings.append(encoder.encode(data).cpu().numpy())
        indices.append(index)
        if i % 100 == 0:
            print(f'Batch {i}/{len(data_loader)}')

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


def local_outlier_factor(embeddings, n_outliers=500, plot_dir=None, **kwargs): 
    clf = LocalOutlierFactor(**kwargs)
    # Array with 1 if inlier, -1 if outlier
    clf.fit_predict(embeddings)

    # print the results
    if plot_dir is not None:
        # create directory
        os.makedirs(plot_dir, exist_ok=True)
        # plot
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.hist(clf.negative_outlier_factor_, bins='fd')
        plt.savefig(os.path.join(plot_dir, 'score_distribution.png'))
        plt.close()
    
    # Only sorts 1 element of negative_outlier_factors_, namely the element
    # that is position k in the sorted array. The elements above and below
    # the kth position are partitioned but not sorted. Returns the indices
    # of the elements of left hand side of the parition i.e. the top k.
    outlier_inds = topk(clf.negative_outlier_factor_, k=n_outliers)
    
    outlier_scores = clf.negative_outlier_factor_[outlier_inds]

    # Only sorts an array of size n_outliers
    sort_inds = np.argsort(outlier_scores)

    # Returns n_outlier best outliers sorted from best to worst
    return outlier_inds[sort_inds], outlier_scores[sort_inds]

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict): 
        if local_frame - traj_dict[key] < 0:            
            return local_frame, key
        else: 
            local_frame -= traj_dict[key]
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))

def write_rewarded_pdbs(rewarded_inds, sim_path, pdb_out_path, data_path):
    # Get list of simulation trajectory files (Assume all are equal length (ns))

    with open_h5(data_path) as h5_file:
        traj_files = h5_file['traj_file'][...]
        sim_lens = h5_file['sim_len'][...]

    traj_dict = dict(zip(traj_files, sim_lens))

    reward_locs = [find_frame(traj_dict, ind) for ind in rewarded_inds]

    # For documentation on mda.Writer methods see:
    #   https://www.mdanalysis.org/mdanalysis/documentation_pages/coordinates/PDB.html
    #   https://www.mdanalysis.org/mdanalysis/_modules/MDAnalysis/coordinates/PDB.html#PDBWriter._update_frame

    outlier_pdbs = []
    for frame, traj_file in reward_locs:
        sim_pdb = glob(os.path.join(os.path.dirname(traj_file), '*.pdb'))[0]
        basename = os.path.basename(os.path.dirname(traj_file))
        out_pdb = join(pdb_out_path, f'{basename}_{frame:06}.pdb')
        u = mda.Universe(sim_pdb, traj_file)
        with mda.Writer(out_pdb) as writer:
            # Write a single coordinate set to a PDB file
            writer._update_frame(u)
            writer._write_timestep(u.trajectory[frame])
        outlier_pdbs.append(out_pdb)

    return outlier_pdbs

def md_checkpoints(sim_path, pdb_out_path, outlier_pdbs):
    # Restarts from check point
    checkpnt_list = sorted(glob(os.path.join(sim_path, 'omm_runs_*/checkpnt.chk')))
    restart_checkpnts = [] 
    for checkpnt in checkpnt_list: 
        checkpnt_filepath = join(pdb_out_path, os.path.basename(os.path.dirname(checkpnt) + '.chk'))
        if not os.path.exists(checkpnt_filepath): 
            shutil.copy2(checkpnt, checkpnt_filepath) 
            # Includes only checkpoint of trajectory that contains an outlier 
            if any(os.path.basename(os.path.dirname(checkpnt)) in outlier for outlier in outlier_pdbs):  
                restart_checkpnts.append(checkpnt_filepath)

    return restart_checkpnts



@click.command()
@click.option('-s', '--sim_path', required=True,
              type=click.Path(exists=True),
              help='OpenMM simulation path containing *.dcd and *.pdb files')

@click.option('-p', '--pdb_out_path', default=join('.', 'outlier_pdbs'),
              help='Path to folder to write output PDB files to.')

@click.option('-o', '--restart_points_path',
              default=os.path.abspath('./restart_points.json'),
              help='Path to write output json file with PDB file list.')

@click.option('-d', '--data_path', required=True,
              type=click.Path(exists=True),
              help='Preprocessed data h5 file path')

@click.option('-m', '--model_paths', required=True,
              callback=parse_list,
              help='List of trained model paths "path1,path2,...".')

@click.option('-t', '--model_type', default='vae-resnet',
             help='Type of model. vae-resnet or aae.')

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

@click.option('-b', '--batch_size', default=256, type=int,
              help='Batch size for forward pass. Limited by available ' \
                   ' memory and size of input example. Does not affect output.' \
                   ' The larger the batch size, the faster the computation.')

def main(sim_path, pdb_out_path, restart_points_path, data_path, model_paths, model_type,
         min_samples, device, dim1, dim2, cm_format, batch_size):

    # Make directory to store output PDB files
    os.makedirs(pdb_out_path, exist_ok=True)

    best_hparams, best_checkpoint = model_selection(model_paths, num_select=1)

    print(best_hparams)
    print(best_checkpoint)

    # Generate embeddings for all contact matrices produced during MD stage
    embeddings, indices = generate_embeddings(model_type, best_hparams, best_checkpoint, dim1, dim2,
                                              device, data_path, cm_format, batch_size)

    print('embeddings shape: ', embeddings.shape)
    
    # Perform DBSCAN clustering on embeddings
    #outlier_inds, labels = perform_clustering(eps_path, best_checkpoint,
    #                                          embeddings, min_samples, eps)

    # Perform OPTICS clustering on embeddings
    #outlier_inds, labels = optics_clustering(embeddings, min_samples=min_samples)

    # Perform LocalOutlierFactor outlier detection on embeddings
    outlier_inds, scores = local_outlier_factor(embeddings, plot_dir=os.path.join(pdb_out_path,"figures"))
    
    print(outlier_inds.shape)
    for ind, score in zip(outlier_inds, scores):
        print(f'ind, score: {ind}, {score}')

    # Map shuffled indices back to in-order MD frames
    simulation_inds = indices[outlier_inds]

    # Write rewarded PDB files to shared path
    outlier_pdbs = write_rewarded_pdbs(simulation_inds, sim_path, pdb_out_path)
    restart_checkpnts = md_checkpoints(sim_path, pdb_out_path, outlier_pdbs)

    restart_points = restart_checkpnts + outlier_pdbs
    with open(restart_points_path, 'w') as restart_file:
        json.dump(restart_points, restart_file)

if __name__ == '__main__':
    main()
