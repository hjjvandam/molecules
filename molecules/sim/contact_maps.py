import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms, align

def _save_sparse_contact_maps(save_file, row, col, rmsd=None, fnc=None):
    """Helper function. Saves COO row/col matrices to file."""
    import h5py
    import numpy as np
    from molecules.utils import open_h5

    kwargs = {'fletcher32': True}

    h5_file = open_h5(save_file, 'w')
    group = h5_file.create_group('contact_maps')
    # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype('int16'))
    # The i'th element of both row,col dset will be
    # arrays of the same length. However, neighboring
    # arrays may be any variable length.
    group.create_dataset('row', dtype=dt, data=row, **kwargs)
    group.create_dataset('col', dtype=dt, data=col, **kwargs)

    # If specified, write rmsd and/or fraction of native contacts
    if rmsd is not None:
        h5_file.create_dataset('rmsd', dtype='float16', data=rmsd, **kwargs)
    if fnc is not None:
        h5_file.create_dataset('fnc', dtype='float16', data=fnc, **kwargs)

    h5_file.flush()
    h5_file.close()

def fraction_of_native_contacts(cm, native_cm):
    return 1 - (cm != native_cm).mean()

def sparse_contact_maps_from_traj(pdb_file, ref_pdb_file, traj_file,
                                  cutoff=8., sel='protein and name CA',
                                  save_file=None, verbose=False):
    """
    Get contact map from trajectory. Requires all row,col indicies
    from the traj to fit into memory at the same time.

    Parameters
    ----------
    sel : str
        Select carbon-alpha atoms in the protein for RMSD and contact maps
    """
    # TODO: update docstring

    # Load simulation trajectory and reference structure into memory
    sim = mda.Universe(pdb_file, traj_file)
    ref = mda.Universe(ref_pdb_file)

    if verbose:
        print('Traj length: ', len(sim.trajectory))

    # TODO: update ca_atoms to be a more general var name
    # Select atoms for RMSD, fnc and contact map
    ca_atoms = sim.select_atoms(sel)

    # Select atoms in reference structure for RMSD
    ref_positions = ref.select_atoms(sel).positions.copy()
    # Generate reference contact map
    ref_cm = distances.contact_matrix(ref_positions, cutoff, returntype='sparse')

    # Align trajectory to compute accurate RMSD
    align.AlignTraj(sim, ref, in_memory=True).run()

    # Buffers for sparse contact row/col data
    row, col = [], []
    # Buffers for RMSD to native state and fraction of native contacts
    params = {'shape': len(sim.trajectory), 'dtype': np.float32}
    rmsd, fnc = np.empty(**params), np.empty(**params)

    for i, frame in enumerate(sim.trajectory):

        # Compute contact map in scipy lil_matrix form
        cm = distances.contact_matrix(ca_atoms.positions, cutoff, returntype='sparse')

        # Compute and store RMSD and fraction of native contacts
        positions = sim.select_atoms(sel).positions
        rmsd[i] = rms.rmsd(positions, ref_positions, center=True,
                           superposition=True)
        fnc[i] = fraction_of_native_contacts(cm, ref_cm)

        # Represent contact map in COO sparse format
        coo = cm.tocoo()
        row.append(coo.row.astype('int16'))
        col.append(coo.col.astype('int16'))

        if verbose:
            print(f'Writing frame {i}/{len(sim.trajectory)}\tfnc:'
                  f'{fnc[i]}\trmsd: {rmsd[i]}\tshape: {coo.shape}')

    if save_file:
        _save_sparse_contact_maps(save_file, row, col, rmsd, fnc)

    return row, col, rmsd, fnc

def _worker(kwargs):
        id_ = kwargs.pop('id')
        return sparse_contact_maps_from_traj(**kwargs), id_

def parallel_traj_to_dset(pdb_file, ref_pdb_file, save_file, traj_files=[],
                          cutoff=8., verbose=False):

    import itertools
    import concurrent.futures

     #    def _worker(kwargs):
     #       id_ = kwargs.pop('id')
     #      return sparse_contact_maps_from_traj(**kwargs), id_

    kwargs = [{'pdb_file': pdb_file,
               'ref_pdb_file': ref_pdb_file,
               'traj_file': traj_file,
               'cutoff': cutoff,
               'verbose': verbose and (i % 80 == 0), # 80 cores on lambda
               'id': i}
              for i, traj_file in enumerate(traj_files)]

    rows, cols, rmsds, fncs, ids = [], [], [], [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for (row, col, rmsd, fnc), id_ in executor.map(_worker, kwargs):
            rows.append(row)
            cols.append(col)
            rmsds.append(rmsd)
            fncs.append(fnc)
            ids.append(id_)
            if verbose:
                print('finished id: ', id_)

    # Negligible runtime (1e-05 seconds)
    rows_, cols_, rmsds_, fncs_, = [], [], [], []
    for _, row, col, rmsd, fnc  in sorted(zip(ids, rows, cols, rmsds, fncs)):
        rows_.append(row)
        cols_.append(col)
        rmsds_.append(rmsd)
        fncs_.append(fnc)

    # Negligible runtime (1e-2 seconds)
    rows_ = list(itertools.chain.from_iterable(rows_))
    cols_ = list(itertools.chain.from_iterable(cols_))
    rmsds_ = np.concatenate(rmsds_)
    fncs_ = np.concatenate(fncs_)

    _save_sparse_contact_maps(save_file, rows_, cols_, rmsds_, fncs_)

def sparse_contact_maps_from_matrices(contact_maps, save_file=None):
    """Convert normal contact matrices to sparse format."""
    from scipy.sparse import coo_matrix

    row, col = [], []
    for cm in map(lambda cm: cm.squeeze(), contact_maps):
        # Represent contact matrix in COO sparse format
        coo = coo_matrix(cm, shape=cm.shape)
        row.append(coo.row.astype('int16'))
        col.append(coo.col.astype('int16'))

    if save_file:
        _save_sparse_contact_maps(save_file, row, col)

    return row, col
