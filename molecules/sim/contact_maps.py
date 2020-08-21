import h5py
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances, rms, align
from molecules.utils import open_h5

def _save_sparse_contact_maps(h5_file, contact_maps, cm_format='sparse-concat', **kwargs):
    """
    Saves contact maps in sparse format. Only stores row,col
    indices of the 1s in the contact matrix. Values must be
    recomputed on the fly as an array of ones equal to the
    length of the row,col vectors. Note, row and col vectors
    have the same length.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 file to write contact maps to

    contact_maps : tuple
        (row, col) where row,col are lists of np.ndarrays.
        row[i], col[i] represents the indices of a contact
        map where a 1 exists.

    cm_format : str
        sparse-concat: new format with single dataset
        sparse-rowcol: old format with group containing row,col datasets

    kwargs : dict
        Optional h5py parameters to be used in dataset creation.

   Note: The sparse-concat format has more efficient chunking than
         the sparse-rowcol format.
   """

    rows, cols = contact_maps

     # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype('int16'))

    # Helper function to create ragged array
    ragged = lambda data: np.array(data, dtype=object)

    if cm_format == 'sparse-concat':
        # list of np arrays of shape (2 * X) where X varies
        data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
        h5_file.create_dataset('contact_map', data=data, chunks=(1,) + data.shape[1:], dtype=dt, **kwargs)

    elif cm_format == 'sparse-rowcol':
        group = h5_file.create_group('contact_map')
        # The i'th element of both row,col dset will be arrays of the
        # same length. However, neighboring arrays may be any variable length.
        group.create_dataset('row', dtype=dt, data=ragged(rows), chunks=(1,), **kwargs)
        group.create_dataset('col', dtype=dt, data=ragged(cols), chunks=(1,), **kwargs)

def _save(save_file, rmsd=None, fnc=None, point_cloud=None,
          contact_maps=None, cm_format='sparse-concat'):
    """
    Saves data to h5 file. All data is optional, can save any
    combination of data input arrays.

    Parameters
    ----------
    save_file : str
        Path of output h5 file used to save datasets.

    rmsd : np.ndarray
        Stores floating point rmsd data.

    fnc : np.ndarray
        Stores floating point fraction of native contact data.

    point_cloud : np.ndarray
        Stores matrix of point cloud data. (N, K, 3) where K
        is the number of selected atoms.

    contact_maps : tuple
        (row, col) where row,col are lists of np.arrays.
        row[i], col[i] represents the indices of a contact
        map where a 1 exists.

    cm_format : str
        format to save contact maps with.
    """
    kwargs = {'fletcher32': True}
    scalar_kwargs = {'fletcher32': True, 'dtype': 'float16', 'chunks':(1,)}

    # Open h5 file in swmr mode
    h5_file = open_h5(save_file, 'w')

    # Save rmsd
    if rmsd is not None:
        h5_file.create_dataset('rmsd', data=rmsd, **scalar_kwargs)

    # Save fraction of native contacts
    if fnc is not None:
        h5_file.create_dataset('fnc', data=fnc, **scalar_kwargs)

    # Save point cloud
    if point_cloud is not None:
        h5_file.create_dataset('point_cloud', data=point_cloud,
                               chunks=(1,) + point_cloud.shape[1:],
                               dtype='float32', **kwargs)

    # Save contact maps
    if contact_maps is not None:
        _save_sparse_contact_maps(h5_file, contact_maps, cm_format=cm_format, **kwargs)

    # Flush data to file and close
    h5_file.flush()
    h5_file.close()

def fraction_of_contacts(cm, ref_cm):
    return 1 - (cm != ref_cm).mean()

def _traj_to_dset(pdb_file, ref_pdb_file, traj_file,
                  sel='protein and name CA', cutoff=8.,
                  rmsd=True, fnc=True,
                  point_cloud=True,
                  contact_map=True,
                  save_file=None,
                  cm_format='sparse-concat',
                  verbose=False):
    """
    Get a point cloud representation from trajectory.

    Parameters
    ----------
    sel: str
        Select any set of atoms in the protein for RMSD and point clouds
    """
    # TODO: update doc string

    # Load simulation and reference structures
    sim = mda.Universe(pdb_file, traj_file)
    ref = mda.Universe(ref_pdb_file)

    if verbose:
        print('Traj length: ', len(sim.trajectory))

    # Atom selection for reference
    atoms = sim.select_atoms(sel)
    # Get atomic coordinates of reference atoms
    ref_positions = ref.select_atoms(sel).positions.copy()
    # Get contact map of reference atoms
    ref_cm = distances.contact_matrix(ref_positions, cutoff, returntype='sparse')

    if rmsd or point_cloud:
        # Align trajectory to compute accurate RMSD or point cloud
        align.AlignTraj(sim, ref, in_memory=True).run()

    # Initialize buffers. Only turn into containers if user specifies to.
    rmsd_data, fnc_data, pc_data, contact_map_data = None, None, None, None

    # Buffers for storing RMSD and fraction of native contacts
    params = {'shape': len(sim.trajectory), 'dtype': np.float16}
    if rmsd:
        rmsd_data = np.empty(**params)
    if fnc:
        fnc_data = np.empty(**params)

    # Buffers for sparse contact row/col data
    if contact_map:
        row, col = [], []
        contact_map_data = (row, col)

    # Buffer for point cloud data
    if point_cloud:
        pc_data = np.empty(shape=(len(sim.trajectory), *ref_positions.shape),
                           dtype=np.float32)

    for i, frame in enumerate(sim.trajectory):

        # Point cloud positions of selected atoms in frame i
        positions = atoms.positions

        if contact_map or fnc:
            # Compute contact map of current frame (scipy lil_matrix form)
            cm = distances.contact_matrix(positions, cutoff, returntype='sparse')

        if rmsd:
            # Compute and store RMSD to reference state
            rmsd_data[i] = rms.rmsd(positions, ref_positions, center=True,
                                    superposition=True)
        if fnc:
            # Compute fraction of contacts to reference state
            fnc_data[i] = fraction_of_contacts(cm, ref_cm)

        if contact_map:
            # Represent contact map in COO sparse format
            coo = cm.tocoo()
            row.append(coo.row.astype('int16'))
            col.append(coo.col.astype('int16'))

        if point_cloud:
            # Store reference atoms point cloud of current frame
            pc_data[i] = positions.copy()

        if verbose:
            if i % 100 == 0: # Print every 100 frames
                str_ = f'Frame {i}/{len(sim.trajectory)}'
                if rmsd:
                    str_ += f'\trmsd: {rmsd_data[i]}'
                if fnc:
                    str_ += f'\tfnc: {fnc_data[i]}'
                str_ += f'\tshape: {positions.shape}'

                print(str_)

    if point_cloud:
        pc_data = np.reshape(pc_data, (len(sim.trajectory), 3, -1))

    if save_file:
        # Write data to HDF5 file
        _save(save_file, rmsd=rmsd_data, fnc=fnc_data,
              point_cloud=pc_data, contact_maps=contact_map_data,
              cm_format=cm_format)

    # Any of these could be None based on the user input
    return rmsd_data, fnc_data, pc_data, contact_map_data

def _worker(kwargs):
        id_ = kwargs.pop('id')
        return _traj_to_dset(**kwargs), id_

def traj_to_dset(pdb_file, ref_pdb_file, save_file, traj_files,
                 rmsd=True, fnc=True, point_cloud=True, contact_map=True,
                 sel='protein and name CA', cutoff=8., num_workers=None,
                 cm_format='sparse-concat', verbose=False):

    if num_workers == 1:
       return _traj_to_dset(pdb_file, ref_pdb_file, traj_files, sel=sel,
                            cutoff=cutoff, rmsd=rmsd, fnc=fnc,
                            point_cloud=point_cloud, contact_map=contact_map,
                            save_file=save_file, cm_format=cm_format, verbose=verbose)

    import itertools
    from concurrent.futures import ProcessPoolExecutor

    # Set num_workers to max possible unless specified by user
    if num_workers is None:
        import os
        num_workers = os.cpu_count()

    # Arguments for workers
    kwargs = [{'pdb_file': pdb_file,
               'ref_pdb_file': ref_pdb_file,
               'traj_file': traj_file,
               'rmsd': rmsd,
               'fnc': fnc,
               'point_cloud': point_cloud,
               'contact_map': contact_map,
               'cutoff': cutoff,
               'verbose': verbose and (i % num_workers == 0),
               'id': i}
              for i, traj_file in enumerate(traj_files)]

    # initialize buffers
    ids = []
    if rmsd:
        rmsds = []
    if fnc:
        fncs = []
    if point_cloud:
        point_clouds = []
    if contact_map:
        rows, cols = [], []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for (rmsd_data, fnc_data, pc_data, contact_map_data), id_ in executor.map(_worker, kwargs):
            if rmsd:
                rmsds.append(rmsd_data)
            if fnc:
                fncs.append(fnc_data)
            if point_cloud:
                point_clouds.append(pc_data)
            if contact_map:
                row, col = contact_map_data
                rows.append(row)
                cols.append(col)

            ids.append(id_)

            if verbose:
                print('finished id: ', id_)

    # Initialize buffers. Only turn into containers if user specifies to.
    rmsds_, fncs_, point_clouds_, contact_maps_ = None, None, None, None
    if rmsd:
        rmsds_ = []
    if fnc:
        fncs_ = []
    if point_cloud:
        point_clouds_ = []
    if contact_map:
        rows_, cols_ = [], []

    # Negligible runtime (1e-05 seconds)
    for _, rmsd_data, fnc_data, pc_data, row, col  in sorted(zip(ids, rmsds, fncs, point_clouds, rows, cols)):
        if rmsd:
            rmsds_.append(rmsd_data)
        if fnc:
            fncs_.append(fnc_data)
        if point_cloud:
            point_clouds_.append(pc_data)
        if contact_map:
            rows_.append(row)
            cols_.append(col)

    # Negligible runtime (1e-2 seconds)
    if rmsd:
        rmsds_ = np.concatenate(rmsds_)
    if fnc:
        fncs_ = np.concatenate(fncs_)
    if point_cloud:
        point_clouds_ = np.concatenate(point_clouds_)
    if contact_map:
        rows_ = list(itertools.chain.from_iterable(rows_))
        cols_ = list(itertools.chain.from_iterable(cols_))
        contact_maps_ = (rows_, cols_)

    _save(save_file, rmsd=rmsds_, fnc=fncs_, point_cloud=point_clouds_,
            contact_maps=contact_maps_, cm_format=cm_format)

    return rmsds_, fncs_, point_clouds_, contact_maps_

def sparse_contact_maps_from_matrices(contact_maps, rmsd=None, fnc=None,
        save_file=None, cm_format='sparse-concat'):
    """Convert normal contact matrices to sparse format."""
    from scipy.sparse import coo_matrix

    row, col = [], []
    for cm in map(lambda cm: cm.squeeze(), contact_maps):
        # Represent contact matrix in COO sparse format
        coo = coo_matrix(cm, shape=cm.shape)
        row.append(coo.row.astype('int16'))
        col.append(coo.col.astype('int16'))

    if save_file:
        _save(save_file, rmsd=rmsd, fnc=fnc, contact_maps=(row, col),
              cm_format=cm_format)

    return row, col
