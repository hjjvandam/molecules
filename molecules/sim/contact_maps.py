import MDAnalysis as mda
from MDAnalysis.analysis import distances


def contact_maps_from_traj(pdb_file, traj_file, contact_cutoff=8.0, savefile=None):
    """
    Get contact map from trajectory.
    """
    import os
    import tables
    mda_traj = mda.Universe(pdb_file, traj_file)
    traj_length = len(mda_traj.trajectory) 
    ca = mda_traj.select_atoms('name CA')
    
    if savefile:
        savefile = os.path.abspath(savefile)
        outfile = tables.open_file(savefile, 'w')
        atom = tables.Float64Atom()
        cm_table = outfile.create_earray(outfile.root, 'contact_maps', atom, shape=(traj_length, 0)) 

    contact_matrices = []
    for frame in mda_traj.trajectory:
        cm_matrix = (distances.self_distance_array(ca.positions) < contact_cutoff) * 1.0
        contact_matrices.append(cm_matrix)

    if savefile:
        cm_table.append(contact_matrices)
        outfile.close() 

    return contact_matrices


def sparse_contact_maps_from_traj(pdb_file, traj_file,
                                  contact_cutoff=8., savefile=None):
    """Get contact map from trajectory."""

    import h5py
    from molecules.utils import open_h5
    from scipy.sparse import coo_matrix

    sim = mda.Universe(pdb_file, traj_file)
    ca = sim.select_atoms('name CA')

    row, col = [], []
    for frame in sim.trajectory:
        # Compute contact matrix
        cm = (distances.self_distance_array(ca.positions) < contact_cutoff) * 1.

        # Represent contact matrix in COO sparse format
        coo = coo_matrix(cm, shape=cm.shape)
        row.append(coo.row.astype('int16'))
        col.append(coo.col.astype('int16'))

    if savefile:
        with open_h5(savfile, 'w') as h5_file:
            group = file.create_group('contact_maps')

            # Specify variable length arrays
            dt = h5py.vlen_dtype(np.dtype('int16'))

            # The i'th element of both row,col dset will be
            # arrays of the same length. However, neighboring
            # arrays may be any variable length.
            group.create_dataset('row', dtype=dt, data=row)
            group.create_dataset('col', dtype=dt, data=col)

    return row, col

