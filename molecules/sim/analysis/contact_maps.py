import os
import tables
import MDAnalysis as mda
from MDAnalysis.analysis import contacts, distances


def contact_maps_from_traj(pdb_file, traj_file, contact_cutoff=8.0, savefile=None):
    """
    Get contact map from trajectory.
    """
    
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
        cm_matrix = contacts.contact_matrix(distances.self_distance_array(ca.positions), radius=contact_cutoff) * 1.0 
        contact_matrices.append(cm_matrix)

    if savefile:
        cm_table.append(contact_matrices)
        outfile.close() 

    return contact_matrices

# TODO: implement for pytorch
def fraction_of_contacts(cm, ref_cm):
    """Computes the fraction of contacts shared between two contact matrices."""
    # if isinstance(cm, np.ndarray):
    #     return (cm == ref_cm).mean()
    # else torch.Tensor
    #return (cm == ref_cm).sum() / (cm.shape[0] * cm.shape[1])
    pass
