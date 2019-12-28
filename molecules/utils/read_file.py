import h5py
import tables 


def read_h5_contact_maps(filename): 
    data = tables.open_file(filename, 'r')
    contact_maps = data.root.contact_maps.read()
    data.close()
    return contact_maps


def read_h5_RMSD(filename):
    data = tables.open_file(filename, 'r')
    RMSD = data.root.RMSD.read()
    data.close()
    return RMSD


def open_h5(h5_file):
    """
    Opens file in single write multiple reader mode
    libver specifies newest available version,
    may not be backwards compatable

    Parameters
    ----------
    h5_file : str
        name of h5 file to open

    Returns
    -------
    open h5py file

    """
    return h5py.File(h5_file, 'r', libver='latest', swmr=True)
