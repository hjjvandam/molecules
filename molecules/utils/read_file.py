import h5py

def open_h5(h5_file, mode = 'r', libver = 'latest', swmr = False, **kwargs):
    """
    Helper function for opening h5 file in context manager

    Parameters
    ----------
    h5_file : str
        name of h5 file to open

    mode : str
        mode to open the file in 'r' or 'w'

    Returns
    -------
    open h5py file to be used in a context manager
    """
    # TODO: bug in scripts/traj_to_dset.py when swmr, libver='latest' is used
    return h5py.File(h5_file, mode, **kwargs)
