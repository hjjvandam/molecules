import h5py 
import numpy as np
import simtk.unit as u
from MDAnalysis.analysis import distances


# TODO: h5py help https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/

class ContactMapReporter:
    def __init__(self, file, report_interval, native_pdb=''):
        self._report_interval = report_interval
        self._file = h5py.File(file, 'w', libver='latest', swmr=True)
        #self._file.swmr_mode = True
        self._cm_dset = self._file.create_dataset('contact_maps', shape=(2,0), maxshape=(None, None))
        self._rmsd_dset = self._file.create_dataset('rmsd', shape=(1,), maxshape=(None,))
        self._native_contact_dset = self._file.create_dataset('native_contact', shape=(1,), maxshape=(None,))

        # TODO: load in native pdb into a mdanalysis universe

    def __del__(self):
        self._file.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _report_contact_maps(self, simulation, state):
        # TODO: can we store contact_map dtype as bool?
        ca_indices = [atom.index for atom in simulation.topology.atoms() if atom.name == 'CA']
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions_ca = positions[ca_indices].astype(np.float32)
        distance_matrix = distances.self_distance_array(positions_ca)
        contact_map = (distance_matrix < 8.0) * 1.0 
        new_shape = (len(contact_map), self._cm_dset.shape[1] + 1)
        self._cm_dset.resize(new_shape)
        self._cm_dset[:, new_shape[1] - 1] = contact_map

    def _report_rmsd(self, simulation, state):
        # TODO: which atoms do we compute rmsd for
        pass

    def _report_native_contact(self, simulation, state):
        # TODO: which atoms do we compute rmsd for
        pass

    def _report(self, simulation, state):
        self._report_contact_maps(simulation, state)
        self._report_rmsd(simulation, state)
        self._report_native_contact(simulation, state)
        self._file.flush()
