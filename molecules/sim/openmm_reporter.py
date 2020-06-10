import h5py 
import numpy as np
import simtk.unit as u
from MDAnalysis import Universe
from MDAnalysis.analysis import distances, rms

# TODO: h5py help https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/

class ContactMapReporter:
    def __init__(self, file, report_interval, native_pdb=''):
        self._report_interval = report_interval
        self._file = h5py.File(file, 'w', libver='latest', swmr=True)
        #self._file.swmr_mode = True
        self._cm_dset = self._file.create_dataset('contact_maps', shape=(2,0), maxshape=(None, None))
        self._rmsd_dset = self._file.create_dataset('rmsd', shape=(1,1), maxshape=(None,1))
        self._native_contact_dset = self._file.create_dataset('native_contact', shape=(1,1), maxshape=(None,1))

        u = Universe(native_pdb)
        self._native_positions = u.select_atoms('protein').positions()
        self._native_ca_positions = u.select_atoms('CA').positions()

    def __del__(self):
        self._file.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _report_contact_maps(self, simulation, state, ca_positions):
        # TODO: can we store contact_map dtype as bool?
        contact_map = (distances.self_distance_array(ca_positions) < 8.) * 1.

        self._cm_dset.resize((len(contact_map), self._cm_dset.shape[1] + 1))
        self._cm_dset[:, -1] = contact_map

    def _report_rmsd(self, simulation, state, positions):
        # RMSD of all protein atoms
        rmsd = rms.rmsd(positions, self._native_positions)

        self._rmsd_dset.resize(1, self._rmsd_dset.shape[1] + 1)
        self._rmsd_dset[0, -1] = rmsd

    def _report_native_contact(self, simulation, state, ca_positions):
        # Native contacts are based on a distance cutoff of 8 Ansgtroms less between between C-alphas
        native_contact = (np.abs(ca_positions - self._native_ca_positions) < 8.).mean()

        self._native_contact_dset.resize(1, self._native_contact_dset.shape[1] + 1)
        self._native_contact_dset[0, -1] = native_contact

    def _report(self, simulation, state):
        ca_indices = [a.index for a in simulation.topology.atoms() if a.name == 'CA']
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        ca_positions = positions[ca_indices].astype(np.float32)

        self._report_contact_maps(simulation, state, ca_positions)
        self._report_rmsd(simulation, state, positions)
        self._report_native_contact(simulation, state, ca_positions)
        self._file.flush()
