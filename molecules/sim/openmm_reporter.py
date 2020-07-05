import h5py 
import numpy as np
import simtk.unit as u
from MDAnalysis import Universe
from MDAnalysis.analysis import distances, rms

# TODO: h5py help https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/

# TODO: this class needs testing for the RMSD addtions.
class ContactMapReporter:
    def __init__(self, file, report_interval, native_pdb):
        self._report_interval = report_interval
        # TODO: May want to use chunked storage. Could chunk into sizes manageable for training.
        #       If chunked storage is not used, then finding continuous disk space as the data
        #       set grows may lead to overhead. Might implicitly happen already since we set maxshape.
        #       http://docs.h5py.org/en/stable/high/dataset.html#chunked-storage

        # TODO: Consider using compression. Might have big disk space savings given the sparsity.
        #       http://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline

        # TODO: Use checksum to check for data corruption.
        #       http://docs.h5py.org/en/stable/high/dataset.html#fletcher32-filter
        self._file = h5py.File(file, 'w', libver='latest', swmr=True)

        # TODO: test the dtype feature.
        # TODO: when reading in data, can convert type to what is needed with
        #       >>> with dset.astype('int16'):
        #       ...     out = dset[:]
        self._cm_dset = self._file.create_dataset('contact_maps', dtype='i1',
                                                  shape=(2,0), maxshape=(None, None))
        self._rmsd_dset = self._file.create_dataset('rmsd', shape=(1,1), maxshape=(None,1))

        self._native_positions = Universe(native_pdb).select_atoms('protein').positions()

    def __del__(self):
        self._file.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _report_contact_maps(self, simulation, state, ca_positions):
        # TODO: http://docs.h5py.org/en/stable/faq.html
        #       h5py supported integer types: 1, 2, 4 or 8 byte, BE/LE, signed/unsigned.
        #       store as 1 byte int
        contact_map = (distances.self_distance_array(ca_positions) < 8.) * 1.

        self._cm_dset.resize(self._cm_dset.shape[1] + 1, axis=0)
        self._cm_dset[:, -1] = contact_map

    def _report_rmsd(self, simulation, state, positions):
        # RMSD of all protein atoms
        rmsd = rms.rmsd(positions, self._native_positions)

        self._rmsd_dset.resize(self._rmsd_dset.shape[1] + 1, axis=0)
        self._rmsd_dset[0, -1] = rmsd

    def _report(self, simulation, state):
        ca_indices = [a.index for a in simulation.topology.atoms() if a.name == 'CA']
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        ca_positions = positions[ca_indices].astype(np.float32)

        self._report_contact_maps(simulation, state, ca_positions)
        self._report_rmsd(simulation, state, positions)
        self._file.flush()
