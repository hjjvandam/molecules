import uuid
import h5py
import numpy as np
import simtk.unit as u
import MDAnalysis as mda
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
        self._file = h5py.File(file, "w", libver="latest", swmr=True)

        # TODO: test the dtype feature.
        # TODO: when reading in data, can convert type to what is needed with
        #       >>> with dset.astype('int16'):
        #       ...     out = dset[:]
        self._cm_dset = self._file.create_dataset(
            "contact_maps", dtype="i1", shape=(2, 0), maxshape=(None, None)
        )
        self._rmsd_dset = self._file.create_dataset(
            "rmsd", shape=(1, 1), maxshape=(None, 1)
        )

        self._native_positions = (
            mda.Universe(native_pdb).select_atoms("protein").positions()
        )

    def __del__(self):
        self._file.close()

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _report_contact_maps(self, simulation, state, ca_positions):
        # TODO: http://docs.h5py.org/en/stable/faq.html
        #       h5py supported integer types: 1, 2, 4 or 8 byte, BE/LE, signed/unsigned.
        #       store as 1 byte int
        contact_map = (distances.self_distance_array(ca_positions) < 8.0) * 1.0

        self._cm_dset.resize(self._cm_dset.shape[1] + 1, axis=0)
        self._cm_dset[:, -1] = contact_map

    def _report_rmsd(self, simulation, state, positions):
        # RMSD of all protein atoms
        rmsd = rms.rmsd(positions, self._native_positions)

        self._rmsd_dset.resize(self._rmsd_dset.shape[1] + 1, axis=0)
        self._rmsd_dset[0, -1] = rmsd

    def _report(self, simulation, state):
        ca_indices = [a.index for a in simulation.topology.atoms() if a.name == "CA"]
        positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        ca_positions = positions[ca_indices].astype(np.float32)

        self._report_contact_maps(simulation, state, ca_positions)
        self._report_rmsd(simulation, state, positions)
        self._file.flush()


def write_contact_map_h5(h5_file, rows, cols):

    # Helper function to create ragged array
    def ragged(data):
        a = np.empty(len(data), dtype=object)
        a[...] = data
        return a

    # Specify variable length arrays
    dt = h5py.vlen_dtype(np.dtype("int16"))

    # list of np arrays of shape (2 * X) where X varies
    data = ragged([np.concatenate(row_col) for row_col in zip(rows, cols)])
    h5_file.create_dataset(
        "contact_map",
        data=data,
        dtype=dt,
        fletcher32=True,
        chunks=(1,) + data.shape[1:],
    )


def write_rmsd(h5_file, rmsd):
    h5_file.create_dataset(
        "rmsd", data=rmsd, dtype="float16", fletcher32=True, chunks=(1,)
    )


def wrap(atoms):
    def wrap_nsp10_16(positions):
        # update the positions
        atoms.positions = positions
        # only porting CA into nsp16
        nsp16 = atoms.segments[0].atoms
        # wrapping atoms into continous frame pbc box
        box_edge = nsp16.dimensions[0]
        box_center = box_edge / 2
        trans_vec = box_center - np.array(nsp16.center_of_mass())
        atoms.translate(trans_vec).wrap()
        trans_vec = box_center - np.array(atoms.center_of_mass())
        atoms.translate(trans_vec).wrap()

        return atoms.positions

    return wrap_nsp10_16


class SparseContactMapReporter:
    def __init__(
        self,
        file,
        reportInterval,
        wrap_pdb_file=None,
        reference_pdb_file=None,
        selection="CA",
        threshold=8.0,
        batch_size=2,  # 1024,
        senders=[],
    ):

        self._file_idx = 0
        self._base_name = file
        self._reference_pdb_file = reference_pdb_file
        self._report_interval = reportInterval
        self._selection = selection
        self._threshold = threshold
        self._batch_size = batch_size
        self._senders = senders

        self._init_batch()

        # Set up for reporting optional RMSD to reference state
        if reference_pdb_file is not None:
            u = mda.Universe(self._reference_pdb_file)
            # Convert openmm atom selection to mda
            selection = f"protein and name {self._selection}"
            self._reference_positions = u.select_atoms(selection).positions.copy()
            self._rmsd = []
        else:
            self._reference_positions = None

        if wrap_pdb_file is not None:
            u = mda.Universe(wrap_pdb_file)
            selection = f"protein and name {self._selection}"
            atoms = u.select_atoms(selection)
            self.wrap = wrap(atoms)
        else:
            self.wrap = None

    def _init_batch(self):
        # Frame counter for writing batches to HDF5
        self._num_frames = 0
        # Row, Column indices for contact matrix in COO format
        self._rows, self._cols = [], []

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return (steps, True, False, False, False, None)

    def _collect_rmsd(self, positions):
        if self.wrap is not None:
            positions = self.wrap(positions)

        rmsd = rms.rmsd(positions, self._reference_positions, superposition=True)
        self._rmsd.append(rmsd)

    def _collect_contact_maps(self, positions):

        contact_map = distances.contact_matrix(
            positions, self._threshold, returntype="sparse"
        )

        # Represent contact map in COO sparse format
        coo = contact_map.tocoo()
        self._rows.append(coo.row.astype("int16"))
        self._cols.append(coo.col.astype("int16"))

    def report(self, simulation, state):
        atom_indices = [
            a.index for a in simulation.topology.atoms() if a.name == self._selection
        ]
        all_positions = np.array(state.getPositions().value_in_unit(u.angstrom))
        positions = all_positions[atom_indices].astype(np.float32)

        self._collect_contact_maps(positions)

        if self._reference_positions is not None:
            self._collect_rmsd(positions)

        self._num_frames += 1

        if self._num_frames == self._batch_size:
            file_name = f"{self._base_name}_{self._file_idx:05d}_{uuid.uuid4()}.h5"

            with h5py.File(file_name, "w", swmr=False) as h5_file:
                write_contact_map_h5(h5_file, self._rows, self._cols)
                # Optionally, write rmsd to the reference state
                if self._reference_positions is not None:
                    write_rmsd(h5_file, self._rmsd)
                    self._rmsd = []

            self._init_batch()
            self._file_idx += 1

            for sender in self._senders:
                sender.send(file_name)
