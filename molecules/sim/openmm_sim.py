import shutil
import random
from pathlib import Path
import parmed
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app

from molecules.sim.openmm_reporter import OfflineReporter


def configure_amber_implicit(
    pdb_file, top_file, dt_ps, temperature_kelvin, platform, platform_properties
):

    # Configure system
    if top_file:
        pdb = parmed.load_file(top_file, xyz=pdb_file)
        system = pdb.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
            implicitSolvent=app.OBC1,
        )
    else:
        pdb = parmed.load_file(pdb_file)
        forcefield = app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(temperature_kelvin, 91.0 / u.picosecond, dt_ps)
    integrator.setConstraintTolerance(0.00001)

    sim = app.Simulation(
        pdb.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, pdb


def configure_amber_explicit(
    pdb_file, top_file, dt_ps, temperature_kelvin, platform, platform_properties
):

    # Configure system
    top = parmed.load_file(top_file, xyz=pdb_file)
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometer,
        constraints=app.HBonds,
    )

    # Congfigure integrator
    integrator = omm.LangevinIntegrator(temperature_kelvin, 1 / u.picosecond, dt_ps)
    system.addForce(omm.MonteCarloBarostat(1 * u.bar, temperature_kelvin))

    sim = app.Simulation(
        top.topology, system, integrator, platform, platform_properties
    )

    # Return simulation and handle to coordinates
    return sim, top


def configure_simulation(ctx, solvent_type, gpu_index, dt_ps, temperature_kelvin):
    # Configure hardware
    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        platform_properties = {"DeviceIndex": str(gpu_index), "CudaPrecision": "mixed"}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        platform_properties = {"DeviceIndex": str(gpu_index)}

    # Select implicit or explicit solvent
    args = (
        ctx.pdb_file,
        ctx.top_file,
        dt_ps,
        temperature_kelvin,
        platform,
        platform_properties,
    )
    if solvent_type == "implicit":
        sim, coords = configure_amber_implicit(*args)
    else:
        assert solvent_type == "explicit"
        sim, coords = configure_amber_explicit(*args)

    # Set simulation positions
    if coords.get_coordinates().shape[0] == 1:
        sim.context.setPositions(coords.positions)
    else:
        positions = random.choice(coords.get_coordinates())
        sim.context.setPositions(positions / 10)

    # Minimize energy and equilibrate
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * u.kelvin, random.randint(1, 10000))

    return sim


def configure_reporters(sim, ctx, report_interval_ps, dt_ps, frames_per_h5, wrap):
    report_freq = int(report_interval_ps / dt_ps)

    # Configure DCD file reporter
    sim.reporters.append(app.DCDReporter(ctx.traj_file, report_freq))

    # Configure contact map reporter
    sim.reporters.append(
        OfflineReporter(
            ctx.h5_prefix,
            report_freq,
            wrap_pdb_file=ctx.pdb_file if wrap else None,
            reference_pdb_file=ctx.reference_pdb_file,
            frames_per_h5=frames_per_h5,
        )
    )

    # Configure simulation output log
    sim.reporters.append(
        app.StateDataReporter(
            ctx.log_file,
            report_freq,
            step=True,
            time=True,
            speed=True,
            potentialEnergy=True,
            temperature=True,
            totalEnergy=True,
        )
    )


def get_system_name(pdb_file: Path) -> str:
    # pdb_file: /path/to/pdb/<system-name>__<everything-else>.pdb
    return pdb_file.with_suffix("").name.split("__")[0]


def get_topology(initial_pdb_dir: Path, pdb_file: Path) -> Path:
    # pdb_file: /path/to/pdb/<system-name>__<everything-else>.pdb
    # top_file: initial_pdb_dir/<system-name>/*.top
    system_name = get_system_name(pdb_file)
    return list(initial_pdb_dir.joinpath(system_name).glob("*.top"))[0]


class SimulationContext:
    def __init__(
        self,
        pdb_file,
        reference_pdb_file,
        omm_dir_prefix,
        omm_parent_dir,
        result_dir,
        initial_pdb_dir,
        solvent_type,
    ):

        self.result_dir = result_dir
        self.reference_pdb_file = reference_pdb_file
        self.workdir = Path(omm_parent_dir).joinpath(omm_dir_prefix)

        self._init_workdir(Path(pdb_file), Path(initial_pdb_dir), solvent_type)

    def _init_workdir(self, pdb_file, initial_pdb_dir, solvent_type):
        # Make node-local dir
        self.workdir.mkdir()

        # Copy files to node-local-dir
        self.pdb_file = self._copy_pdb_file(pdb_file)
        if solvent_type == "explicit":
            self.top_file = self._copy_top_file(initial_pdb_dir)

    def _copy_pdb_file(self, pdb_file: Path) -> str:
        # On initial iterations need to change the PDB file names to include
        # the system information to look up the topology
        if "__" in pdb_file.name:
            copy_to_file = pdb_file.name
        else:
            # System name is the name of the subdirectory containing pdb/top files
            system_name = pdb_file.parent.name
            copy_to_file = f"{system_name}__{pdb_file.name}"

        local_pdb_file = shutil.copy(pdb_file, self.workdir.joinpath(copy_to_file))
        return local_pdb_file

    def _copy_top_file(self, initial_pdb_dir: Path) -> str:
        top_file = get_topology(initial_pdb_dir, Path(self.pdb_file))
        copy_to_file = top_file.name
        local_top_file = shutil.copy(top_file, self.workdir.joinpath(copy_to_file))
        return local_top_file

    def move_results(self):
        shutil.move(self.workdir.as_posix(), self.result_dir)


def run_simulation(
    pdb_file: str,
    reference_pdb_file: str,
    omm_dir_prefix: str,
    local_run_dir: str,
    gpu_index: int,
    solvent_type: str,
    report_interval_ps: float,
    simulation_length_ns: float,
    dt_ps: float,
    temperature_kelvin: float,
    result_dir: str,
    initial_pdb_dir: str,
    wrap: bool,
):

    # Handle files
    ctx = SimulationContext(
        pdb_file=pdb_file,
        reference_pdb_file=reference_pdb_file,
        omm_dir_prefix=omm_dir_prefix,
        omm_parent_dir=local_run_dir,
        result_dir=result_dir,
        initial_pdb_dir=initial_pdb_dir,
        solvent_type=solvent_type,
    )

    # Create openmm simulation object
    sim = configure_simulation(
        ctx=ctx,
        gpu_index=gpu_index,
        solvent_type=solvent_type,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
    )

    # Write all frames to a single HDF5 file
    frames_per_h5 = int(simulation_length_ns / report_interval_ps)

    configure_reporters(sim, ctx, report_interval_ps, dt_ps, frames_per_h5, wrap)

    # Number of steps to run each simulation
    nsteps = int(simulation_length_ns / dt_ps)

    sim.step(nsteps)

    # Move simulation data to persistent storage
    ctx.move_results()
