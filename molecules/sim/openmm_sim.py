import random
import parmed
import simtk.unit as u
import simtk.openmm as omm
import simtk.openmm.app as app


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


def configure_simulation(
    pdb_file, top_file, solvent_type, gpu_index, dt_ps, temperature_kelvin
):
    # Configure hardware
    try:
        platform = omm.Platform_getPlatformByName("CUDA")
        platform_properties = {"DeviceIndex": str(gpu_index), "CudaPrecision": "mixed"}
    except Exception:
        platform = omm.Platform_getPlatformByName("OpenCL")
        platform_properties = {"DeviceIndex": str(gpu_index)}

    # Select implicit or explicit solvent
    args = (
        pdb_file,
        top_file,
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
