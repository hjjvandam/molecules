import yaml
import argparse
import simtk.unit as u
from molecules.sim.openmm_sim import run_simulation


def get_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="YAML config file", required=True)
    path = parser.parse_args().config
    with open(path) as fp:
        config = yaml.safe_load(fp)
    return config


def build_simulation_params(cfg: dict) -> dict:
    return dict(
        pdb_file=cfg["pdb_file"],
        reference_pdb_file=cfg["reference_pdb_file"],
        omm_dir_prefix=cfg["omm_dir_prefix"],
        local_run_dir=cfg["local_run_dir"],
        gpu_index=0,
        solvent_type=cfg["solvent_type"],
        report_interval_ps=float(cfg["report_interval_ps"]) * u.picoseconds,
        frames_per_h5=cfg["frames_per_h5"],
        simulation_length_ns=float(cfg["simulation_length_ns"]) * u.nanoseconds,
        dt_ps=float(cfg["dt_ps"]) * u.picoseconds,
        temperature_kelvin=float(cfg["temperature_kelvin"]) * u.kelvin,
        result_dir=cfg["result_dir"],
        initial_pdb_dir=cfg["initial_pdb_dir"],
        wrap=cfg["wrap"],
    )


if __name__ == "__main__":
    cfg = get_config()
    simulation_kwargs = build_simulation_params(cfg)
    run_simulation(**simulation_kwargs)
