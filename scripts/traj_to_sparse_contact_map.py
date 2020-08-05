import click
import numpy as np
from molecules.utils import open_h5
from molecules.sim.contact_maps import sparse_contact_maps_from_traj

@click.command()

@click.option('-p', 'pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing PDB file')

@click.option('-n', 'native_pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing native state PDB file')

@click.option('-t', 'traj_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing MD trajectory (DCD)')

@click.option('-o', 'out_path', required=True,
              help='Path to file to write sparse contact matrices to')

@click.option('-v', '--verbose', is_flag=True)

def main(pdb_path, native_pdb_path, traj_path, out_path, verbose):

    sparse_contact_maps_from_traj(pdb_path, native_pdb_path,
                                  traj_path, save_file=out_path,
                                  verbose=verbose)

if __name__ == '__main__':
    main()

