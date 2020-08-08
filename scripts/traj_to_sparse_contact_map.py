import os
import glob
import click
from molecules.sim.contact_maps import sparse_contact_maps_from_traj

@click.command()

@click.option('-p', 'pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing PDB file')

@click.option('-r', 'ref_pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing reference state PDB file')

@click.option('-t', 'traj_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing MD trajectory (DCD) ' \
                   'OR directory containing many .dcd files. If ' \
                   'directory, files are sorted lexographically ' \
                   'by their names and then concatenated.')

@click.option('-o', 'out_path', required=True,
              help='Path to file to write sparse contact matrices to')

@click.option('-v', '--verbose', is_flag=True)

def main(pdb_path, ref_pdb_path, traj_path, out_path, verbose):

    if os.path.isdir(traj_path):
        traj_path = sorted(glob.glob(os.path.join(traj_path, '*.dcd')))

    sparse_contact_maps_from_traj(pdb_path, ref_pdb_path,
                                  traj_path, save_file=out_path,
                                  verbose=verbose)

if __name__ == '__main__':
    main()

