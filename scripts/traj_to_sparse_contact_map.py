import os
import glob
import click
from molecules.sim.contact_maps import (sparse_contact_maps_from_traj,
                                        parallel_traj_to_dset)

@click.command()

@click.option('-p', 'pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing PDB file')

@click.option('-r', 'ref_pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing reference state PDB file')

@click.option('-t', 'traj_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing MD trajectory ' \
                   'OR directory containing many traj files. If ' \
                   'directory, files are sorted lexographically ' \
                   'by their names and then concatenated.')

@click.option('-e', '--ext', default='dcd',
              help='Trajectory file extension')

@click.option('-o', 'out_path', required=True,
              help='Path to file to write sparse contact matrices to')

@click.option('-P', '--parallel', is_flag=True)

@click.option('-v', '--verbose', is_flag=True)

def main(pdb_path, ref_pdb_path, traj_path, ext, out_path, parallel, verbose):

    if os.path.isdir(traj_path):
        traj_path = sorted(glob.glob(os.path.join(traj_path, f'*.{ext}')))

        if verbose:
            print(f'Collected {len(traj_path)} {ext} files')

    if parallel:
        assert isinstance(traj_path, list)
        parallel_traj_to_dset(pdb_path, ref_pdb_path,
                              out_path, traj_path,
                              verbose=verbose)
    else:
         sparse_contact_maps_from_traj(pdb_path, ref_pdb_path,
                                  traj_path, save_file=out_path,
                                  verbose=verbose)


if __name__ == '__main__':
    main()

