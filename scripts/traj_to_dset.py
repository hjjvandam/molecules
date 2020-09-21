import os
import glob
import click
from molecules.sim.dataset import traj_to_dset

def parse_dict(ctx, param, value):
    if value is not None:
        token = value.split(",")
        result = {}
        for item in token:
            k, v = item.split("=")
            result[k] = v
        return result

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

@click.option('-e', '--pattern', default='*.dcd',
              help='Trajectory file pattern.')

@click.option('-o', '--out_path', required=True,
              help='Path to file to write dataset to.')

@click.option('-w', '--num_workers', default=None, type=int,
              help='Number of parallel workers for processing multiple ' \
                   'traj files in parallel. Defaults to the smaller of ' \
                   'number of cpus on machine or the number of traj files.')

@click.option('-s', '--selection', default='protein and name CA',
              help='Atom selection for creating contact maps, ' \
                    'point clouds and computing fnc.')

@click.option('--rmsd', is_flag=True,
              help='Computes and saves RMSD.')

@click.option('--fnc', is_flag=True,
              help='Computes and saves fraction of reference contacts.')

@click.option('--contact_map', is_flag=True,
              help='Computes and saves contact maps.')

@click.option('-cmp', '--contact_maps_parameters', callback=parse_dict,
              default="kernel_type=threshold,threshold=8.0",
              help='Kernel type parameters for contact maps. Only relevant if contact maps are computed')

@click.option('--point_cloud', is_flag=True,
              help='Computes and saves point cloud.')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Options: [sparse-concat, sparse-rowcol]. Refers to format the ' \
                   'the contact maps are stored as. latest gives single ' \
                   'dset format while oldest gives group with row,col dset.')

@click.option('-v', '--verbose', is_flag=True)

def main(pdb_path, ref_pdb_path, traj_path, pattern, out_path, num_workers,
         selection, contact_maps_parameters, rmsd, fnc, contact_map, point_cloud, cm_format, verbose):

    if os.path.isdir(traj_path):
        traj_path = sorted(glob.glob(os.path.join(traj_path, pattern)))

        if not traj_path:
            raise ValueError('No traj files found in directory matching ' \
                             f'the pattern {pattern}')

        if verbose:
            print(f'Collected {len(traj_path)} traj files')

    traj_to_dset(topology=pdb_path, ref_topology=ref_pdb_path, traj_files=traj_path,
                 save_file=out_path, rmsd=rmsd, fnc=fnc, point_cloud=point_cloud,
                 contact_map=contact_map, distance_kernel_params=contact_maps_parameters,
                 sel=selection, cm_format=cm_format, num_workers=num_workers, verbose=verbose)
    
if __name__ == '__main__':
    main()

