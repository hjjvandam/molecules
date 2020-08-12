import click
import numpy as np
from molecules.utils import open_h5
from molecules.sim.contact_maps import sparse_contact_maps_from_matrices

@click.command()
@click.option('-i', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-o', 'out_path', required=True,
              help='Path to file to write sparse contact matrices to')

def main(input_path, out_path):

    f = open_h5(input_path)
    contact_maps = f['contact_maps'][...]
    if 'rmsd' in f.keys():
        rmsd = f['rmsd'][..., 2]
    else:
        rmsd = None
    f.close()

    # TODO: currently dummy data
    fnc = np.ones(len(contact_maps), dtype=np.float32)

    sparse_contact_maps_from_matrices(contact_maps, rmsd=rmsd,
                                      fnc=fnc, save_file=out_path)


if __name__ == '__main__':
    main()

