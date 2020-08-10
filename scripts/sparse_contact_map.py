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
    contact_maps = np.array(f['contact_maps'][:])
    f.close()

    sparse_contact_maps_from_matrices(contact_maps, out_path)


if __name__ == '__main__':
    main()

