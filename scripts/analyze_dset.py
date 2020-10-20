import click

def plot_histogram(data, name):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    plt.hist(data, bins=50)
    plt.gca().set(title=name, ylabel='Frequency')
    plt.savefig(f'./{"_".join(name.split())}_plot.png')
    plt.close()

@click.command()
@click.option('-i', '--input', 'input_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing preprocessed contact matrix data')

@click.option('-h', '--dim1', required=True, type=int,
              help='H of (H,W) shaped contact matrix')

@click.option('-w', '--dim2', required=True, type=int,
              help='W of (H,W) shaped contact matrix')

@click.option('-f', '--cm_format', default='sparse-concat',
              help='Format of contact map files. Options ' \
                   '[full, sparse-concat, sparse-rowcol]')

@click.option('-dn', '--dataset_name', default='contact_map',
              help='Name of the dataset in the HDF5 file.')

def main(input_path, dim1, dim2, cm_format, dataset_name):
    import h5py as h5
    import numpy as np

    nelem = []
    f = h5.File(input_path, 'r')

    plot_histogram(f['rmsd'][...], 'RMSD to reference state')
    plot_histogram(f['fnc'][...], 'Fraction of reference contacts')

    if cm_format == 'sparse-concat':
        for item in f[dataset_name]:
            # Counts number of 1s since each contact map stores row,col positions
            nelem.append(len(item) / 2)
    elif cm_format == 'sparse-rowcol':
        print(f['rmsd'][...])
        rows = f[dataset_name]['col']
        print(rows)
        for i in range(len(rows)):
            print(rows[i, ...])
            nelem.append(len(row))
    elif cm_format == 'full':
        for item in f[dataset_name][...]:
            nelem.append(np.sum(item))

    sparsity = np.array(nelem) / (dim1 * dim2)
    
    print(f'Contacts: {np.mean(nelem)} +- {np.std(nelem)}')
    print(f'Sparsity: {np.mean(sparsity)} +- {np.std(sparsity)}')
    
    f.close()


if __name__ == "__main__":
    main()
