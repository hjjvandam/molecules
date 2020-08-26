import os
import glob
import click
import h5py
import numpy as np

@click.command()

@click.option('-d', '--data_root', required=True,
              type=click.Path(exists=True),
              help='Directory path containing individual h5 files.')

@click.option('-p', '--pattern', default='*.h5',
              help='File name pattern to glob particular h5 files. ' \
                   'Wildcards allowed. EX: *_traj[0-10].h5')

@click.option('-o', '--out_path', required=True,
              help='Path to file to write concatenated dataset to.')

@click.option('--rmsd', is_flag=True,
              help='Concatenates RMSD data field in new dataset.')

@click.option('--fnc', is_flag=True,
              help='Concatenates fraction of reference contacts data field in new dataset.')

@click.option('--contact_map', is_flag=True,
              help='Concatenates contact maps data field in new dataset. ' \
                   'Note: only works for sparse-concat formatted contact maps.')

@click.option('--point_cloud', is_flag=True,
              help='Concatenates point cloud data field in new dataset.')

@click.option('-v', '--verbose', is_flag=True)

def main(data_root, pattern, out_path, rmsd, fnc, contact_map, point_cloud, verbose):

    fields = []
    if rmsd: fields.append('rmsd')
    if fnc: fields.append('fnc')
    if contact_map: fields.append('contact_map')
    if point_cloud: fields.append('point_cloud')

    if not fields:
        raise ValueError('No data fields selected to concatenate. Add any ' \
                         ' combination of --rmsd --fnc --contact_map --point_cloud')

    # Get list of input h5 files
    files = sorted(glob.glob(os.path.join(data_root, pattern)))
    files = list(filter(lambda x: x != out_path, files))

    if verbose:
        print(f'Collected {len(files)} h5 files.')

    # Open output file
    fout = h5py.File(out_path, 'w', libver='latest')

    # Initialize data buffers
    data = {x: [] for x in fields}

    for in_file in files:

        if verbose:
            print('Reading', in_file)

        with h5py.File(in_file, 'r', libver='latest', driver='core', backing_store=False) as fin:
            for field in fields:
                data[field].append(fin[field][...])

    # Concatenate data
    for field in data:
        data[field] = np.concatenate(data[field])

    # Centor of mass (CMS) subtraction
    if 'point_cloud' in data:
        if verbose:
            print('Subtract center of mass (CMS) from point cloud')
        cms = np.mean(data['point_cloud'][:, 0:3, :].astype(np.float128), axis=2, keepdims=True).astype(np.float32)
        data['point_cloud'][:, 0:3, :] -= cms

    # Create new dsets from concatenated dataset
    for field, concat_dset in data.items():
        shape = concat_dset.shape
        chunkshape = ((1,) + shape[1:])
        # Create dataset
        if concat_dset.dtype != np.object:
            if np.any(np.isnan(concat_dset)):
                raise ValueError('NaN detected in concat_dset.')
            dset = fout.create_dataset(field, shape, chunks=chunkshape, dtype=concat_dset.dtype)
        else:
            dset = fout.create_dataset(field, shape, chunks=chunkshape, dtype=h5py.vlen_dtype(np.int16))
        # write data
        dset[...] = concat_dset[...]

    # Clean up
    fout.flush()
    fout.close()

if __name__ == '__main__':
    main()

