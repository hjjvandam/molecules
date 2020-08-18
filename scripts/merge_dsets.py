import os
import glob
import click
import h5py
import numpy as np


@click.command()

@click.option('-p', 'pdb_path', required=True,
              type=click.Path(exists=True),
              help='Path to file containing PDB file')

def main():

    # data root                                                                                                                                                                                  
    data_root = '/raid/tkurth/covid_data/spike-for-real/closed'
    # output file                                                                                                                                                                                
    outputfile = os.path.join(data_root, 'spike-full-point-cloud_closed.h5')
    # get list                                                                                                                                                                                   
    filelist = sorted(glob.glob(data_root + '/*_rep[0-9].h5'))
    filelist = [x for x in filelist if x != outputfile]
    # open file                                                                                                                                                                                  
    fout = h5py.File(outputfile, 'w', libver='latest')
    # get input                                                                                                                                                                                  
    fields = ["contact_map", "point_cloud", "rmsd", "fnc"]
    #fields = ["point_cloud", "rmsd", "fnc"]                                                                                                                                                     
    data = {x:[] for x in fields}
    for inputfile in filelist:
        print(inputfile)
        with h5py.File(inputfile, 'r', libver='latest', driver='core', backing_store=False) as fin:
            for field in fields:
                data[field].append(fin[field][...])
    # stack the stuff                                                                                                                                                                            
    for key in data:
        data[key] = np.concatenate(data[key], axis = 0)
    # do CMS subtract here:                                                                                                                                                                      
    print("Subtract CMS from point cloud")
    cms = np.mean(data["point_cloud"][:, 0:3, :].astype(np.float128), axis = 2, keepdims = True).astype(np.float32)
    data["point_cloud"][:, 0:3, :] -= cms
    # create datasets                                                                                                                                                                            
    for key in data:
        shape = data[key].shape
        chunkshape = ((1,) + shape[1:])
        # create dataset                                                                                                                                                                         
        if data[key].dtype != np.object:
            if np.any(np.isnan(data[key])):
                raise ValueError("Error, NaN detected in output.")
            dset = fout.create_dataset(key, shape, chunks=chunkshape, dtype=data[key].dtype)
        else:
            dset = fout.create_dataset(key, shape, chunks=chunkshape, dtype=h5py.vlen_dtype(np.int16))
        # write data                                                                                                                                                                             
        dset[...] = data[key][...]
    # clean up                                                                                                                                                                                   
    fout.close()
