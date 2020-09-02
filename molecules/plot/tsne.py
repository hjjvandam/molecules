import os
import time
import wandb
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from molecules.utils import open_h5

def pca(embeddings, dim=50):
    # TODO: use pca to drop embeddings to dim 50
    # TODO: run PCA in pytorch and reduce dimension down to 50 (maybe even lower)
        #       then run tSNE on outputs of PCA. This works for sparse matrices
        #       https://pytorch.org/docs/master/generated/torch.pca_lowrank.html
    return embeddings

def _load_data(embeddings_path, colors, embeddings_dset='embeddings'):
    color_arrays = {}
    with open_h5(embeddings_path) as f:
        # Load embeddings from h5 file
        embeddings = f[embeddings_dset][...]
        # May contain rmsd, fnc
        for color in colors:
            color_arrays[color] = f[color][...]

    return embeddings, color_arrays

def plot_tsne(embeddings_path, out_dir='./', colors=['rmsd'],
              pca=True, projection_type='2d',
              target_perplexity=30,
              perplexities=[5, 30, 50, 100, 200],
              pca_dim=50,
              wandb_config=None,
              global_step=0, epoch=1):
    
    embeddings, color_arrays = _load_data(embeddings_path, colors)

    if pca and embeddings.shape[1] > pca_dim:
        embeddings = pca(embeddings, pca_dim)

    # create plot grid
    nrows = len(perplexities)
    ncols = 3 if projection_type == '3d' else 1

    for color_name, color_arr in color_arrays.items():

        # create colormaps
        cmi = plt.get_cmap('jet')
        vmin, vmax = np.min(color_arr), np.max(color_arr)
        cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmi)
        scalar_map.set_array(color_arr)

        # create figure
        fig, axs = plt.subplots(figsize=(ncols * 4, nrows * 4),
                                nrows=nrows, ncols=ncols)

        # set up constants
        color = scalar_map.to_rgba(color_arr)
        if color_name == 'rmsd':
            titlestring = f'RMSD to reference state after epoch {epoch}'
        elif color_name == 'fnc':
            titlestring = f'Fraction of contacts to reference state after epoch {epoch}'

        for idr, perplexity in enumerate(perplexities):
        
            # Outputs 3D embeddings using all available processors
            tsne = TSNE(n_components=int(projection_type[0]), n_jobs=-1, perplexity=perplexity)

            emb_trans = tsne.fit_transform(embeddings)

            # plot            
            if projection_type == '3d':
                z1, z2, z3 = emb_trans[:, 0], emb_trans[:, 1], emb_trans[:, 2]
                z1mm = np.min(z1), np.max(z1)
                z2mm = np.min(z2), np.max(z2)
                z3mm = np.min(z3), np.max(z3)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                z3mm = (z3mm[0] * 0.95, z3mm[1] * 1.05)
                # x-y
                ax1 = axs[idr, 0]
                ax1.scatter(z1, z2, marker='.', c=color)
                ax1.set_xlim(z1mm)
                ax1.set_ylim(z2mm)
                ax1.set_xlabel(r'$z_1$')
                ax1.set_ylabel(r'$z_2$')
                # x-z
                ax2 = axs[idr, 1]
                ax2.scatter(z1, z3, marker='.', c=color)
                ax2.set_xlim(z1mm)
                ax2.set_ylim(z3mm)
                ax2.set_xlabel(r'$z_1$')
                ax2.set_ylabel(r'$z_3$')
                if idr == 0:
                    ax2.set_title(titlestring)
                # y-z
                ax3 = axs[idr, 2]
                ax3.scatter(z2, z3, marker='.', c=color)
                ax3.set_xlim(z2mm)
                ax3.set_ylim(z3mm)
                ax3.set_xlabel(r'$z_2$')
                ax3.set_ylabel(r'$z_3$')
                # colorbar
                divider = make_axes_locatable(axs[idr, 2])
                cax = divider.append_axes('right', size='5%', pad=0.1)
                fig.colorbar(scalar_map, ax=axs[idr, 2], cax=cax)
            
            else:
                ax = axs[idr]
                z1, z2 = emb_trans[:, 0], emb_trans[:, 1]
                ax.scatter(z1, z2, marker='.', c=color)
                z1mm = np.min(z1), np.max(z1)
                z2mm = np.min(z2), np.max(z2)
                z1mm = (z1mm[0] * 0.95, z1mm[1] * 1.05)
                z2mm = (z2mm[0] * 0.95, z2mm[1] * 1.05)
                ax.set_xlim(z1mm)
                ax.set_ylim(z2mm)
                ax.set_xlabel(r'$z_1$')
                ax.set_ylabel(r'$z_2$')
                if idr == 0:
                    ax.set_title(titlestring)
                # colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.1)
                fig.colorbar(scalar_map, ax=axs, cax=cax)

            # plot as 3D object on wandb
            if (wandb_config is not None) and (perplexity == target_perplexity):
                point_data = np.concatenate([emb_trans, color[:,:3] * 255.], axis=1)
                wandb.log({'step t-SNE embeddings 3D': wandb.Object3D(point_data, caption=f'perplexity {perplexity}')}, step=global_step)

        # tight layout
        plt.tight_layout()

        # save figure
        time_stamp = time.strftime(f'2d-embeddings-{color_name}-step-{global_step}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(out_dir, time_stamp), dpi=300)

        # wandb logging
        if wandb_config is not None:
            img = Image.open(os.path.join(out_dir, time_stamp))
            wandb.log({'step 2D t-SNE embeddings': [wandb.Image(img, caption='Latent Space Visualizations')]}, step=global_step)

        # close plot
        plt.close(fig)


def plot_tsne_publication(embeddings_path, out_dir='./', colors=['rmsd'],
                          pca=False, wandb_config=None,
                          tensorboard_writer=None,
                          global_step=0, epoch=1):
    """Generate publication quality 3d t-SNE plot."""

    embeddings, color_arrays = _load_data(embeddings_path, colors)

    if pca and embeddings.shape[1] > 50:
        embeddings = pca(embeddings)

    # Outputs 3D embeddings using all available processors
    tsne = TSNE(n_components=3, n_jobs=-1)

    embeddings = tsne.fit_transform(embeddings)

    z1, z2, z3 = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    z1_min_max = np.min(z1), np.max(z1)
    z2_min_max = np.min(z2), np.max(z2)
    z3_min_max = np.min(z3), np.max(z3)

    # TODO: make grid plot of fnc, rmsd
    for color_name, color_arr in color_arrays.items():

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cnorm = matplotlib.colors.Normalize(vmin=np.min(color_arr),
                                            vmax=np.max(color_arr))
        scalar_map = matplotlib.cm.ScalarMappable(norm=cnorm,
                                                  cmap=plt.get_cmap('jet'))
        scalar_map.set_array(color_arr)
        fig.colorbar(scalar_map)
        color = scalar_map.to_rgba(color_arr)

        ax.scatter3D(z1, z2, z3, marker='.', c=color)
        ax.set_xlim3d(z1_min_max)
        ax.set_ylim3d(z2_min_max)
        ax.set_zlim3d(z3_min_max)
        ax.set_xlabel(r'$z_1$')
        ax.set_ylabel(r'$z_2$')
        ax.set_zlabel(r'$z_3$')

        if color_name == 'rmsd':
            ax.set_title(f'RMSD to reference state after epoch {epoch}')
        elif color_name == 'fnc':
            ax.set_title(f'Fraction of contacts to reference state after epoch {epoch}')

        time_stamp = time.strftime(f'3d-embeddings-{color_name}-step-{global_step}-%Y%m%d-%H%M%S.png')
        plt.savefig(os.path.join(out_dir, time_stamp), dpi=300)

        if tensorboard_writer is not None:
            tensorboard_writer.add_figure('epoch 3D t-SNE embeddings', fig, global_step)

        if wandb_config is not None:
            img = Image.open(os.path.join(out_dir, time_stamp))
            wandb.log({'step 3D t-SNE embeddings': [wandb.Image(img, caption='Latent Space Visualizations')]}, step=global_step)

        ax.clear()
        plt.close(fig)

if __name__ == '__main__':
    # data1 = np.random.normal(size=(100, 6))
    # data2 = np.random.normal(size=(100, 6), loc=10, scale=2)
    # data3 = np.random.normal(size=(100, 6), loc=5, scale=1)
    # data = np.concatenate((data1, data2, data3))
    # rmsd = np.random.normal(size=300)
    # fnc = np.random.normal(size=300)

    # scaler_kwargs = {'fletcher32': True}
    # with open_h5('test_embed.h5', 'w', swmr=False) as h5_file:
    #     h5_file.create_dataset('embeddings', data=data, **scaler_kwargs)
    #     h5_file.create_dataset('rmsd', data=rmsd, **scaler_kwargs)
    #     h5_file.create_dataset('fnc', data=fnc, **scaler_kwargs)


    plot_tsne('test_embed.h5', './tmpdir', '3d', pca=False, colors=['fnc'])
    #plot_tsne('test_embed.h5', './tmpdir', '2d', pca=False, colors=['rmsd', 'fnc'], projection_type='3d_project')