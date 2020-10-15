#!/bin/bash

# -i {sparse_matrix_path} -o ./ --model_id {cvae_dir} -f sparse-concat -t resnet --dim1 168 --dim2 168 -d 21 --amp --distributed -b {batch_size} -e {epoch} -S 3
data_path=${1}
output_dir=${2}
model_id=${3}
residues=${4}
latent_dim=${5}
amp=${6}
distributed=${7}
batch_size=${8}
epoch=${9}
sample_interval=${10}
optimizer=${11}
loss_weights=${12}
init_weights=${13}

if [ "$distributed" == "distributed" ]
then
	distributed='--distributed'
else
	distributed=''
fi

if [ "$amp" == "amp" ]
then
	amp="--amp"
else
	amp=""
fi

if [ "$init_weights" != "" ]
then
	init_weights="-iw $init_weights"
else
	init_weights=""
fi

# create output dir
#mkdir -p ${output_dir}

conda_path='/usr/workspace/cv_ddmd/conda/pytorch/'
script_path='/usr/workspace/cv_ddmd/lee1078/git/molecules/examples/example_aae.py'

# important variables
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_PORT=29500
export MASTER_ADDR=$(cat ${LSB_DJOB_HOSTFILE} | uniq | sort | grep -v batch | head -n1)
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
export WANDB_MODE=dryrun

# determine gpu
enc_gpu=$(( ${LOCAL_RANK} ))

#echo "REPORT: rank:${RANK}, local_rank:${LOCAL_RANK} enc:${enc_gpu}"

# launch code
cmd="${conda_path}/bin/python -u ${script_path} \
       -i ${data_path} \
       -o ${output_dir} \
       ${amp} ${distributed} \
       -m ${model_id} \
       -dn point_cloud \
       -rn rmsd \
       --encoder_kernel_sizes 5 3 3 1 1 \
       -nf 0 \
       -np ${residues} \
       -e ${epoch} \
       -b ${batch_size} \
       -E ${enc_gpu} -D ${enc_gpu} -G ${enc_gpu} \
       -opt ${optimizer} ${init_weights} \
       -lw ${loss_weights} \
       -S ${sample_interval} \
       -ti $(($epoch+1)) \
       -d ${latent_dim} \
       --num_data_workers 0" # -ndw
echo ${cmd}
($cmd)
