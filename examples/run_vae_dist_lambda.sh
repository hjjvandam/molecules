#!/bin/bash

# run tag
run_tag=${2}
output_dir="/runs/${run_tag}"
checkpoint=${3}

cflag=""
if [ ! -z "${checkpoint}" ]; then
    cflag="-c ${checkpoint}"
fi

# connect to wandb
wandb login ${1}

# create output dir
mkdir -p ${output_dir}

# MPI settings
#export WORLD_SIZE=${PMI_SIZE}
#export RANK=${PMI_RANK}
#export LOCAL_RANK=${MPI_LOCALRANKID}
#export MASTER_PORT=29500
#export MASTER_ADDR=127.0.0.1

# SLURM settings
export WORLD_SIZE=$(( ${SLURM_TASKS_PER_NODE} * ${SLURM_NNODES} ))
export RANK=${SLURM_PROCID}
export LOCAL_RANK=$(( ${SLURM_PROCID} % ${SLURM_TASKS_PER_NODE} ))
export MASTER_PORT=29500
export MASTER_ADDR=${SLURM_LAUNCH_NODE_IPADDR}

# determine gpu
enc_gpu=$(( 2 * ${LOCAL_RANK} ))
dec_gpu=$(( 2 * ${LOCAL_RANK} + 1 ))

# launch code
python ./example_vae.py \
       -i "data/fs-peptide/fspep-sparse-rowcol.h5" \
       -o ${output_dir} -m ${run_tag} ${cflag} \
       --wandb_project_name molecules-debug \
       --amp --distributed \
       -f sparse-rowcol \
       -t resnet \
       -e 2 \
       -b 128 \
       -E ${enc_gpu} -D ${dec_gpu} \
       -S 8 \
       -h 22 -w 22 -d 11
