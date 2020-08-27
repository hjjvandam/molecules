#!/bin/bash

# run tag
run_tag=${2}
output_dir="/runs/${run_tag}"
checkpoint=${3}

# checkpoint
cflag=""
if [ ! -z "${checkpoint}" ]; then
    cflag="-c ${checkpoint}"
fi

# create output dir
mkdir -p ${output_dir}

# determine local rank
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_ADDR=${}
export MASTER_PORT=29500

# login to wandb
if [ "${RANK}" == "0" ]; do
    wandb login ${1}
fi

# determine gpu
enc_gpu=$(( ${LOCAL_RANK} ))
dec_gpu=$(( ${LOCAL_RANK} + 3 ))

# launch code
python ./example_vae.py \
       -i "/data/spike-full-point-cloud_closed.h5" \
       -o ${output_dir} -m ${run_tag} ${cflag} \
       --wandb_project_name covid_dl \
       --amp --distributed \
       -f sparse-concat \
       -t resnet \
       -e 150 \
       -b 12 \
       -E ${enc_gpu} -D ${dec_gpu} \
       -S 3 \
       -h 3768 -w 3768 -d 256
