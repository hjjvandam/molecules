#!/bin/bash

# run tag
run_tag=${2}
data_dir="/gpfs/alpine/med110/proj-shared/tkurth/spike"
output_dir="/gpfs/alpine/med110/proj-shared/tkurth/runs/${run_tag}"
checkpoint=${3}

# checkpoint
cflag=""
if [ ! -z "${checkpoint}" ]; then
    cflag="-c ${checkpoint}"
fi

# create output dir
mkdir -p ${output_dir}

# important variables
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_PORT=29500
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
export WANDB_MODE=dryrun

# login to wandb
if [ "${RANK}" == "0" ]; then
    wandb login ${1}
fi

# determine gpu
enc_gpu=$(( ${LOCAL_RANK} ))
dec_gpu=$(( ${LOCAL_RANK} + 3 ))

#echo "REPORT: rank:${RANK}, local_rank:${LOCAL_RANK} enc:${enc_gpu} dec:${dec_gpu}"

# launch code
python -u ./example_vae.py \
       -i "${data_dir}/closed/spike_closed.h5" \
       -o ${output_dir} -m ${run_tag} ${cflag} \
       --wandb_project_name covid_dl \
       --amp --distributed \
       -f sparse-concat \
       -t resnet \
       -e 150 \
       -b 2 \
       -E ${enc_gpu} -D ${dec_gpu} \
       -S 3 \
       -h 3768 -w 3768 -d 256
