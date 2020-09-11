#!/bin/bash
#set -Eeuo pipefail
set -x
# run tag
run_tag=${2}
output_dir="runs/${run_tag}"
checkpoint=${3}

cflag=""
if [ ! -z "${checkpoint}" ]; then
    cflag="-c ${checkpoint}"
fi

# login to wandb
if [ "${RANK}" == "0" ]; then
    wandb login ${1}
fi

# create output dir
mkdir -p ${output_dir}

# SLURM settings
export WORLD_SIZE=$(( ${SLURM_TASKS_PER_NODE} * ${SLURM_NNODES} ))
export RANK=${SLURM_PROCID}
#export LOCAL_RANK=$(( ${SLURM_PROCID} % ${SLURM_TASKS_PER_NODE} ))
export LOCAL_RANK=$(( ${SLURM_PROCID} % $(echo ${SLURM_TASKS_PER_NODE} |awk '{split($1,a,"("); print a[1]}') ))
export MASTER_PORT=29500
export MASTER_ADDR=${SLURM_LAUNCH_NODE_IPADDR}
export WANDB_MODE=dryrun

echo ${SLURM_PROCID} ${SLURM_TASKS_PER_NODE} ${LOCAL_RANK}

# determine gpu
enc_gpu=$(( 2 * ${LOCAL_RANK} ))
dec_gpu=$(( 2 * ${LOCAL_RANK} + 1 ))

# launch code
python molecules/examples/example_vae.py \
       -i "data/fs-peptide/fspep-sparse-rowcol.h5" \
       -o ${output_dir} -m ${run_tag} ${cflag} \
       -dn contact_maps \
       --wandb_project_name molecules-debug \
       --amp --distributed \
       -opt "name=Adam,lr=1e-4" \
       -f sparse-rowcol \
       -t resnet \
       -e 10 \
       -ti 2 \
       -b 128 \
       -E ${enc_gpu} -D ${dec_gpu} \
       -S 8 \
       -h 22 -w 22 -d 11
