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
python /homes/abrace/molecules/examples/example_aae.py \
       -i 3clpro-residues-303-cutoff-16.h5 \
       -dn point_cloud \
       -rn rmsd \
       -o ${output_dir} -m ${run_tag} ${cflag} \
       -wp molecules-debug \
       --encoder_kernel_sizes 5 3 3 1 1 \
       -nf 0 \
       -opt "name=Adam,lr=1e-4" \
       -e 20 \
       -ti 5 \
       -b 32 \
       -E ${enc_gpu} -D ${enc_gpu} -G ${enc_gpu} \
       -S 10 \
       -np 303 -d 48 \
       -lw "lambda_rec=0.5,lambda_gp=10" \
       --distributed

