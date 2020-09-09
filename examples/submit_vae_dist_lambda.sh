#!/bin/bash
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -A lambda
#SBATCH -J test_ddp
#SBATCH -t 00:10:00
#ranks per node
set -x
rankspernode=4
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
#parameters
run_tag="fspeptide-test-${SLURM_NNODES}_run-2"
data_dir_prefix="data"
wandb_token=${WANDB_TOKEN}
output_dir="runs"
#checkpoint="/runs/spike-cmaps_nnodes-1_run-3/model-spike-cmaps_nnodes-1_run-3/checkpoint/epoch-71-20200901-173533.pt"
#run training
srun -u --wait=30 -N ${SLURM_NNODES} -n ${totalranks} -c $(( 40 / ${rankspernode} )) --cpu_bind=cores --mpi=pmi2 \
     molecules/examples/run_vae_dist_lambda.sh ${wandb_token} ${run_tag} ${checkpoint}
