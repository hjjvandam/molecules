#!/bin/bash
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH -A lambda
#SBATCH -J spike_hpo
#SBATCH -t 00:30:00
#ranks per node
set -x
rankspernode=4
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
#parameters
run_tag="aae-ddp-test-${SLURM_NNODES}_bs-128_opt_RMSprop_lr-1e-4_cutoff-8"
data_dir_prefix="data"
wandb_token=${WANDB_TOKEN}
output_dir="runs"
#checkpoint="/runs/spike-cmaps_nnodes-1_run-3/model-spike-cmaps_nnodes-1_run-3/checkpoint/epoch-71-20200901-173533.pt"
#run training
srun -l -u --wait=30 -N ${SLURM_NNODES} -n ${totalranks} -c $(( 40 / ${rankspernode} )) --cpu_bind=cores --mpi=pmi2 \
    /homes/abrace/molecules/examples/bin/lambda/run_aae_dist_lambda.sh ${wandb_token} ${run_tag} ${checkpoint}

