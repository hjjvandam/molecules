#!/bin/bash

# set to dev 1
export CUDA_VISIBLE_DEVICES=0

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

# spike data
#spike_data="/data/spike-for-real/closed/spike-full-point-cloud_closed.h5"
spike_data="/data/spike-for-real/closed/spike_closed.h5"

# checkpoint

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29045

mpirun -np 1 --allow-run-as-root \
       python example_aae.py -i ${spike_data} \
       -dn "point_cloud" \
       -rn "rmsd" \
       --resume \
       -o /data/runs/ -m spike-closed-correct-1-seq-embedding --wandb_project_name covid_dl \
       --encoder_kernel_sizes 5 3 3 1 1 \
       -np 3768 -nf 0 \
       -E 0 -G 0 -D 0 \
       -e 200 -b 16 \
       -opt "name=Adam,lr=0.0001" \
       -d 256 \
       -lw "lambda_rec=0.5,lambda_gp=10." \
       -S 3
