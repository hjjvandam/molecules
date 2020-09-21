#!/bin/bash

# set to dev 1
export CUDA_VISIBLE_DEVICES=1

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

# spike data
#spike_data="/data/spike-for-real/open/spike_open.h5"
data="/data/3clpro/3clpro-monomer-cutoff-16.h5"
#spike_data="/data/spike-for-real/closed/spike_closed.h5"

# checkpoint

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29045

mpirun -np 1 --allow-run-as-root \
       python example_aae.py -i ${data} \
       -dn "point_cloud" \
       -rn "rmsd" \
       --resume \
       -o /data/runs/ -m aae-3clpro-16A-1 --wandb_project_name covid_dl \
       --encoder_kernel_sizes 5 3 3 1 1 \
       -np 301 -nf 0 \
       -E 0 -G 0 -D 0 \
       -e 200 -b 32 \
       -opt "name=Adam,lr=0.0001" \
       -d 64 \
       -lw "lambda_rec=0.5,lambda_gp=10." \
       -S 3
