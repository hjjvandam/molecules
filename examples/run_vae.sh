#!/bin/bash

# use gpu 1
export CUDA_VISIBLE_DEVICES=1,2

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# checkpoint
#checkpoint="/data/runs/model-spike-cmaps-merge-debug/checkpoint/epoch-4-20200831-220400.pt"
checkpoint="/data/runs/checkpoint_test/epoch-50-20200828-145839.pt"

python example_vae.py -i /data/spike-for-real/closed/spike_closed.h5 \
       -c ${checkpoint} \
       -a --distributed \
       -f sparse-concat \
       -t resnet \
       -o /data/runs/ -m spike-cmaps-merge-debug \
       --wandb_project_name covid_dl \
       -opt "name=RMSprop,lr=1e-3" \
       -e 150 \
       -b 4 \
       -E 0 -D 1 \
       -S 3 \
       -h 3768 -w 3768 -d 256
