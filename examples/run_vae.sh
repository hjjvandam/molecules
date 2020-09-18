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
#checkpoint="/data/runs/checkpoint_test/epoch-50-20200828-145839.pt"

python example_vae.py -i /data/3clpro/3clpro-monomer.h5 \
       -a \
       -f sparse-concat \
       -t resnet \
       -o /data/runs/ -m cmaps-3clpro \
       --wandb_project_name covid_dl \
       -opt "name=Adam,lr=1e-3" \
       -e 150 \
       -b 256 \
       -E 0 -D 0 \
       -S 10 \
       -h 301 -w 301 -d 38
