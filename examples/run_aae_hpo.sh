#!/bin/bash

# set to dev 1
export CUDA_VISIBLE_DEVICES=0

# spike data
#spike_data="/data/spike-for-real/closed/spike-full-point-cloud_closed.h5"
spike_data="/data/spike-for-real/closed/spike_closed.h5"


# run the hpo
python hpo_aae.py -i ${spike_data} \
       -dn "point_cloud" \
       -rn "rmsd" \
       -o /data/runs/ -m spike-closed-hpo \
       --wandb_project_name covid_dl \
       --wandb_api_key 6c8b9db0b520487f05d32ebc76fcea156bd85d58 \
       -np 3768 -nf 0 \
       -E 0 -G 0 -D 0 \
       -e 200 \
       -lw "lambda_rec=0.5,lambda_gp=10." \
       -S 3
