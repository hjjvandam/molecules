#!/bin/bash

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

python example_vae.py -i /data/test/small_data.h5 \
       -t resnet \
       -o /data/runs/ -m test-cmaps-3 \
       --wandb_project_name covid_dl \
       -e 150 \
       -b 128 \
       -E 0 -D 0 \
       -h 22 -w 22 -d 11
