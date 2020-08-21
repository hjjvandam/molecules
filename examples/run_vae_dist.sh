#!/bin/bash

# run tag
run_tag="spike-cmaps_nnodes-${SLURM_NNODES}_run-1"
output_dir="/runs/${run_tag}"

# connect to wandb
wandb login ${1}

# create output dir
mkdir -p ${output_dir}

# launch code
python ./example_vae.py \
       -i "/data/spike-full-point-cloud_closed.h5" \
       -o ${output_dir} -m ${run_tag} \
       --wandb_project_name covid_dl \
       --amp --distributed \
       -f sparse-concat \
       -t resnet \
       -e 150 \
       -b 2 \
       -E 0 -D 0 \
       -S 3 \
       -h 3768 -w 3768 -d 471
