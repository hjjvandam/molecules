#!/bin/bash

# connect to wandb
wandb login 6c8b9db0b520487f05d32ebc76fcea156bd85d58

python example_aae.py -i /data/test/small_data.h5 \
       -dn point_clouds \
       -rn rmsd \
       -o /data/runs/ -m test-point-4 --wandb_project_name covid_dl \
       -np 21 -nf 0 \
       -E 0 -G 0 -D 0 \
       -e 10 -b 200 \
       -d 256 \
       -lrec 1. \
       -lgp 10. \
       -ndw 4
