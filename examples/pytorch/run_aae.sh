#!/bin/bash

python example_aae.py -i /data/test/point_clouds_fs.h5 \
       -o /data/runs/ -m test-point \
       -np 22 -nf 0 \
       -E 0 -G 0 \
       -e 10 -b 10 \
       -d 256 \
       -lrec 1. \
       -lgp 10.
