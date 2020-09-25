#!/bin/bash

cmd_params=$@

conda_path='/gpfs/alpine/proj-shared/med110/atrifan/scripts/pytorch-1.6.0_cudnn-8.0.2.39_nccl-2.7.8-1_static_mlperf'
script_path='/gpfs/alpine/world-shared/ven201/tkurth/molecules/examples/outlier_detection/optics.py'

# important variables
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
export RANK=${OMPI_COMM_WORLD_RANK}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

# launch code
#cmd="${conda_path}/bin/python -u ${script_path} \

#cmd_params=${cmd_params}" \"--device\" \"cuda:"$LOCAL_RANK"\""
echo ${cmd_params}
($cmd_params)
