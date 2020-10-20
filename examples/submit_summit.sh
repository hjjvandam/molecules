#!/bin/bash
#BSUB -W 120
#BSUB -P MED110
#BSUB -J spike_train
#BSUB -alloc_flags "NVME"

# Load modules
module load gcc/7.4.0
module load python/3.6.6-anaconda3-5.3.0
module load cuda/10.1.243
module load hdf5/1.10.4

# activate env
#source activate /gpfs/alpine/proj-shared/med110/atrifan/scripts/pytorch-1.6.0_cudnn-8.0.2.39_nccl-2.7.8-1_static_mlperf
source activate /ccs/home/tkurth/project/pytorch/pytorch-1.6.0_cudnn-8.0.2.39_nccl-2.7.8-1_py-3.6_static_mlperf

# run tag
wandb_token=6c8b9db0b520487f05d32ebc76fcea156bd85d58
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | uniq | grep -v batch)
run_tag="cmaps-spike-summit-2-nnodes${nnodes}"

# we need that here too
nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

# launch job
jsrun -n ${nnodes} -r 1 -g 6 -a 3 -c 42 -d packed \
    ./run_vae_dist_summit.sh ${wandb_token} ${run_tag}
