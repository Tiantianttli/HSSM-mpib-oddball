#!/bin/bash

# job name:
#SBATCH -J lbaReal_singleSub

# priority
#SBATCH --account=carney-frankmj-condo

# output file
#SBATCH --output ../../slurm/slurm_lbaReal_singleSub_%A_%a.out

# Request runtime, memory, cores
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1

#SBATCH --array=0-70

# --------------------------------------------------------------------------------------

# BASIC SETUP

# Read in arguments:
network_id=None
config_file=None
conda_env_name=hssm_gpyou
bashrc_path=/users/afengler/.bashrc
data_gen_base_path=/users/afengler/data/proj_tt/mpib-HSSM-oddball-data/models

while [ ! $# -eq 0 ]
    do
        case "$1" in
            --config_path | -cf)
                config_path=$2
                ;;
            --conda_env_name | -ce)
                conda_env_name=$2
                ;;
            --bashrc_path | -bp)
                bashrc_path=$2
                ;;
            --data_gen_base_path | -db)
                data_gen_base_path=$2
                ;;
        esac
        shift 2
    done

echo "The config file supplied is: $config_path"

# USER-INPUT NEEDED
source $bashrc_path  # NOTE: This file needs to contain conda initialization stuff

# TODO: This double conda deactivate can be simplified further --> key is understanding how to handle .bashrc / .bash_profile correctly
conda deactivate
conda deactivate
conda activate $conda_env_name


# # Ensure each task has a unique PYTENSOR_FLAGS
export PYTENSOR_FLAGS="compiledir_format=compiledir_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


python -u run_lba_real_singleSub_array.py --identity $SLURM_ARRAY_TASK_ID \
                                        #   --config_path $config_path \
                                        #   --data_gen_base_path $data_gen_base_path







