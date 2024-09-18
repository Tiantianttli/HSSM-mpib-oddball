#!/bin/bash
#SBATCH --job-name aFixDependsBiasNeural 
#SBATCH --time 80:00:00
#SBATCH --cpus-per-task 4
#SBATCH --mem 32GB
#SBATCH --mail-user tli
#SBATCH --mail-type BEGIN,FAIL,END
#SBATCH --chdir .
#SBATCH --partition long


python hssm_aFixed_subjectFit_dependsSwitch_neuralRegress_bias_numpyro.py
