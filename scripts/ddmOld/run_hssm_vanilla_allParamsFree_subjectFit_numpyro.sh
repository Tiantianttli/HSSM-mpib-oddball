#!/bin/bash
#SBATCH --job-name allFreeVanilla 
#SBATCH --time 80:00:00
#SBATCH --cpus-per-task 4
#SBATCH --mem 32GB
#SBATCH --mail-user tli
#SBATCH --mail-type BEGIN,FAIL,END
#SBATCH --chdir .
#SBATCH --partition long


python hssm_vanilla_allParamsFree_subjectFit_numpyro.py
