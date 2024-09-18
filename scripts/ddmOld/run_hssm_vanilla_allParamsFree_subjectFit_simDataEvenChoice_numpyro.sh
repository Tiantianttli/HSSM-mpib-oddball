#!/bin/bash
#SBATCH --job-name simAllFreeVanillaEvenChoice 
#SBATCH --time 80:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem 32GB
#SBATCH --mail-user tli
#SBATCH --mail-type BEGIN,FAIL,END
#SBATCH --chdir .
#SBATCH --partition long


python ssms_vanilla_allParamsFree_subjectFit_evenChoice_numpyro.py
