#!/bin/bash
#SBATCH --job-name simaFixDependsBiasNeuralDummy 
#SBATCH --time 80:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem 32GB
#SBATCH --mail-user tli
#SBATCH --mail-type BEGIN,FAIL,END
#SBATCH --chdir .
#SBATCH --partition long


python ssms_aFixed_subjectFit_dependsSwitch_neuralRegress_dummyVar_numpyro.py
