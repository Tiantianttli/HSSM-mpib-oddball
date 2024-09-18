# Basics
import os
import sys
import time
from matplotlib import pyplot as plt
import arviz as az  # Visualization
import numpy as np
import pandas as pd
import pathlib
import seaborn
from pathlib import Path 
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrix
from ssms.basic_simulators.simulator import simulator
import bambi as bmb

import pytensor  # Graph-based tensor library
import hssm

# Set float type to float32 to avoid a current bug in PyMC mentioned above
# This will not be necessary in the future
hssm.set_floatX("float32")
# from jax.config import config

# config.update("jax_enable_x64", False)


basepath = '/mnt/beegfs/home/tli/oddballLearningMPIB'

# Check whether save directories exist; if not, create them
pathlib.Path(basepath+'/models/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/results/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/plots/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/ppc/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/models/hssm/aug24/numpyro/ssms/').mkdir(parents=True, exist_ok=True)


df = pd.read_csv(basepath+'/data/OB_singletrial_YAOA.csv')

df['response'] = df['accuracy']
# And then modify the 'response' column where the condition is met
df.loc[df['response'] == 0, 'response'] = -1

df.rename(columns={'reactiontime': 'rt'}, inplace=True)
df.rename(columns={'id': 'participant_id'}, inplace=True)

oddballDataDay2 = df[df['response'].notna()]

day2SubList=oddballDataDay2['participant_id'].unique()

vanillaHDDMallFree_simData=pd.DataFrame();
for sub in range(len(day2SubList)):
    
    idNum=day2SubList[sub]
    
    # figure out trial number from real data for the sub to determine sample numbers in the simulator
    trialNum=len(oddballDataDay2.loc[oddballDataDay2['participant_id'] == idNum])
    
    idCol=pd.DataFrame({'participant_id': [idNum] * trialNum})

    # Specify parameters based on gaussians using stats from the model fit with real data
    v_true, a_true, z_true, t_true = [np.random.normal(2.021,0.005), 
                                      np.random.normal(1.405,0.068), 
                                      np.random.normal(0.573,0.056),
                                      np.random.normal(0.372,0.029)]

    # Simulate data
    sim_out = simulator(
        theta=[v_true, a_true, z_true, t_true],  # parameter list
        model="ddm",  # specify model (many are included in ssms)
        n_samples=trialNum,  # number of samples for each set of parameters
    )

    # Turn into nice dataset
    # Turn data into a pandas dataframe
    subDataset = pd.DataFrame(
        np.column_stack([idCol["participant_id"],sim_out["rts"][:, 0], sim_out["choices"][:, 0]]),
        columns=["participant_id", "rt", "response"],
    )

    vanillaHDDMallFree_simData=pd.concat([vanillaHDDMallFree_simData, subDataset], ignore_index=True)


simple_ddm_model_hier_all_noPrior = hssm.HSSM(
    data=vanillaHDDMallFree_simData,
    model="ddm",
    hierarchical=True)

inferenceSimData_simple_ddm_model_hier_all_noPrior=simple_ddm_model_hier_all_noPrior.sample(
    sampler="nuts_numpyro",
    chains=2,
    cores=1,
    draws=10000,
    tune=2000,
    idata_kwargs=dict(log_likelihood=True),)

fileName = basepath + '/models/hssm/aug24/numpyro/ssms/inferenceSimData_simple_ddm_model_hier_all_noPrior.nc'
data = inferenceSimData_simple_ddm_model_hier_all_noPrior
az.to_netcdf(data, fileName)

print('done')