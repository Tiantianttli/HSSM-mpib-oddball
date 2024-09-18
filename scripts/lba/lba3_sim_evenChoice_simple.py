# run on hssm_test
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
from pandas.api.types import CategoricalDtype
import ssms
import pytensor  # Graph-based tensor library
import hssm
import random
# Set float type to float32 to avoid a current bug in PyMC 
# This will not be necessary in the future
hssm.set_floatX("float32")

basepath = '/users/afengler/data/proj_tt/mpib-HSSM-oddball-data'

# Check whether save directories exist; if not, create them
pathlib.Path(basepath+'/models/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/results/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/plots/hssm/').mkdir(parents=True, exist_ok=True)
pathlib.Path(basepath+'/ppc/hssm/').mkdir(parents=True, exist_ok=True)

df2 = pd.read_csv(basepath+'/data/OB_singletrial_YAOA.csv')
df2['switch_condition_num']

#dataForModel["trl_condition_bin"] (1==standard, 2==oddball)
#dataForModel["trl_condition"] (1==standard, 2==oddball1,3==oddball2)
choice1=pd.Series([np.nan]*len(df2["trl_condition"]))
for i in range(len(df2["trl_condition"])): 
    if df2["accuracy"].iloc[i] == 1:
        choice1[i]=int(df2["trl_condition"].iloc[i]-1)
    elif df2["accuracy"].iloc[i] == 0 and df2["trl_condition_bin"].iloc[i] == 2:   
        choice1[i]=0
    elif df2["accuracy"].iloc[i] == 0 and df2["trl_condition_bin"].iloc[i] == 1: 
        choice1[i]=random.randint(1, 2)
    
df2['response']=choice1    
df2.rename(columns={'reactiontime': 'rt'}, inplace=True)
df2.rename(columns={'id': 'participant_id'}, inplace=True)

oddballDataDay2Choice = df2[df2['response'].notna()]

oddballDataDay2Choice = oddballDataDay2Choice.copy()
oddballDataDay2Choice = oddballDataDay2Choice[oddballDataDay2Choice['switch_condition_num'].notna()]

oddballDataDay2Choice = oddballDataDay2Choice.reset_index(drop=True)

day2SubList=oddballDataDay2Choice['participant_id'].unique()


#sim data
vanillaLBAallFree_simDataEvenChoice=pd.DataFrame();
for sub in range(len(day2SubList)):
    
    idNum=day2SubList[sub]
    
    # figure out trial number from real data for the sub to determine sample numbers in the simulator
    trialNum=len(oddballDataDay2Choice.loc[oddballDataDay2Choice['participant_id'] == idNum])
    
    idCol=pd.DataFrame({'participant_id': [idNum] * trialNum})

    
    # Specify parameters based on gaussians using stats from the model fit with real data
    A_true, b_true, v0_true, v1_true, v2_true  = [np.random.normal(0.137,0.020),
                                                 np.random.normal(0.498,0.032),
                                                 np.random.normal(1/3,0.01), 
                                                 np.random.normal(1/3,0.01), 
                                                 np.random.normal(1/3,0.01)
                                                 ]

#     v0_true, v1_true, v2_true, a_true, z_true = [1/3, 
#                                                  1/3, 
#                                                  1/3,
#                                                  0.5,
#                                                  0.2]
    # Simulate data
    sim_out = simulator(
        theta=[A_true, b_true, v0_true, v1_true, v2_true],  # parameter list
        model="lba3",  # specify model (many are included in ssms)
        n_samples=trialNum,  # number of samples for each set of parameters
    )

    # Turn into nice dataset
    # Turn data into a pandas dataframe
    subDataset = pd.DataFrame(
        np.column_stack([idCol["participant_id"],sim_out["rts"][:, 0], sim_out["choices"][:, 0]]),
        columns=["participant_id", "rt", "response"],
    )

    vanillaLBAallFree_simDataEvenChoice=pd.concat([vanillaLBAallFree_simDataEvenChoice, subDataset], ignore_index=True)


simple_LBA_model_hier_all_noPrior = hssm.HSSM(
    data=vanillaLBAallFree_simDataEvenChoice,
    model="lba3",
    choices = [0,1,2],
    hierarchical=True,
    noncentered=False,
    loglik_kind="analytical")

pathlib.Path(basepath+'/models/lba/sept24/numpyro/').mkdir(parents=True, exist_ok=True)

inferenceData_simple_LBA_model_hier_all_noPrior=simple_LBA_model_hier_all_noPrior.sample(
    sampler="nuts_numpyro",
    chains=4,
    cores=1,
    draws=10,
    tune=10,
    discard_tuned_samples=False,
    idata_kwargs=dict(log_likelihood=False),)

fileName = basepath + '/models/lba/sept24/numpyro/inferenceData_sim_evenChoice_simple_LBA_model_hier_all_noPrior.nc'
data = inferenceData_simple_LBA_model_hier_all_noPrior
az.to_netcdf(data, fileName)


print('done')