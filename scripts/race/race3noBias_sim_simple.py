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


vanillaRACE3allFree_simData=pd.DataFrame();
for sub in range(len(day2SubList)):
    
    idNum=day2SubList[sub]
    
    # figure out trial number from real data for the sub to determine sample numbers in the simulator
    trialNum=len(oddballDataDay2Choice.loc[oddballDataDay2Choice['participant_id'] == idNum])
    
    idCol=pd.DataFrame({'participant_id': [idNum] * trialNum})

    
    # Specify parameters based on gaussians using stats from the model fit with real data
    v0_true, v1_true, v2_true, a_true, z_true, t_true = [np.random.normal(2.446,0.013), 
                                                         np.random.normal(0.726,0.091),
                                                         np.random.normal(0.741,0.077),
                                                         np.random.normal(1.382,0.047),
                                                         np.random.normal(0.053,0.003),
                                                         np.random.normal(0.314,0.0026)]

#     v0_true, v1_true, v2_true, a_true, z_true = [1/3, 
#                                                  1/3, 
#                                                  1/3,
#                                                  0.5,
#                                                  0.2]
    # Simulate data
    sim_out = simulator(
        theta=[v0_true, v1_true, v2_true, a_true, z_true, t_true],  # parameter list
        model="race_no_bias_3",  # specify model (many are included in ssms)
        n_samples=trialNum,  # number of samples for each set of parameters
    )

    # Turn into nice dataset
    # Turn data into a pandas dataframe
    subDataset = pd.DataFrame(
        np.column_stack([idCol["participant_id"],sim_out["rts"][:, 0], sim_out["choices"][:, 0]]),
        columns=["participant_id", "rt", "response"],
    )

    vanillaRACE3allFree_simData=pd.concat([vanillaRACE3allFree_simData, subDataset], ignore_index=True)

networkPath=basepath + r"/race_3_no_bias_lan_f2d991a6635b11efb6b1a0423f3e9b4e_torch_model.onnx"

simple_race_model_hier_all_noPrior = hssm.HSSM(
    data=vanillaRACE3allFree_simData,
    model="race_no_bias_3",
    hierarchical=True,
    loglik_kind="approx_differentiable",
    model_config = {
        "list_params": ["v0", "v1", "v2", "a", "z", "t"],
        "bounds": {
            "v0":  (0.0, 2.5),
            "v1": (0.0, 2.5),
            "v2": (0.0, 2.5),
            "a": (1.0, 3.0),
            "z": (0.0, 0.9),
            "t": (0.001, 2),
        },
        "backend": "jax",
    },
    choices=3,
    loglik = networkPath,
    p_outlier = 0,)



pathlib.Path(basepath+'/models/race/sept24/numpyro/').mkdir(parents=True, exist_ok=True)

inferenceData_simple_race_model_hier_all_noPrior=simple_race_model_hier_all_noPrior.sample(
    sampler="nuts_numpyro",
    chains=4,
    cores=1,
    draws=3000,
    tune=5000,
    discard_tuned_samples=False,
    idata_kwargs=dict(log_likelihood=True),)

fileName = basepath + '/models/race/sept24/numpyro/inferenceData_sim_simple_race_model_hier_all_noPrior_new.nc'
data = inferenceData_simple_race_model_hier_all_noPrior
az.to_netcdf(data, fileName)

print('done')