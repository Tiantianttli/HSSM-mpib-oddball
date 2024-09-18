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



networkPath=basepath + r"/race_3_no_bias_lan_f2d991a6635b11efb6b1a0423f3e9b4e_torch_model.onnx"

simple_race_model_noHier_all_noPrior = hssm.HSSM(
    data=oddballDataDay2Choice,
    model="race_no_bias_3",
    hierarchical=False,
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

inferenceData_simple_race_model_noHier_all_noPrior=simple_race_model_noHier_all_noPrior.sample(
    sampler="nuts_numpyro",
    chains=2,
    cores=1,
    draws=3000,
    tune=3000,
    discard_tuned_samples=False,
    idata_kwargs=dict(log_likelihood=True),)

fileName = basepath + '/models/race/sept24/numpyro/inferenceData_simple_race_model_noHier.nc'
data = inferenceData_simple_race_model_noHier_all_noPrior
az.to_netcdf(data, fileName)

print('done')