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

dummyV0=[]
dummyV1=[]
dummyV2=[]
for trial in range(len(oddballDataDay2Choice)):
    if oddballDataDay2Choice.trl_condition.iloc[trial]==1:
                trialV0=1
                trialV1=0
                trialV2=0 
    elif oddballDataDay2Choice.trl_condition.iloc[trial]==2:
                trialV0=0
                trialV1=1
                trialV2=0 
    elif oddballDataDay2Choice.trl_condition.iloc[trial]==3:
                trialV0=0
                trialV1=0
                trialV2=1 
    dummyV0=np.append(dummyV0,trialV0)
    dummyV1=np.append(dummyV1,trialV1)
    dummyV2=np.append(dummyV2,trialV2)

oddballDataDay2Choice["dummyV0"]=dummyV0
oddballDataDay2Choice["dummyV1"]=dummyV1
oddballDataDay2Choice["dummyV2"]=dummyV2



simple_LBA_model_hier_dependTrialType = hssm.HSSM(
    data=oddballDataDay2Choice,
    model="lba3",
    choices = [0,1,2],
    hierarchical=True,
    noncentered=False,
    include=[
        {
            "name": "v0",
            "formula": "v0 ~ 1 + (dummyV0) + (1|participant_id)",
            "link": "identity",
        },
        {
            "name": "v1",
            "formula": "v1 ~ 1 + (dummyV1) + (1|participant_id)",
            "link": "identity",
        },
        {
            "name": "v2",
            "formula": "v2 ~ 1 + (dummyV2) + (1|participant_id)",
            "link": "identity",
        }],
    loglik_kind="analytical")

pathlib.Path(basepath+'/models/lba/sept24/numpyro/').mkdir(parents=True, exist_ok=True)

inferenceData_simple_LBA_model_hier_all_dependsTrialType=simple_LBA_model_hier_dependTrialType.sample(
    sampler="nuts_numpyro",
    chains=2,
    cores=1,
    draws=5000,
    tune=5000,
    discard_tuned_samples=False,
    idata_kwargs=dict(log_likelihood=True),)

fileName = basepath + '/models/lba/sept24/numpyro/inferenceData_simple_LBA_model_trialDepend_hier.nc'
data = inferenceData_simple_LBA_model_hier_all_dependsTrialType
az.to_netcdf(data, fileName)


print('done')