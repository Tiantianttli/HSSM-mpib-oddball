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

def lba_real_singleSub(subNum):
    
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

    subID=day2SubList[subNum]
    subData=oddballDataDay2Choice[oddballDataDay2Choice.participant_id==subID]

    subData=oddballDataDay2Choice[oddballDataDay2Choice.participant_id==subID]
    dummyV0=[]
    dummyV1=[]
    dummyV2=[]
    for trial in range(len(subData)):
        if subData.trl_condition.iloc[trial]==1:
                trialV0=1
                trialV1=0
                trialV2=0 
        elif subData.trl_condition.iloc[trial]==2:
                trialV0=0
                trialV1=1
                trialV2=0 
        elif subData.trl_condition.iloc[trial]==3:
                trialV0=0
                trialV1=0
                trialV2=1 
        dummyV0=np.append(dummyV0,trialV0)
        dummyV1=np.append(dummyV1,trialV1)
        dummyV2=np.append(dummyV2,trialV2)

    subData["dummyV0"]=dummyV0
    subData["dummyV1"]=dummyV1
    subData["dummyV2"]=dummyV2


    simple_LBA_model = hssm.HSSM(
        data=subData,
        model="lba3",
        choices = [0,1,2],
        hierarchical=False,
        noncentered=False,
        loglik_kind="analytical",
        include=[
            {
                "name": "v0",
                "formula": "v0 ~ 1 + dummyV0",
                "link": "identity",
            },
            {
                "name": "v1",
                "formula": "v1 ~ 1 + dummyV1",
                "link": "identity",
            },
            {
                "name": "v2",
                "formula": "v2 ~ 1 + dummyV2",
                "link": "identity",
            }])

    
    inferenceData_simple_LBA_model=simple_LBA_model.sample(
        sampler="nuts_numpyro",
        chains=2,
        cores=2,
        draws=3000,
        tune=3000,
        discard_tuned_samples=False,
        idata_kwargs=dict(log_likelihood=False),)


    pathlib.Path(basepath+'/models/lba/sept24/numpyro/singleSub/'+str(subID)+'/').mkdir(parents=True, exist_ok=True)


    fileName = basepath + '/models/lba/sept24/numpyro/singleSub/'+str(subID)+ '/inferenceData_lba_simple_dependsTrialType'+str(subID)+'.nc'
    data = inferenceData_simple_LBA_model
    az.to_netcdf(data, fileName)

    print('done')