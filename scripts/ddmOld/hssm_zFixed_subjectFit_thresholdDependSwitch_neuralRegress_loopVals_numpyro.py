
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

import pytensor  # Graph-based tensor library
import hssm

# Set float type to float32 to avoid a current bug in PyMC mentioned above
# This will not be necessary in the future
hssm.set_floatX("float32")
# from jax.config import config

# config.update("jax_enable_x64", False)



def hssm_zVals_func(zVal):
    basepath = '/mnt/beegfs/home/tli/oddballLearningMPIB'

    # Check whether save directories exist; if not, create them
    pathlib.Path(basepath+'/models/hssm/aug24/numpyro/zVals/threshold/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(basepath+'/results/hssm/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(basepath+'/plots/hssm/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(basepath+'/ppc/hssm/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(basepath+'/models/hssm/ssms').mkdir(parents=True, exist_ok=True)


    
    df = pd.read_csv(basepath+'/data/OB_singletrial_YAOA.csv')
    df['response'] = df['accuracy']
    # And then modify the 'response' column where the condition is met
    df.loc[df['response'] == 0, 'response'] = -1

    df.rename(columns={'reactiontime': 'rt'}, inplace=True)
    df.rename(columns={'id': 'participant_id'}, inplace=True)

    oddballDataDay2 = df[df['response'].notna()]

    data=oddballDataDay2[:]
    dataForModel = data[data['pupil'].notna()]

    dataForModel = dataForModel.copy()
    dataForModel = dataForModel[dataForModel['beta_insula'].notna()]

    dataForModel = dataForModel.copy()
    dataForModel = dataForModel[dataForModel['switch_condition_num'].notna()]


    dataForModel['switch_condition_num']=isinstance(dataForModel['switch_condition_num'].dtype, CategoricalDtype)


    ddm_model_hier_aFix_dependsSwitch_trialNeuReg = hssm.HSSM(
        data=dataForModel,
        model="ddm",
        z=zVal,
        hierarchical=True,
        include=[
            {
                "name": "a",
                "formula": "a ~ 1 + (1|participant_id)+(switch_condition_num|participant_id)+(pupil|participant_id)+(beta_insula|participant_id)",
                "link": "identity",
            }
        ],    
    )

    inferenceData_simple_ddm_model_hier_aFix_dependsSwitch_trialNeuReg=ddm_model_hier_aFix_dependsSwitch_trialNeuReg.sample(
        sampler="nuts_numpyro",
        chains=2,
        cores=1,
        draws=10000,
        tune=2000,
        idata_kwargs=dict(log_likelihood=True),)

    fileName = basepath + '/models/hssm/aug24/numpyro/zVals/threshold/inferenceData_simple_ddm_model_hier_zFix_ThresholdDependSwitch_trialNeuReg_'+str(zVal)+'.nc'
    data = inferenceData_simple_ddm_model_hier_aFix_dependsSwitch_trialNeuReg
    az.to_netcdf(data, fileName)
    
    
    print('DONE')
    
    
    