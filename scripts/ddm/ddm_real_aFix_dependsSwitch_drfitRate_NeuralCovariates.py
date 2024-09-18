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



dataForModel['trl_condition'] = dataForModel['trl_condition'].astype('category')

ddm_model_hier_aFix_dependsSwitch_trialNeuReg = hssm.HSSM(
    data=dataForModel,
    model="ddm",
    a=1.6,
    hierarchical=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + (1|participant_id)+(trl_condition|participant_id)+(pupil|participant_id)+(beta_insula|participant_id)",
            "link": "identity",
        }
    ],    
)


inferenceData_ddm_aFix_hier=ddm_model_hier_aFix_dependsSwitch_trialNeuReg.sample(
    sampler="nuts_numpyro",
    chains=2,
    draws=2000,
    tune=5000,
    discard_tuned_samples=False,
    idata_kwargs=dict(log_likelihood=True),
    )



pathlib.Path(basepath+'/models/ddm/sept24/numpyro/').mkdir(parents=True, exist_ok=True)


fileName = basepath + '/models/ddm/sept24/numpyro/inferenceData_ddm_aFix_driftRate_neuralCovariates_offSet.nc'
data = inferenceData_ddm_aFix_hier
az.to_netcdf(data, fileName)

print('done')