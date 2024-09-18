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

data=oddballDataDay2[:]
dataForModel = data[data['pupil'].notna()]

dataForModel = dataForModel.copy()
dataForModel = dataForModel[dataForModel['beta_insula'].notna()]

dataForModel = dataForModel.copy()
dataForModel = dataForModel[dataForModel['switch_condition_num'].notna()]


dataForModel['switch_condition_num']=isinstance(dataForModel['switch_condition_num'].dtype, CategoricalDtype)

day2SubList=dataForModel['participant_id'].unique()

HDDMaFixedPupilInsula_simData=pd.DataFrame();
for sub in range(len(day2SubList)):
    
    idNum=day2SubList[sub]
    
    # figure out trial number from real data for the sub to determine sample numbers in the simulator
    trialNum=len(dataForModel.loc[dataForModel['participant_id'] == idNum])
    
    idCol=pd.DataFrame({'participant_id': [idNum] * trialNum})

    # Specify parameters based on gaussians using stats from the model fit with real data
    v_true, a_true, z_true, t_true, z_switch_condition_num, z_pupil, z_beta_insula = [np.random.normal(4.155,0.097), 
                                                                                      3.47, 
                                                                                      np.random.normal(0.801,0.008),
                                                                                      np.random.normal(0.399,0.014),
                                                                                      np.random.normal(0.012,0.241),
                                                                                      np.random.normal(0.025,0.237),
                                                                                      np.random.normal(0.014,0.251),
                                                                                     ]
    
    # get pupil and insula data form real data as z covariates
    pupilData=pd.array(dataForModel['pupil'].loc[dataForModel['participant_id']== idNum])
    insulaData=pd.array(dataForModel['beta_insula'].loc[dataForModel['participant_id']== idNum])
    
    # get trial switch condition data form real data as z covariates
    switchCondition=pd.array(dataForModel['switch_condition_num'].loc[dataForModel['participant_id']== idNum])
    
    # z changes trial wise
    for trial in range(trialNum):
        z_true_trialwise = z_true+z_switch_condition_num*switchCondition[trial]+z_pupil*pupilData[trial]+z_beta_insula*insulaData[trial]
    
    theta_mat = np.zeros((trialNum, 4))
    theta_mat[:, 0] = v_true
    theta_mat[:, 1] = a_true
    theta_mat[:, 2] = z_true_trialwise
    theta_mat[:, 3] = t_true
    
    # Simulate data
    sim_out = simulator(
        theta=theta_mat,  # parameter list
        model="ddm",  # specify model (many are included in ssms)
        n_samples=1  
    )

    
    
    # Turn into nice dataset
    # Turn data into a pandas dataframe
    subDataset = pd.DataFrame(
        np.column_stack([pd.array(idCol["participant_id"]),sim_out["rts"][:, 0], sim_out["choices"][:, 0],pupilData,insulaData,switchCondition]),
        columns=["participant_id", "rt", "response","pupil","beta_insula","switch_condition_num"],
    )

    HDDMaFixedPupilInsula_simData=pd.concat([HDDMaFixedPupilInsula_simData, subDataset], ignore_index=True)



ddm_model_hier_aFix_dependsSwitch_trialNeuReg = hssm.HSSM(
    data=HDDMaFixedPupilInsula_simData,
    model="ddm",
    a=3.47,
    hierarchical=True,
    include=[
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant_id)+(switch_condition_num|participant_id)+(pupil|participant_id)+(beta_insula|participant_id)",
            "link": "identity",
        }
    ],    
)

inferenceSimData_simple_ddm_model_hier_aFix_dependsSwitch_trialNeuReg=ddm_model_hier_aFix_dependsSwitch_trialNeuReg.sample(
    sampler="nuts_numpyro",
    chains=2,
    cores=1,
    draws=10000,
    tune=2000,
    idata_kwargs=dict(log_likelihood=True),)

fileName = basepath + '/models/hssm/aug24/numpyro/ssms/inferenceSimData_simple_ddm_model_hier_aFix_dependsSwitch_trialNeuReg.nc'
data = inferenceSimData_simple_ddm_model_hier_aFix_dependsSwitch_trialNeuReg
az.to_netcdf(data, fileName)

print('done')