"""This script contain useful funcitons when working with cortical surface data in python
"""

import numpy as np
import pandas as pd
import os

PUBLIC_PATH='/public/lama'

if not os.path.isdir(PUBLIC_PATH):
    PUBLIC_PATH='/Volumes/public/lama' # Used when hyades is mounted to own computer
if not os.path.isdir(PUBLIC_PATH):
    PUBLIC_PATH=os.path.expanduser('~') # If not on hyades, data should lie in home folder on own computer

# ----- ROI functions ------
def get_roi_mask(aal_list):
    """Get mask of anatomical areas
    Explanation of anatomical atlas areas can be found in /public/lama/data/surface/aal_full.txt
    Note: Area labels for left and right are different, include both.  
    
    Parameters
    ----------
    aal_list : list<int>
        List of anatomical areas the mask should contain
    """

    roi = {'left': [],
           'right': []}

    for hemisphere in ['left', 'right']:
        labels = np.loadtxt(f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_{hemisphere}_smooth_ana_clean.labels', skiprows=1)

        roi[hemisphere] = np.isin(labels, [aal_list])+0 # +0 to make 1/0 instead of true/false

    return roi

def lookup_roi(aal_area):
    """Get name of anatomical area.
    Explanation of anatomical atlas areas can be found in /public/lama/data/surface/aal_full.txt
    
    Parameters
    ----------
    aal_area : int
        Value of anatomical area
    """

    aal_full = pd.read_csv(f'{PUBLIC_PATH}/data/surface/aal_full.txt', names=['val', 'name'])

    name = aal_full[aal_full.val == aal_area].name.squeeze().replace(' ', '_')

    return name

# ----- Brainstat functions ------
from brainspace.mesh.mesh_io import read_surface
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

surf = {'left': read_surface(f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii'),
        'right': read_surface(f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii')}

def unpaired_ttest(data_group1, data_group2, correction=None, cluster_threshold=0.001):

    result = {'left': [], 'right': []}

    group1_subjects = data_group1['left'].columns
    group2_subjects = data_group2['left'].columns

    group1 = pd.DataFrame(np.concatenate([np.ones(len(group1_subjects)), np.zeros(len(group2_subjects))]), columns=['group1'])
    group2 = pd.DataFrame(np.concatenate([np.zeros(len(group1_subjects)), np.ones(len(group2_subjects))]), columns=['group2'])

    for hemisphere in ['left', 'right']:
        data = pd.concat([data_group1[hemisphere], data_group2[hemisphere]], axis=1).T

        term_group1 = FixedEffect(group1)
        term_group2 = FixedEffect(group2)

        model = term_group1 + term_group2
        contrast = term_group2.group2 - term_group1.group1

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    print(f'Group 1: N={len(group1_subjects)}, group 2: N={len(group2_subjects)}')

    return result

def paired_ttest(data1, data2, correction=None, cluster_threshold=0.001):
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(data1['left'].columns) & set(data2['left'].columns)))

    measurement_1 = pd.DataFrame(np.concatenate([np.ones(len(common_subjects)), np.zeros(len(common_subjects))]), columns=['measurment_1'])
    measurement_2 = pd.DataFrame(np.concatenate([np.zeros(len(common_subjects)), np.ones(len(common_subjects))]), columns=['measurment_2'])
    subjects = pd.DataFrame(np.tile(np.eye(len(common_subjects)), 2).T, columns=common_subjects)

    for hemisphere in ['left', 'right']:
        data = pd.concat([data1[hemisphere][common_subjects], data2[hemisphere][common_subjects]], axis=1).T

        term_meas1 = FixedEffect(measurement_1, add_intercept=False)
        term_meas2 = FixedEffect(measurement_2, add_intercept=False)
        term_subject = FixedEffect(subjects, add_intercept=False)

        model = term_meas1 + term_meas2 + term_subject
        contrast = term_meas2.measurment_2 - term_meas1.measurment_1

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    print(f'N={len(common_subjects)}')
    
    return result

def correlation(surface_data, predictors, correction=None, cluster_threshold=0.001):
    """
    Predictiors is a pandas dataframe with subject_id as column headers (same id as in surface_data)
    If more than one row is in the dataframe, the remaining rows are used as covariates.
    """
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data['left'].columns) & set(surface_data['right'].columns) & set(predictors.columns)))

    for hemisphere in ['left', 'right']:
        data = surface_data[hemisphere][common_subjects].T

        terms = {}
        model = []
        for predictor in predictors[common_subjects].index: 
            terms[predictor] = FixedEffect(predictors[common_subjects].loc[predictor, :], names=predictor)

            model = model + terms[predictor]        

        contrast = predictors[common_subjects].loc[predictors.index[0], :].values

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    return result, common_subjects
