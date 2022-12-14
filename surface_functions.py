"""This script contain useful funcitons when working with cortical surface data in python
"""

from unicodedata import name
import numpy as np
import pandas as pd
import os

from scipy.stats import pearsonr

PUBLIC_PATH='/public/lama'

if not os.path.isdir(PUBLIC_PATH):
    PUBLIC_PATH='/Volumes/public/lama' # Used when hyades is mounted to own computer
if not os.path.isdir(PUBLIC_PATH):
    PUBLIC_PATH=os.path.expanduser('~') # If not on hyades, data should lie in home folder on own computer

SURFACE_GII = {'left': f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii',
               'right': f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii'}

ATLAS_LABELS = {'left': f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth_ana_clean.labels',
               'right': f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth_ana_clean.labels'}

ATLAS_LOOKUP = f'{PUBLIC_PATH}/data/surface/aal_full.txt'

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
        labels = np.loadtxt(ATLAS_LABELS[hemisphere], skiprows=1)

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

    aal_full = pd.read_csv(ATLAS_LOOKUP, names=['val', 'name'])

    name = aal_full[aal_full.val == aal_area].name.squeeze().replace(' ', '_')

    return name

# ----- Brainstat functions ------
from brainspace.mesh.mesh_io import read_surface
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

surf = {'left': read_surface(f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii'),
        'right': read_surface(f'{PUBLIC_PATH}/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii')}

def unpaired_ttest(data_group1, data_group2, covars=None, correction=None, cluster_threshold=0.001, alpha=0.05):
    """
    alpha : float | 0.05
        Threshold of corrected clusters
    covars : pandas df
        Dataframe with covariates (one per row) and header is id (matching ids in data_group1 and data_group2)
    """
    result = {'left': [], 'right': []}

    group1_subjects = data_group1['left'].columns
    group2_subjects = data_group2['left'].columns

    group1 = pd.DataFrame(np.concatenate([np.ones(len(group1_subjects)), np.zeros(len(group2_subjects))]), columns=['group1'])
    group2 = pd.DataFrame(np.concatenate([np.zeros(len(group1_subjects)), np.ones(len(group2_subjects))]), columns=['group2'])

    for hemisphere in ['left', 'right']:
        data = pd.concat([data_group1[hemisphere], data_group2[hemisphere]], axis=1).T

        mask = ~data.isna().any(axis=0).values 

        term_group1 = FixedEffect(group1)
        term_group2 = FixedEffect(group2)

        model = term_group1 + term_group2

        if covars is not None:
            for covar in covars.index:
                covar_term = FixedEffect(covars.loc[covar][group1_subjects].append(covars.loc[covar][group2_subjects]).values, names=covar)

                model = model + covar_term

        contrast = term_group2.group2 - term_group1.group1

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    print(f'Group 1: N={len(group1_subjects)}, group 2: N={len(group2_subjects)}')

    cluster_mask = get_cluster_mask(result, correction, alpha)
    cluster_summary = get_cluster_summary(result)

    return result, cluster_mask, cluster_summary

def paired_ttest(data1, data2, correction=None, cluster_threshold=0.001, alpha=0.05):
    """
    alpha : float | 0.05
        Threshold of corrected clusters
    """
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(data1['left'].columns) & set(data2['left'].columns)))

    measurement_1 = pd.DataFrame(np.concatenate([np.ones(len(common_subjects)), np.zeros(len(common_subjects))]), columns=['measurment_1'])
    measurement_2 = pd.DataFrame(np.concatenate([np.zeros(len(common_subjects)), np.ones(len(common_subjects))]), columns=['measurment_2'])
    subjects = pd.DataFrame(np.tile(np.eye(len(common_subjects)), 2).T, columns=common_subjects)

    for hemisphere in ['left', 'right']:
        data = pd.concat([data1[hemisphere][common_subjects], data2[hemisphere][common_subjects]], axis=1).T

        mask = ~data.isna().any(axis=0).values 

        term_meas1 = FixedEffect(measurement_1, add_intercept=False)
        term_meas2 = FixedEffect(measurement_2, add_intercept=False)
        term_subject = FixedEffect(subjects, add_intercept=False)

        model = term_meas1 + term_meas2 + term_subject
        contrast = term_meas2.measurment_2 - term_meas1.measurment_1

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    print(f'N={len(common_subjects)}')

    cluster_mask = get_cluster_mask(result, correction, alpha)
    cluster_summary = get_cluster_summary(result)
    
    return result, common_subjects, cluster_mask, cluster_summary

def correlation(surface_data, predictors, correction=None, cluster_threshold=0.001, alpha=0.05):
    """
    Correlation of surface with value (e.g. demography data such as age or cognitive score)

    Parameters
    ----------
    surface_data : dict('left', 'right')
        Independent surface data
    Predictors : DataFrame
        Predictiors is a pandas dataframe with subject_id as column headers (same id as in surface_data)
        If more than one row, the rest of the rows are considere covariates
    correction : str | None
        Mulitple comparison correction: 'rft' or 'fdr'
    cluster_threshold : float | 0.001
        Primary cluster threshold
    alpha : float | 0.05
        Threshold of corrected clusters
    """
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data['left'].columns) & set(surface_data['right'].columns) & set(predictors.columns)))

    for hemisphere in ['left', 'right']:
        data = surface_data[hemisphere][common_subjects].T

        mask = ~data.isna().any(axis=0).values 

        terms = {}
        model = []
        for predictor in predictors[common_subjects].index: 
            terms[predictor] = FixedEffect(predictors[common_subjects].loc[predictor, :], names=predictor)

            model = model + terms[predictor]

        contrast = predictors[common_subjects].loc[predictors.index[0], :].values

        # --- Run model ---
        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    cluster_mask = get_cluster_mask(result, correction, alpha)
    cluster_summary = get_cluster_summary(result)
    
    return result, common_subjects, cluster_mask, cluster_summary

def correlation_pearson(param, predictor):
    """
    Pearson correlation of surface with value (e.g. demography data such as age or cognitive score) or surface

    Parameters
    ----------
    data : dict('left', 'right')
        Independent surface data
    Predictors : DataFrame
        Predictiors is a pandas dataframe with subject_id as column headers (same id as in surface_data)
        If more than one row, the rest of the rows are considere covariates
    """
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(param['left'].columns) & set(param['right'].columns) & 
                                  set(predictor['left'].columns) & set(predictor['right'].columns)))

    for hemisphere in ['left', 'right']:
        data = param[hemisphere][common_subjects].T

        mask = ~data.isna().any(axis=0).values

        # Initialise t values to nan
        r = np.zeros(mask.shape)
        r[:] = np.nan
        
        vert_list = np.where(mask==True)[0]

        # Run model for each vertex
        for i in vert_list:
            # --- Correlation with other surface ---
            r_corr, _ = pearsonr(predictor[hemisphere][common_subjects].iloc[i,:].values.T, data[[i]])
            r[i] = r_corr[0]

        result[hemisphere] = r
    
    return result, common_subjects

def correlation_other_surface(surface_data, surface_data_predictor, predictor_name='surface_data', covariates=None, correction=None, cluster_threshold=0.001, alpha=0.05):
    """
    Correlation between two surfaces

    Parameters
    ----------
    surface_data : dict('left', 'right')
        Independent surface data
    surface_data_predictor : dict('left', 'right')
        Predictor surface data
    predictor_name : str | 'surface_data'
        Name of predictor variable
    covariates : DataFrame | None
         Covariates (not surface data) can be given as a pandas dataframe with subject_id as column headers (same id as in surface_data)
    correction : str | None
        Mulitple comparison correction: 'rft' or 'fdr'
    cluster_threshold : float | 0.001
        Primary cluster threshold
    alpha : float | 0.05
        Threshold of corrected clusters
    """
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data['left'].columns) & set(surface_data['right'].columns) & 
                                  set(surface_data_predictor['left'].columns) & set(surface_data_predictor['right'].columns)))

    for hemisphere in ['left', 'right']:
        data = surface_data[hemisphere][common_subjects].T

        mask = ~data.isna().any(axis=0).values

        # Initialise t values to nan
        t = np.zeros(mask.shape)
        t[:] = np.nan
        
        vert_list = np.where(mask==True)[0]

        # Run model for each vertex
        for i in vert_list:
            # --- Correlation with other surface ---
            term = FixedEffect(surface_data_predictor[hemisphere][common_subjects].iloc[i,:].values.T)
            model = term
            contrast = model.x0

            if covariates is not None:
                for covar in covariates[common_subjects].index: 
                    term = FixedEffect(covariates[common_subjects].loc[covar, :], names=covar)

                    model = model + term

            # --- Run model ---
            slm = SLM(model, contrast)
            slm.fit(data[[i]])

            t[i] = slm.t[0][0]
        
        # Run with mean data to compute multple comparison
        term_slope = FixedEffect(surface_data_predictor[hemisphere][common_subjects].mean().values, names=predictor_name)
        model = term_slope
        if covariates is not None:
            for covar in covariates[common_subjects].index: 
                term = FixedEffect(covariates[common_subjects].loc[covar, :], names=covar)
                model = model + term

        contrast = model.matrix[predictor_name].values

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        slm.t = np.array([t])
        if correction is not None:
            slm.multiple_comparison_corrections(True)

        result[hemisphere] = slm

    cluster_mask = get_cluster_mask(result, correction, alpha)
    cluster_summary = get_cluster_summary(result)

    return result, common_subjects, cluster_mask, cluster_summary
    
def get_cluster_mask(result, correction, alpha):
    if correction is not None:
        # Get mask of surviving clusters (alpha*2, to get one-sided result)
        cluster_mask = {'left': result['left'].P['pval']['C'] < alpha*2 if result['left'].P['pval']['C'] is not None else np.zeros_like(result['left'].mask),
                        'right': result['right'].P['pval']['C'] < alpha*2 if result['right'].P['pval']['C'] is not None else np.zeros_like(result['right'].mask)}
    else:
        cluster_mask = {'left': np.ones_like(result['left'].mask),
                        'right': np.ones_like(result['right'].mask)}

    return cluster_mask

def get_cluster_summary(result):
    """Calculate summary of surviving clusters
    
    Return information on cluster area (mm^2), cluster_id, cluster location (MNI coordinates) and cluster corrected p-value
    """

    cluster_summary = pd.DataFrame({'hemisphere': [], 'clusid': [], 'cluster_area_mm2': [], 'mni_coord': [], 'anatomical_location': [], 'clus_pval_fwer': []})

    aal_full = pd.read_csv(ATLAS_LOOKUP, names=['val', 'name'])

    for hemisphere in ['left', 'right']:
        mni_coord = result[hemisphere].surf.Points
        labels = np.loadtxt(ATLAS_LABELS[hemisphere], skiprows=1)
        for posneg in [0, 1]:
            cluster_survived = result[hemisphere].P['clus'][posneg][result[hemisphere].P['clus'][posneg].P < 0.05]

            if cluster_survived.empty:
                continue

            for clusid in cluster_survived.clusid:
                clus_pval = result[hemisphere].P['clus'][posneg][result[hemisphere].P['clus'][posneg].clusid == clusid].P.values[0]
                peak_vertex = list(result[hemisphere].P['peak'][posneg][result[hemisphere].P['peak'][posneg].clusid == clusid].vertid)[0]

                anatomical_label = labels[peak_vertex]
                anatomical_loc = aal_full[aal_full.val == anatomical_label].name.squeeze()

                peak_coord = mni_coord[peak_vertex]
                peak_coord = [round(c) for c in peak_coord] # Round coordinates
                
                idx = np.where(result[hemisphere].P['clusid'][posneg][0] == clusid)[0]
                polys = result[hemisphere].surf.polys2D[np.isin(result[hemisphere].surf.polys2D, idx).all(axis=1)]

                area = 0
                for p in polys:
                    a = mni_coord[p[0]]
                    b = mni_coord[p[1]]
                    c = mni_coord[p[2]]

                    x = np.cross((a - b), (b - c))
                    A = np.sqrt(x.dot(x)) / 2

                    area += A

                cluster_summary = cluster_summary.append({'hemisphere': hemisphere, 'clusid': clusid, 'cluster_area_mm2': f'{area:.0f}', 'mni_coord': peak_coord, 'anatomical_location': anatomical_loc, 'clus_pval_fwer': clus_pval}, ignore_index=True)

    return cluster_summary


# def find_clusters(stat_data, threshold, threshold_is_min=False):
#     """Segments cortical surface p-map at given threshold and finds surviving clusters
#     Will do this for both hemispheres

#     stat_data : dict{'left', 'right'}
#         Statistical data for left and right hemisphere (e.g. pval, tval or zscore)
#     threshold : float
#         Value to threshold the statistical data
#     threshold_is_min : bool | False
#         If true, threshold is used as lower limit. Default is upper limit. 
#     """

#     indexed_clusters = {'left': np.zeros(len(stat_data['left']), ),
#                                 'right': np.zeros(len(stat_data['right']), )}
#     size = {'left': {}, 'right': {}}

#     for hemisphere in ['left', 'right']: 
#         surf = read_surface(SURFACE_GII[hemisphere])
#         faces_all = surf.polys2D
#         vert_idx = np.arange(faces_all.max() + 1)

#         if threshold_is_min:
#             not_used_indexes = set(vert_idx[stat_data[hemisphere].ravel() >= threshold])
#         else:
#             not_used_indexes = set(vert_idx[stat_data[hemisphere].ravel() <= threshold])

#         # == Only used faces containing not_used indexes ==
#         faces = faces_all[np.isin(faces_all, list(not_used_indexes)).any(axis=1)]

#         # -------------------------
#         # ----- Find clusters -----
#         # -------------------------
#         cluster_val = 1
#         while not_used_indexes:
#             in_cluster = {not_used_indexes.pop()}
#             neighbours = set(faces[np.isin(faces, list(in_cluster)).any(axis=1)].ravel()) & not_used_indexes
#             in_cluster = in_cluster | neighbours

#             while True:
#                 neighbours = set(faces[np.isin(faces, list(neighbours)).any(axis=1)].ravel()) & not_used_indexes
#                 not_used_indexes = not_used_indexes - neighbours
#                 if len(neighbours) == 0:
#                     break
#                 else:
#                     in_cluster = in_cluster | neighbours

#             indexed_clusters[hemisphere][list(in_cluster)] = cluster_val
#             size[hemisphere][cluster_val] = len(in_cluster)

#             cluster_val = cluster_val + 1

#     return indexed_clusters, size
