"""This script contain useful functions when working with cortical surface data in python.

Many functions are dependent on the Brainstat python module: https://brainstat.readthedocs.io/en/master/

Author: Lasse Stensvig Madsen
Mail: lasse.madsen@cfin.au.dk

Last edited: 12/9 - 2023
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

from scipy.stats import pearsonr
from surface_plot import plot_mean_stats, plot_stats, cluster_plot

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
    """
    Get mask of anatomical areas
    Explanation of anatomical atlas areas can be found in /public/lama/data/surface/aal_full.txt
    Note: Area labels for left and right are different: include both.

    Parameters
    ----------
    aal_list : list of int
        List of anatomical areas the mask should contain
    Returns
    -------
    roi : dict
        A dictionary containing the mask of anatomical areas for both the left and right hemispheres
    """

    roi = {'left': [],
           'right': []}

    for hemisphere in ['left', 'right']:
        labels = np.loadtxt(ATLAS_LABELS[hemisphere], skiprows=1)

        roi[hemisphere] = np.isin(labels, [aal_list])+0 # +0 to make 1/0 instead of true/false

    return roi

def lookup_roi(aal_area):
    """
    Get the name of the anatomical area.
    Explanation of anatomical atlas areas can be found in /public/lama/data/surface/aal_full.txt

    Parameters
    ----------
    aal_area : int
        The value of the anatomical area
    Returns
    -------
    name : str
        The name of the anatomical area
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

def unpaired_ttest(data_group1, data_group2, covars=None, correction='rft', cluster_threshold=0.001, alpha=0.05, 
                   plot=True, outdir=None, group_names=('Group 1', 'Group2'), param_name=None, clobber=False):
    """
    Perform an unpaired t-test on the two groups of data.
    
    Parameters
    ----------
    data_group1 : dict of DataFrame
        A dictionary containing the data for the first group of subjects, with 'left' and 'right' as keys.
    data_group2 : dict of DataFrame
        A dictionary containing the data for the second group of subjects, with 'left' and 'right' as keys.
    covars : pandas DataFrame, optional
        Dataframe with covariates (one per row) and header as id (matching ids in data_group1 and data_group2).
    correction : str or None, optional
        Correction method for multiple comparisons. If None, no correction is performed.
    cluster_threshold : float, optional
        Threshold for cluster formation, in percent (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    cluster_mask : np.ndarray
        An array containing the binary cluster mask for significant clusters, shape is (num_vertices,).
    cluster_summary : pandas DataFrame
        A DataFrame containing information about each significant cluster, with columns 'hemisphere', 'x', 'y', 'z', 
        'size', and 'p'.
    """
    if not correction in {'rft'} and correction is not None:
        print('Wrong correction method! Should be "rft" or None. Please try again.')
        return

    result = {'left': [], 'right': []}

    group1_subjects = data_group1['left'].columns
    group2_subjects = data_group2['left'].columns

    # Define covariates, if any
    if covars is not None:
        covar_term = None

        # Make sure all subjects have covars
        group1_subjects = sorted(list(set(group1_subjects) & set(covars.columns)))
        group2_subjects = sorted(list(set(group2_subjects) & set(covars.columns)))

        # Only use data from subjects with covars
        for hemisphere in ['left', 'right']:
            data_group1[hemisphere] = data_group1[hemisphere][group1_subjects]
            data_group2[hemisphere] = data_group2[hemisphere][group2_subjects]

        for covar in covars.index: 
            covar_term = covar_term + FixedEffect(pd.concat([covars.loc[covar][group1_subjects],covars.loc[covar][group2_subjects]], names=covar))

    print(f'Group 1: N={len(group1_subjects)}, group 2: N={len(group2_subjects)}')

    groups = pd.DataFrame({'group': ['0']*len(group1_subjects) + ['1']*len(group2_subjects)})

    # Calculate unpaired t-test
    for hemisphere in ['left', 'right']:
        data = pd.concat([data_group1[hemisphere], data_group2[hemisphere]], axis=1).T

        mask = ~data.isna().any(axis=0).values 

        term_groups = FixedEffect(groups)
        model = term_groups

        if covars is not None:
            model = model + covar_term

        contrast = term_groups.group_1 - term_groups.group_0

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        result[hemisphere] = slm
    

    cluster_mask = get_cluster_mask(result, correction, alpha)
    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result)

    if plot:
        if param_name is None:
            print('Please set parameter name!')
        elif outdir is None:
            print('Please specify outdir.')
        else:
            if covar is None:
                outdir = f'{outdir}/Unpaired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_p{cluster_threshold}'
                outfile_fwe_corrected = f'{outdir}/Unpaired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_fweCorrected_p{cluster_threshold}.jpg'
                outfile_uncorrected = f'{outdir}/Unpaired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_uncorrected_p{cluster_threshold}.jpg'
                cluster_summary_file = f'{outdir}/ClusterSum_Unpaired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_fweCorrected_{cluster_threshold}.csv'

            else:
                outdir = f'{outdir}/Unpaired_ttest_{param_name}+{"+".join(covars.index)}_{group_names[0]}_{group_names[1]}_p{cluster_threshold}'
                outfile_fwe_corrected = f'{outdir}/Unpaired_ttest_{param_name}+{"+".join(covars.index)}_{group_names[0]}_{group_names[1]}_fweCorrected_p{cluster_threshold}.jpg'
                outfile_uncorrected = f'{outdir}/Unpaired_ttest_{param_name}+{"+".join(covars.index)}_{group_names[0]}_{group_names[1]}_uncorrected_p{cluster_threshold}.jpg'
                cluster_summary_file = f'{outdir}/ClusterSum_Unpaired_ttest_{param_name}+{"+".join(covars.index)}_{group_names[0]}_{group_names[1]}_fweCorrected_{cluster_threshold}.csv'

            if not os.path.isdir(outdir) or not os.listdir(outdir) or clobber:
                Path(outdir).mkdir(exist_ok=True)
                print(f'Plotting results to {outdir}...')
                # ---- Calculate mean for each group ---- 
                mean_data = {'Group1': {'left': data_group1['left'].mean(axis=1), 'right': data_group1['right'].mean(axis=1)},
                             'Group2': {'left': data_group2['left'].mean(axis=1), 'right': data_group2['right'].mean(axis=1)}}

                # ---- Plot results ----
                t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
                mean_titles = [f'{group_names[0]} (n={len(group1_subjects)})', f'{group_names[1]} (n={len(group2_subjects)})']
                if correction is not None:
                    plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_fwe_corrected, p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, mean_titles=mean_titles, stats_titles='Difference', cluster_mask=cluster_mask, t_lim=[-5, 5])
                    cluster_plot.boxplot(data_group1, data_group2, result, outdir, group_names[0], group_names[1], param_name, alpha=cluster_threshold)
                    cluster_summary.to_csv(cluster_summary_file)

                plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_uncorrected, p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, mean_titles=mean_titles, stats_titles='Difference', t_lim=[-5, 5])
            else:
                print(f'{outdir} exists! Use clobber to overwrite.')

    return result, cluster_mask, cluster_summary

def paired_ttest(data1, data2, correction=None, cluster_threshold=0.001, alpha=0.05, 
                 plot=True, outdir=None, group_names=('Group 1', 'Group2'), param_name=None, clobber=False):
    """
    Perform a paired t-test on the data.
    
    Parameters
    ----------
    data_group1 : dict of DataFrame
        A dictionary containing the data for the first set of measurements with 'left' and 'right' as keys.
    data_group2 : dict of DataFrame
        A dictionary containing the data for the second set of measurements with 'left' and 'right' as keys.
    correction : str or None, optional
        Correction method for multiple comparisons. If None, no correction is performed.
    cluster_threshold : float, optional
        Threshold for cluster formation, in percent (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    cluster_mask : np.ndarray
        An array containing the binary cluster mask for significant clusters, shape is (num_vertices,).
    cluster_summary : pandas DataFrame
        A DataFrame containing information about each significant cluster, with columns 'hemisphere', 'x', 'y', 'z', 
        'size', and 'p'.
    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return

    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(data1['left'].columns) & set(data2['left'].columns)))
    print(f'N={len(common_subjects)}')

    for hemisphere in ['left', 'right']:
        data1[hemisphere] = data1[hemisphere][common_subjects]
        data2[hemisphere] = data2[hemisphere][common_subjects]

    measurements = pd.DataFrame({'measurements': ['0']*len(common_subjects) + ['1']*len(common_subjects)})

    for hemisphere in ['left', 'right']:
        data = pd.concat([data1[hemisphere], data2[hemisphere]], axis=1).T

        mask = ~data.isna().any(axis=0).values 

        term_meas = FixedEffect(measurements, add_intercept=False)
        term_subject = FixedEffect(common_subjects*2, add_intercept=False)

        model = term_meas + term_subject
        contrast = term_meas.measurements_1 - term_meas.measurements_0

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        result[hemisphere] = slm
    
    cluster_mask = get_cluster_mask(result, correction, alpha)

    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result)

    if plot:
        if param_name is None:
            print('Please set parameter name!')
        elif outdir is None:
            print('Please specify outdir.')
        else:
            outdir = f'{outdir}/Paired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_p{cluster_threshold}'
            outfile_fwe_corrected = f'{outdir}/Paired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_fweCorrected_p{cluster_threshold}.jpg'
            outfile_uncorrected = f'{outdir}/Paired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_uncorrected_p{cluster_threshold}.jpg'
            cluster_summary_file = f'{outdir}/ClusterSum_Unpaired_ttest_{param_name}_{group_names[0]}_{group_names[1]}_fweCorrected_{cluster_threshold}.csv'

            if not os.path.isdir(outdir) or not os.listdir(outdir) or clobber:
                Path(outdir).mkdir(exist_ok=True)
                print(f'Plotting results to {outdir}...')
                # ---- Calculate mean for each group ---- 
                mean_data = {'Group1': {'left': data1['left'].mean(axis=1), 'right': data1['right'].mean(axis=1)},
                             'Group2': {'left': data2['left'].mean(axis=1), 'right': data2['right'].mean(axis=1)}}

                # ---- Plot results ----
                t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
                mean_titles = [f'{group_names[0]} (n={len(common_subjects)})', f'{group_names[1]} (n={len(common_subjects)})']
                if correction is not None:
                    plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_fwe_corrected, p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, mean_titles=mean_titles, stats_titles='Difference', cluster_mask=cluster_mask, t_lim=[-5, 5])
                    cluster_plot.boxplot(data1, data2, result, outdir, group_names[0], group_names[1], param_name, alpha=cluster_threshold)
                    cluster_summary.to_csv(cluster_summary_file)
                plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_uncorrected, p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, mean_titles=mean_titles, stats_titles='Difference', t_lim=[-5, 5])
            else:
                print(f'{outdir} exists! Use clobber to overwrite.')
    
    return result, common_subjects, cluster_mask, cluster_summary

def correlation(surface_data, predictors, correction=None, cluster_threshold=0.001, alpha=0.05):
    """
    Calculate the correlation of surface with value (e.g. demography data such as age or cognitive score)

    Parameters
    ----------
    surface_data : dict of DataFrame
        A dictionary containing the data for the surface measurements with 'left' and 'right' as keys.
    predictors : DataFrame
        Pandas dataframe with subject_id as column headers (same id as in surface_data)
        If more than one row, the rest of the rows are considered covariates.
    correction : str, optional
        Multiple comparison correction: 'rft' or 'fdr'.
    cluster_threshold : float, optional
        Threshold for cluster formation, in percent (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
        
    Returns
    -------
    result : dict
        A dictionary containing the SLM corrrelation results for each hemisphere, with 'left' and 'right' as keys.
    common_subjects : list
        List of subjects used for the correlation analysis.
    cluster_mask : np.ndarray
        An array containing the binary cluster mask for significant clusters, shape is (num_vertices,).
    cluster_summary : pandas DataFrame
        A DataFrame containing information about each significant cluster, with columns 'hemisphere', 'x', 'y', 'z', 
        'size', and 'p'.
    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return

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
    
    if correction is None:
        cluster_summary = None
    else:
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

def correlation_other_surface(surface_data, surface_data_predictor, predictor_name='surface_data', covariates=None, correction=None, cluster_threshold=0.001, alpha=0.05,
                              plot=True, outdir=None, indep_name=None, clobber=False):
    """
    Calculate the correlation of two surfaces 

    Parameters
    ----------
    surface_data : dict of DataFrame
        A dictionary containing the data for the first surface measurements with 'left' and 'right' as keys.
    surface_data_predictor : dict of DataFrame
        A dictionary containing the data for the second surface measurements with 'left' and 'right' as keys.
    predictor_name : str, optional
        Name of the surface data for the second surface measurement (default is 'surface_data')
    covariates : pandas DataFrame, optional
        Dataframe with covariates (one per row) and header as id (matching ids in surface_data and surface_data_predictor).
    correction : str, optional
        Multiple comparison correction: 'rft' or 'fdr'.
    cluster_threshold : float, optional
        Threshold for cluster formation, in percent (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
        
    Returns
    -------
    result : dict
        A dictionary containing the SLM corrrelation results for each hemisphere, with 'left' and 'right' as keys.
    common_subjects : list
        List of subjects used for the correlation analysis.
    cluster_mask : np.ndarray
        An array containing the binary cluster mask for significant clusters, shape is (num_vertices,).
    cluster_summary : pandas DataFrame
        A DataFrame containing information about each significant cluster, with columns 'hemisphere', 'x', 'y', 'z', 
        'size', and 'p'.
    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return
    
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data['left'].columns) & set(surface_data['right'].columns) & 
                                  set(surface_data_predictor['left'].columns) & set(surface_data_predictor['right'].columns)))
                                  

    # Define covariates, if any
    if covariates is not None:
        # Make sure all subjects have covars
        common_subjects = sorted(list(set(common_subjects) & set(covariates.columns)))

        covar_term = None
        for covar in covariates[common_subjects].index: 
            covar_term = covar_term + FixedEffect(covariates[common_subjects].loc[covar, :], names=covar)

    # Only use data from subjects with covars and both data
    for hemisphere in ['left', 'right']:
        surface_data[hemisphere] = surface_data[hemisphere][common_subjects]
        surface_data_predictor[hemisphere] = surface_data_predictor[hemisphere][common_subjects]

    for hemisphere in ['left', 'right']:
        data = surface_data[hemisphere].T
        data_predictor = surface_data_predictor[hemisphere].T

        mask = (~data.isna().any(axis=0) & ~data_predictor.isna().any(axis=0)).values

        # Initialise t values to nan
        t = np.zeros(mask.shape)
        t[:] = np.nan
        
        vert_list = np.where(mask==True)[0]

        # Run model for each vertex
        for i in vert_list:
            # --- Correlation with other surface ---
            term = FixedEffect(data_predictor[i].values)
            model = term
            contrast = model.x0

            if covariates is not None:
                model = model + covar_term

            # --- Run model ---
            slm = SLM(model, contrast)
            slm.fit(data[[i]])

            t[i] = slm.t[0][0]
        
        # Run with mean data to compute multple comparison
        term_slope = FixedEffect(surface_data_predictor[hemisphere].mean().values, names=predictor_name)
        model = term_slope
        if covariates is not None:
            model = model + covar_term

        contrast = model.matrix[predictor_name].values

        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm.fit(data.values)

        slm.t = np.array([t])
        if correction is not None:
            slm.multiple_comparison_corrections(True)

        result[hemisphere] = slm

    cluster_mask = get_cluster_mask(result, correction, alpha)
    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result)
    
    if plot:
        if indep_name is None or predictor_name == 'surface_data':
            print('Please set parameter names!')
        elif outdir is None:
            print('Please specify outdir.')
        else:
            outdir = f'{outdir}/Correlation_{predictor_name}_{indep_name}_p{cluster_threshold}'
            outfile_fwe_corrected = f'{outdir}/Correlation_{predictor_name}_{indep_name}_p{cluster_threshold}_fweCorrected_p{cluster_threshold}.jpg'
            outfile_uncorrected = f'{outdir}/Correlation_{predictor_name}_{indep_name}_p{cluster_threshold}_uncorrected_p{cluster_threshold}.jpg'
            cluster_summary_file = f'{outdir}/ClusterSum_Correlation_{predictor_name}_{indep_name}_p{cluster_threshold}_fweCorrected_{cluster_threshold}.csv'

            if not os.path.isdir(outdir) or not os.listdir(outdir) or clobber:
                Path(outdir).mkdir(exist_ok=True)
                print(f'Plotting results to {outdir}...')

                # ---- Plot results ----
                t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
            
                title = ''
                if correction is not None:
                    plot_stats.plot_tval(t_value, outfile_fwe_corrected, p_threshold=cluster_threshold, df=result['left'].df, cluster_mask=cluster_mask, t_lim=[-5, 5], title=title, cbar_loc='left')
                    cluster_plot.correlation_plot(result, surface_data, indep_name, common_subjects, outdir)
                    cluster_summary.to_csv(cluster_summary_file)
                plot_stats.plot_tval(t_value, outfile_uncorrected, p_threshold=cluster_threshold, df=result['left'].df, t_lim=[-5, 5], title=title, cbar_loc='left')
            else:
                print(f'{outdir} exists! Use clobber to overwrite.')

    return result, common_subjects, cluster_mask, cluster_summary
    
def get_cluster_mask(result, correction, alpha):
    """
    Returns a mask indicating the clusters that survive the statistical test.

    Parameters
    ----------
    result : dict of DataFrame
        Results of the statistical test for each hemisphere with 'left' and 'right' as keys.
    correction : str or None
        Type of multiple comparison correction used. Valid values are 'rft', 'fdr' or None.
    alpha : float
        Threshold of corrected clusters.

    Returns
    -------
    dict('left', 'right')
        A dictionary with a mask indicating the clusters that survive the statistical test for each hemisphere.
    """
    if correction is not None:
        # Get mask of surviving clusters (alpha*2, to get one-sided result)
        cluster_mask = {'left': result['left'].P['pval']['C'] < alpha*2 if result['left'].P['pval']['C'] is not None else np.zeros_like(result['left'].mask),
                        'right': result['right'].P['pval']['C'] < alpha*2 if result['right'].P['pval']['C'] is not None else np.zeros_like(result['right'].mask)}
    else:
        cluster_mask = {'left': np.ones_like(result['left'].mask),
                        'right': np.ones_like(result['right'].mask)}

    return cluster_mask

def get_cluster_summary(result):
    """
    Calculate summary of surviving clusters.

    Parameters:
    -----------
    result : dict
        A dictionary containing the results of a statistical analysis for each hemisphere with 'left' and 'right' as keys.

    Returns:
    --------
    cluster_summary : pandas.DataFrame
        A DataFrame with information on cluster area (mm^2), cluster_id, cluster location (MNI coordinates),
        anatomical location, and cluster corrected p-value.
    """
    cluster_summary = pd.DataFrame({'clusid': [], 'Anatomical location (peak)': [], 'Hemisphere': [], 'Cluster area (mm2)': [], 'MNI coordinates (x,y,z)': [], 'Cluster FWE p-value': []})

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

                # Find peak vertex
                clus_indices = np.where(result[hemisphere].P['clusid'][posneg] == clusid)[1]

                peak_vertex = clus_indices[0]
                max_value = abs(result[hemisphere].t[0])[peak_vertex]

                for index in clus_indices:
                    if abs(result[hemisphere].t[0][index]) > max_value:
                        max_value = result[hemisphere].t[0][index]
                        peak_vertex = index

                # peak_vertex = list(result[hemisphere].P['peak'][posneg][result[hemisphere].P['peak'][posneg].clusid == clusid].vertid)[0]

                anatomical_label = labels[peak_vertex]
                anatomical_loc = aal_full[aal_full.val == anatomical_label].name.squeeze()
                anatomical_loc = anatomical_loc.replace(' left', '')
                anatomical_loc = anatomical_loc.replace(' right', '')

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

                cluster_summary = pd.concat([cluster_summary, pd.DataFrame({'clusid': [clusid], 'Anatomical location (peak)': anatomical_loc, 'Hemisphere': hemisphere, 'Cluster area (mm2)': f'{area:.0f}', 'MNI coordinates (x,y,z)': str(peak_coord), 'Cluster FWE p-value': f'{clus_pval:.2g}'})], ignore_index=True)

    return cluster_summary