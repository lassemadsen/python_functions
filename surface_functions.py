"""This script contain useful functions when working with cortical surface data in python.

Many functions are dependent on the Brainstat python module: https://brainstat.readthedocs.io/en/master/

Author: Lasse Stensvig Madsen
Mail: lasse.madsen@cfin.au.dk

Last edited: 17/2 - 2025
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
                   plot=False, outdir=None, group_names=('Group 1', 'Group2'), param_name=None, clobber=False, **kwargs):
    """
    Perform an unpaired t-test on the two groups of data.
    
    Parameters
    ----------
    data_group1 : dict of DataFrame
        A dictionary containing the data for the first group of subjects, with 'left' and 'right' as keys.
    data_group2 : dict of DataFrame
        A dictionary containing the data for the second group of subjects, with 'left' and 'right' as keys.
    covars : pandas DataFrame, optional
        Dataframe with covariates (subject_id as index and data as columns)
    correction : str or None, optional | 'rft'
        Correction method for multiple comparisons. If None, no correction is performed (default: 'rft').
    cluster_threshold : float, optional
        Primary cluster defining threshold (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    plot : bool, optional
        If True, generate and save result plots (default: False).
    outdir : str or None, optional
        Directory where output plots and results will be saved. If None, no output is saved.
    group_names : tuple of str, optional
        Names of the two groups being compared (default: ('Group 1', 'Group 2')).
    param_name : str or None, optional
        Name of the parameter being analyzed. If None, a warning will be printed.
    clobber : bool, optional
        If True, overwrite existing output files (default: False).
    **kwargs : dict
        Additional keyword arguments for plotting functions.
    
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    cluster_mask : dict of np.ndarray
        A dictionary containing the binary cluster mask for significant clusters for each hemisphere, with 'left' and 'right' as keys.
    cluster_summary : pandas DataFrame or None
        A DataFrame containing information about each significant cluster if correction is applied. The DataFrame includes columns:
        'hemisphere' (left or right), 'x', 'y', 'z' (coordinates of cluster peak), 'size' (cluster size), and 'p' (corrected p-value).
        Returns None if no correction is applied.
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
        group1_subjects = sorted(list(set(group1_subjects) & set(covars.index)))
        group2_subjects = sorted(list(set(group2_subjects) & set(covars.index)))

        for covar in covars: 
            covar_term = covar_term + FixedEffect(pd.concat([covars.loc[group1_subjects, covar],covars.loc[group2_subjects, covar]], names=covar))

    print(f'Group 1: N={len(group1_subjects)}, group 2: N={len(group2_subjects)}')

    groups = pd.DataFrame({'group': ['0']*len(group1_subjects) + ['1']*len(group2_subjects)})

    # Calculate unpaired t-test
    for hemisphere in ['left', 'right']:
        data = pd.concat([data_group1[hemisphere][group1_subjects], data_group2[hemisphere][group2_subjects]], axis=1).T

        # Get mask 
        mask = ~data.isna().any(axis=0).values 

        # Brainstat RFT correction does not work well with mask. Values not in mask are set to 0.
        # Note: Unsure how well this works if many vertices are nan.
        data.iloc[:,~mask] = 0

        term_groups = FixedEffect(groups)
        model = term_groups

        if covars is not None:
            model = model + covar_term

        contrast = term_groups.group_1 - term_groups.group_0

        # slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)
        slm.mask = mask

        result[hemisphere] = slm
    
    cluster_mask = get_cluster_mask(result, correction, alpha)
    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result, alpha)

    if plot:
        if param_name is None:
            print('Please set parameter name!')
        elif outdir is None:
            print('Please specify outdir.')
        else:
            if covars is None:
                outdir = f'{outdir}/Unpaired_ttest/{group_names[0].replace(" ", "_")}_vs_{group_names[1].replace(" ", "_")}/{param_name.replace(" ", "_")}'
                basename = f'{param_name.replace(" ", "_")}_p{cluster_threshold}'
                outfile_fwe_corrected = f'{outdir}/{basename}_fweCorrected.jpg'
                outfile_uncorrected = f'{outdir}/{basename}_uncorrected.jpg'
                cluster_summary_file = f'{outdir}/ClusterSum_{basename}_fweCorrected.csv'

            else:
                outdir = f'{outdir}/Unpaired_ttest/{group_names[0].replace(" ", "_")}_vs_{group_names[1].replace(" ", "_")}/{param_name.replace(" ", "_")}+{"+".join(covars.columns)}'
                basename = f'{param_name.replace(" ", "_")}+{"+".join(covars.columns)}_p{cluster_threshold}'
                outfile_fwe_corrected = f'{outdir}/{basename}_fweCorrected.jpg'
                outfile_uncorrected = f'{outdir}/{basename}_uncorrected.jpg'
                cluster_summary_file = f'{outdir}/ClusterSum_{basename}_fweCorrected.csv'

            print(f'Plotting results to {outdir}...')
            # ---- Calculate mean for each group ---- 
            mean_data = {'Group1': {'left': data_group1['left'][group1_subjects].mean(axis=1), 'right': data_group1['right'][group1_subjects].mean(axis=1)},
                         'Group2': {'left': data_group2['left'][group2_subjects].mean(axis=1), 'right': data_group2['right'][group2_subjects].mean(axis=1)}}

            # ---- Plot results ----
            t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
            mask = {'left': result['left'].mask, 'right': result['right'].mask}
            mean_titles = [f'{group_names[0]} (n={len(group1_subjects)})', f'{group_names[1]} (n={len(group2_subjects)})']
            if correction is not None:
                plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_fwe_corrected, 
                                                p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, 
                                                mean_titles=mean_titles, stats_titles='Difference', cluster_mask=cluster_mask, 
                                                mask=mask, t_lim=[-5, 5], clobber=clobber, 
                                                cb_mean_title=f'Mean {param_name}', **kwargs)
                cluster_plot.boxplot({'left': data_group1['left'][group1_subjects], 'right': data_group1['right'][group1_subjects]}, 
                                        {'left': data_group2['left'][group2_subjects], 'right': data_group2['right'][group2_subjects]},
                                        result, outdir, group_names[0], group_names[1], param_name, alpha=alpha, 
                                        cluster_summary=cluster_summary, clobber=clobber)
                cluster_summary.to_csv(cluster_summary_file)

            plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_uncorrected, 
                                            p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, 
                                            mean_titles=mean_titles, stats_titles='Difference', t_lim=[-5, 5], 
                                            mask=mask, cb_mean_title=f'Mean {param_name}', clobber=clobber, **kwargs)

    return result, cluster_mask, cluster_summary

def paired_ttest(data1, data2, correction='rft', cluster_threshold=0.001, alpha=0.05, 
                 plot=False, outdir=None, group_names=('Group 1', 'Group2'), param_name=None, 
                 clobber=False, **kwargs):
    """
    Perform a paired t-test on the data.
    
    Parameters
    ----------
    data1 : dict of DataFrame
        A dictionary containing the data for the first set of measurements with 'left' and 'right' as keys.
    data2 : dict of DataFrame
        A dictionary containing the data for the second set of measurements with 'left' and 'right' as keys.
    correction : str or None, optional | 'rft'
        Correction method for multiple comparisons. If None, no correction is performed.
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    plot : bool, optional
        If True, generate and save result plots (default: False).
    outdir : str or None, optional
        Directory where output plots and results will be saved. If None, no output is saved.
    group_names : tuple of str, optional
        Names of the two groups being compared (default: ('Group 1', 'Group 2')).
    param_name : str or None, optional
        Name of the parameter being analyzed. If None, a warning will be printed.
    clobber : bool, optional
        If True, overwrite existing output files (default: False).
    **kwargs : dict
        Additional keyword arguments for plotting functions.
    
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    common_subjects : list
        List containing IDs used in the analysis (included in both data1 and data2)
    cluster_mask : dict of np.ndarray
        A dictionary containing the binary cluster mask for significant clusters for each hemisphere, with 'left' and 'right' as keys.
    cluster_summary : pandas DataFrame or None
        A DataFrame containing information about each significant cluster if correction is applied. The DataFrame includes columns:
        'hemisphere' (left or right), 'x', 'y', 'z' (coordinates of cluster peak), 'size' (cluster size), and 'p' (corrected p-value).
        Returns None if no correction is applied.
    

    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return
    if plot:
        if param_name is None:
            print('Please set parameter name when plot=True.')
            return
        elif outdir is None:
            print('Please specify outdir when plot=True.')
            return

    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(data1['left'].columns) & set(data2['left'].columns)))
    print(f'N={len(common_subjects)}')

    measurements = pd.DataFrame({'measurements': ['0']*len(common_subjects) + ['1']*len(common_subjects)})

    for hemisphere in ['left', 'right']:
        data = pd.concat([data1[hemisphere][common_subjects], data2[hemisphere][common_subjects]], axis=1).T

        # Get mask 
        mask = ~data.isna().any(axis=0).values 

        # Brainstat RFT correction does not work well with mask. Values not in mask are set to 0.
        # Note: Unsure how well this works if many vertices are nan.
        data.iloc[:,~mask] = 0

        term_meas = FixedEffect(measurements, add_intercept=False)
        term_subject = FixedEffect(common_subjects*2, add_intercept=False)

        model = term_meas + term_subject
        contrast = term_meas.measurements_1 - term_meas.measurements_0

        #slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)
        slm.mask = mask

        result[hemisphere] = slm
    
    cluster_mask = get_cluster_mask(result, correction, alpha)

    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result, alpha)

    if plot:
        outdir = f'{outdir}/Paired_ttest/{group_names[0].replace(" ", "_")}_vs_{group_names[1].replace(" ", "_")}/{param_name.replace(" ", "_")}'
        basename = f'{param_name.replace(" ", "_")}_p{cluster_threshold}'
        outfile_fwe_corrected = f'{outdir}/{basename}_fweCorrected.jpg'
        outfile_uncorrected = f'{outdir}/{basename}_uncorrected.jpg'
        cluster_summary_file = f'{outdir}/ClusterSum_{basename}_fweCorrected.csv'

        Path(outdir).mkdir(exist_ok=True, parents=True)
        print(f'Plotting results to {outdir}...')
        # ---- Calculate mean for each group ---- 
        mean_data = {'Group1': {'left': data1['left'][common_subjects].mean(axis=1), 'right': data1['right'][common_subjects].mean(axis=1)},
                    'Group2': {'left': data2['left'][common_subjects].mean(axis=1), 'right': data2['right'][common_subjects].mean(axis=1)}}

        # ---- Plot results ----
        mask = {'left': result['left'].mask, 'right': result['right'].mask}
        t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
        mean_titles = [f'{group_names[0]} (n={len(common_subjects)})', f'{group_names[1]} (n={len(common_subjects)})']
        if correction is not None:
            plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_fwe_corrected, 
                                            p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, 
                                            mean_titles=mean_titles, stats_titles='Difference', cluster_mask=cluster_mask, 
                                            mask=mask, t_lim=[-5, 5], clobber=clobber, cb_mean_title=f'Mean {param_name}', **kwargs)
            cluster_plot.boxplot({'left': data1['left'][common_subjects], 'right': data1['right'][common_subjects]},
                                {'left': data2['left'][common_subjects], 'right': data2['right'][common_subjects]},
                                result, outdir, group_names[0], group_names[1], param_name, paired=True,
                                cluster_summary=cluster_summary, alpha=alpha, clobber=clobber)
            cluster_summary.to_csv(cluster_summary_file)
        plot_mean_stats.plot_mean_stats(mean_data['Group1'], mean_data['Group2'], t_value, outfile_uncorrected,
                                        p_threshold=cluster_threshold, df=result['left'].df, plot_tvalue=True, 
                                        mean_titles=mean_titles, stats_titles='Difference', t_lim=[-5, 5], mask=mask,
                                        clobber=clobber, cb_mean_title=f'Mean {param_name}', **kwargs)
    
    return result, common_subjects, cluster_mask, cluster_summary

def correlation(surface_data, indep_data, correction='rft', cluster_threshold=0.001, alpha=0.05, 
                plot=False, outdir=None, dep_name=None, clobber=False, **kwargs):
    """
    Calculate the correlation of surface with value (e.g. demography data such as age or cognitive score)

    Parameters
    ----------
    surface_data : dict of DataFrame
        A dictionary containing the data for the surface measurements with 'left' and 'right' as keys.
    indep_data : DataFrame
        Pandas dataframe with subject_id as index and data as columns
        If more than one columns, the rest are considered covariates.
    correction : str, optional | 'rft'
        Multiple comparison correction: 'rft' or 'fdr'.
    cluster_threshold : float, optional
        Primary cluster defining threshold (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    plot : bool, optional
        If True, generate and save result plots (default: False).
    outdir : str or None, optional
        Directory where output plots and results will be saved. If None, no output is saved.
    dep_name : str or None, optional
        Name of the dependent variable. Has to be set if plot=True.
    clobber : bool, optional
        If True, overwrite existing output files (default: False).
    **kwargs : dict
        Additional keyword arguments for plotting functions.
        
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    common_subjects : list
        List containing IDs used in the analysis (included in both data1 and data2)
    cluster_mask : dict of np.ndarray
        A dictionary containing the binary cluster mask for significant clusters for each hemisphere, with 'left' and 'right' as keys.
    cluster_summary : pandas DataFrame or None
        A DataFrame containing information about each significant cluster if correction is applied. The DataFrame includes columns:
        'hemisphere' (left or right), 'x', 'y', 'z' (coordinates of cluster peak), 'size' (cluster size), and 'p' (corrected p-value).
        Returns None if no correction is applied.
    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return
    if plot:
        if dep_name is None:
            print('Please set parameter name when plot=True.')
            return
        elif outdir is None:
            print('Please specify outdir when plot=True.')
            return

    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data['left'].columns) & set(surface_data['right'].columns) & set(indep_data.index)))

    for hemisphere in ['left', 'right']:
        data = surface_data[hemisphere][common_subjects].T

        # Get mask
        mask = ~data.isna().any(axis=0).values 

        # Brainstat RFT correction does not work well with mask. Values not in mask are set to 0.
        # Note: Unsure how well this works if many vertices are nan.
        data.iloc[:,~mask] = 0

        terms = {}
        model = []
        for var in indep_data: 
            terms[var] = FixedEffect(indep_data.loc[common_subjects, var], names=var)

            model = model + terms[var]

        contrast = indep_data.loc[common_subjects,indep_data.columns[0]].values

        # --- Run model ---
        # slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold, mask=mask)
        slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
        slm.fit(data.values)
        slm.mask = mask

        result[hemisphere] = slm
    
    cluster_mask = get_cluster_mask(result, correction, alpha)
    
    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result, alpha)

    if plot:
        indep_names = indep_data.columns
        outdir = f'{outdir}/Correlation/{dep_name}_vs_{"+".join(indep_names)}'
        basename = f'{dep_name}_vs_{"+".join(indep_names)}_p{cluster_threshold}'
        outfile_fwe_corrected = f'{outdir}/{basename}_fweCorrected.jpg'
        outfile_uncorrected = f'{outdir}/{basename}_uncorrected.jpg'
        cluster_summary_file = f'{outdir}/ClusterSum_{basename}_fweCorrected.csv'

        Path(outdir).mkdir(exist_ok=True, parents=True)
        print(f'Plotting results to {outdir}...')

        # ---- Plot results ----
        mask = {'left': result['left'].mask, 'right': result['right'].mask}
        t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
    
        title = f'{dep_name}~{indep_names[0]} (n={len(common_subjects)})'
        if correction is not None:
            plot_stats.plot_tval(t_value, outfile_fwe_corrected, p_threshold=cluster_threshold, df=result['left'].df, 
                                 cluster_mask=cluster_mask, mask=mask, t_lim=[-5, 5], title=title, cbar_loc='left', 
                                 clobber=clobber, **kwargs)
            cluster_plot.correlation_plot(result, {'left': surface_data['left'][common_subjects], 'right': surface_data['right'][common_subjects]},
                                          indep_data.loc[common_subjects, :], dep_name, indep_names[0], outdir, cluster_summary=cluster_summary, 
                                          alpha=alpha, clobber=clobber)
            cluster_summary.to_csv(cluster_summary_file)
            
        plot_stats.plot_tval(t_value, outfile_uncorrected, p_threshold=cluster_threshold, df=result['left'].df, 
                                mask=mask, t_lim=[-5, 5], title=title, cbar_loc='left', clobber=clobber, **kwargs)

    # TODO test plotting
    
    return result, common_subjects, cluster_mask, cluster_summary

def correlation_pearson(param, indep_data):
    """
    Pearson correlation of surface with value (e.g. demography data such as age or cognitive score) or surface

    Parameters
    ----------
    data : dict('left', 'right')
        Independent surface data
    indep_data : DataFrame
        Pandas dataframe with subject_id as column headers (same id as in surface_data)
        If more than one row, the rest of the rows are considere covariates
    """
    
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(param['left'].columns) & set(param['right'].columns) & 
                                  set(indep_data['left'].columns) & set(indep_data['right'].columns)))

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
            r_corr, _ = pearsonr(indep_data[hemisphere][common_subjects].iloc[i,:].values.T, data[[i]])
            r[i] = r_corr[0]

        result[hemisphere] = r
    
    return result, common_subjects

def correlation_other_surface(surface_data_dep, surface_data_indep, covars=None, correction='rft', 
                              cluster_threshold=0.001, alpha=0.05, plot=False, outdir=None, group_name=None, 
                              dep_name=None, indep_name=None, quadratic=False, clobber=False, **kwargs):
    """
    Calculate the correlation of two surfaces 

    Parameters
    ----------
    surface_data_dep : dict of DataFrame
        A dictionary containing the data for the first surface measurements with 'left' and 'right' as keys.
    surface_data_indep : dict of DataFrame
        A dictionary containing the data for the second surface measurements with 'left' and 'right' as keys.
    covars : pandas DataFrame, optional
        Dataframe with covariates (subject_id as index and data as columns)
    correction : str or 'None', optional | 'rft'
        Correction method for multiple comparisons. If None, no correction is performed (default: 'rft').
    cluster_threshold : float, optional
        Primary cluster defining threshold (default 0.001).
    alpha : float, optional
        Threshold of corrected clusters (default 0.05).
    plot : bool, optional
        If True, generate and save result plots (default: False).
    outdir : str or None, optional
        Directory where output plots and results will be saved. If None, no output is saved.
    dep_name : str or None, optional
        Name of the dependent variable. Has to be set if plot=True.
    indep_name : str or None, optional
        Name of the independent variable. Has to be set if plot=True.
    clobber : bool, optional
        If True, overwrite existing output files (default: False).
    **kwargs : dict
        Additional keyword arguments for plotting functions.
        
    Returns
    -------
    result : dict
        A dictionary containing the SLM results for each hemisphere, with 'left' and 'right' as keys.
    common_subjects : list
        List containing IDs used in the analysis (included in both data1 and data2)
    cluster_mask : dict of np.ndarray
        A dictionary containing the binary cluster mask for significant clusters for each hemisphere, with 'left' and 'right' as keys.
    cluster_summary : pandas DataFrame or None
        A DataFrame containing information about each significant cluster if correction is applied. The DataFrame includes columns:
        'hemisphere' (left or right), 'x', 'y', 'z' (coordinates of cluster peak), 'size' (cluster size), and 'p' (corrected p-value).
        Returns None if no correction is applied.
    """
    if not correction in {'rft', 'fdr'} and correction is not None:
        print('Wrong correction method! Should be "rft" or "fdr" or None. Please try again.')
        return

    if plot:
        if indep_name is None or dep_name is None:
            print('Please set parameter names when plot=True.')
            return
        elif outdir is None:
            print('Please specify outdir when plot=True.')
            return
    
    result = {'left': [], 'right': []}

    common_subjects = sorted(list(set(surface_data_dep['left'].columns) & set(surface_data_dep['right'].columns) & 
                                  set(surface_data_indep['left'].columns) & set(surface_data_indep['right'].columns)))
                                  
    # Define covariates, if any
    if covars is not None:
        # Make sure all subjects have covars
        common_subjects = sorted(list(set(common_subjects) & set(covars.index)))

        covar_term = None
        for covar in covars:
            covar_term = covar_term + FixedEffect(covars.loc[common_subjects, covar], names=covar)

    for hemisphere in ['left', 'right']:
        data_dep = surface_data_dep[hemisphere][common_subjects].T
        data_indep = surface_data_indep[hemisphere][common_subjects].T

        # Get mask
        mask = (~data_dep.isna().any(axis=0) & ~data_indep.isna().any(axis=0)).values

        # Brainstat RFT correction does not work well with mask. Values not in mask are set to 0.
        # Note: Unsure how well this works if many vertices are nan.
        data_dep.iloc[:,~mask] = 0
        data_indep.iloc[:,~mask] = 0
        
        # Initialise t values to nan
        t = np.zeros(mask.shape)
        t[:] = np.nan
        
        vert_list = np.where(mask==True)[0]

        # Run model for each vertex
        for i in vert_list:
            # --- Correlation with other surface ---
            term = FixedEffect(data_indep[i].values)

            if quadratic:
                quad_term = FixedEffect(data_indep[i].values ** 2, names=['quad'], add_intercept=False)
                model = term + quad_term
                contrast = model.quad
            else:
                model = term
                contrast = model.x0

            if covars is not None:
                model = model + covar_term

            # --- Run model ---
            slm = SLM(model, contrast)
            slm.fit(data_dep[[i]])

            t[i] = slm.t[0][0]
        
        if correction is not None:
            # To run correction with two surfaces, we have to run  with mean data for one of them to compute multple comparison.
            # However, when calculating random filed theory correction, it matters which surface is the dependent and indpendent variable. 
            # This is because the correction estimates the number of resolution elements (ressels) based on residuals. An more noisy surface map will give different
            # values compared to a more smooth. To make sure there is not difference in results depending of which surface is set to indep/dep variable respectivly, 
            # the correction is run with both where ressels are calculated. Finally, the average ressels across the two runs is used to produce the final correction, which 
            # is independent on which surface is first. 

            # 1. Run with indep_data as fit()
            term = FixedEffect(surface_data_dep[hemisphere][common_subjects].mean().values)
            if quadratic:
                quad_term = FixedEffect(surface_data_dep[hemisphere][common_subjects].mean().values ** 2, names=['quad'], add_intercept=False)
                model = term + quad_term
                contrast = model.quad
            else:
                model = term
                contrast = model.x0

            if covars is not None:
                model = model + covar_term

            slm1 = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
            slm1.fit(data_indep.values) 
            slm1.t = np.array([t])
            slm1.multiple_comparison_corrections(True) # Run with actual t-values

            # 2. Run with dep_data as fit()
            term = FixedEffect(surface_data_indep[hemisphere][common_subjects].mean().values)
            if quadratic:
                quad_term = FixedEffect(surface_data_indep[hemisphere][common_subjects].mean().values ** 2, names=['quad'], add_intercept=False)
                model = term + quad_term
                contrast = model.quad
            else:
                model = term
                contrast = model.x0

            if covars is not None:
                model = model + covar_term

            slm2 = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
            slm2.fit(data_dep.values) 
            slm2.t = np.array([t])
            slm2.multiple_comparison_corrections(True) # Run with actual t-values

            # 3. Finally run where slm.resls is set to the average of the two (whether fit is run with indep or dep does not change cluster p-values when slm.resl are set before)
            term = FixedEffect(surface_data_indep[hemisphere][common_subjects].mean().values)
            if quadratic:
                quad_term = FixedEffect(surface_data_indep[hemisphere][common_subjects].mean().values ** 2, names=['quad'], add_intercept=False)
                model = term + quad_term
                contrast = model.quad
            else:
                model = term
                contrast = model.x0

            if covars is not None:
                model = model + covar_term

            slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
            slm.fit(data_dep.values) 
            slm.t = np.array([t])
            slm.resl = (slm1.resl + slm2.resl) / 2
            slm.multiple_comparison_corrections(True) # Run with actual t-values

        else:
            term = FixedEffect(surface_data_dep[hemisphere][common_subjects].mean().values)
            if quadratic:
                quad_term = FixedEffect(surface_data_dep[hemisphere][common_subjects].mean().values ** 2, names=['quad'], add_intercept=False)
                model = term + quad_term
                contrast = model.quad
            else:
                model = term
                contrast = model.x0

            if covars is not None:
                model = model + covar_term
        
            slm = SLM(model, contrast, surf=surf[hemisphere], correction=correction, cluster_threshold=cluster_threshold)
            slm.t = np.array([t])

        slm.mask = mask

        result[hemisphere] = slm

    cluster_mask = get_cluster_mask(result, correction, alpha)
    if correction is None:
        cluster_summary = None
    else:
        cluster_summary = get_cluster_summary(result, alpha)

    if group_name is not None:
        outdir = f'{outdir}/Correlation/{group_name}'
    else:
        outdir = f'{outdir}/Correlation'

    if quadratic:
        outdir = outdir.replace('Correlation', 'Correlation_quad')
    
    if plot:
        if covars is None:
            basename = f'{dep_name}_vs_{indep_name}_p{cluster_threshold}'
            outdir = f'{outdir}/{dep_name}_vs_{indep_name}'
        else:
            basename = f'{dep_name}_vs_{indep_name}+{"+".join(covars.columns)}_p{cluster_threshold}'
            outdir = f'{outdir}/{dep_name}_vs_{indep_name}+{"+".join(covars.columns)}'

        outfile_fwe_corrected = f'{outdir}/{basename}_fweCorrected.jpg'
        outfile_uncorrected = f'{outdir}/{basename}_uncorrected.jpg'
        cluster_summary_file = f'{outdir}/ClusterSum_{basename}_fweCorrected.csv'

        Path(outdir).mkdir(exist_ok=True, parents=True)
        print(f'Plotting results to {outdir}...')

        # ---- Plot results ----
        mask = {'left': result['left'].mask, 'right': result['right'].mask}
        t_value = {'left': result['left'].t[0], 'right': result['right'].t[0]}
    
        title = f'{dep_name}~{indep_name} (n={len(common_subjects)})'
        if correction is not None:
            plot_stats.plot_tval(t_value, outfile_fwe_corrected, p_threshold=cluster_threshold, df=result['left'].df, 
                                 cluster_mask=cluster_mask, mask=mask, t_lim=[-5, 5], title=title, cbar_loc='left', 
                                 clobber=clobber, **kwargs)
            cluster_plot.correlation_plot(result, {'left': surface_data_dep['left'][common_subjects], 'right': surface_data_dep['right'][common_subjects]},
                                          {'left': surface_data_indep['left'][common_subjects], 'right': surface_data_indep['right'][common_subjects]},
                                          dep_name, indep_name, outdir, cluster_summary=cluster_summary, alpha=alpha, quadratic=quadratic, clobber=clobber)
            cluster_summary.to_csv(cluster_summary_file)
            
        plot_stats.plot_tval(t_value, outfile_uncorrected, p_threshold=cluster_threshold, df=result['left'].df, 
                             mask=mask, t_lim=[-5, 5], title=title, cbar_loc='left', clobber=clobber, **kwargs)

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

def get_cluster_summary(result, alpha):
    """
    Calculate summary of surviving clusters.

    Parameters:
    -----------
    result : dict
        A dictionary containing the results of a statistical analysis for each hemisphere with 'left' and 'right' as keys.
    alpha : float
        Threshold of corrected clusters.

    Returns:
    --------
    cluster_summary : pandas DataFrame
        A DataFrame containing information about each significant cluster if correction is applied. The DataFrame includes columns:
        'hemisphere' (left or right), 'x', 'y', 'z' (coordinates of cluster peak), 'size' (cluster size), and 'p' (corrected p-value).
        Returns None if no correction is applied.
    """
    cluster_summary = pd.DataFrame({'clusid': [], 
                                    'Anatomical location (peak)': [], 
                                    'Hemisphere': [], 
                                    'Cluster area (mm2)': [],
                                    'MNI coordinates (x,y,z)': [], 
                                    'Cluster FWE p-value': []})

    aal_full = pd.read_csv(ATLAS_LOOKUP, names=['val', 'name'])

    for hemisphere in ['left', 'right']:
        mni_coord = result[hemisphere].surf.Points
        labels = np.loadtxt(ATLAS_LABELS[hemisphere], skiprows=1)
        for posneg in ['pos','neg']:
            if posneg == 'pos':
                posneg_idx = 0
            else:
                posneg_idx = 1
            cluster_survived = result[hemisphere].P['clus'][posneg_idx][result[hemisphere].P['clus'][posneg_idx].P < alpha]

            if cluster_survived.empty:
                continue

            for clusid in cluster_survived.clusid:
                clus_pval = result[hemisphere].P['clus'][posneg_idx][result[hemisphere].P['clus'][posneg_idx].clusid == clusid].P.values[0]

                # Find peak vertex
                clus_indices = np.where(result[hemisphere].P['clusid'][posneg_idx] == clusid)[1]

                peak_vertex = clus_indices[0]
                max_value = abs(result[hemisphere].t[0])[peak_vertex]

                for index in clus_indices:
                    if abs(result[hemisphere].t[0][index]) > max_value:
                        max_value = result[hemisphere].t[0][index]
                        peak_vertex = index

                anatomical_label = labels[peak_vertex]
                anatomical_loc = aal_full[aal_full.val == anatomical_label].name.squeeze()
                anatomical_loc = anatomical_loc.replace(' left', '')
                anatomical_loc = anatomical_loc.replace(' right', '')

                peak_coord = mni_coord[peak_vertex]
                peak_coord = [round(c) for c in peak_coord] # Round coordinates
                
                idx = np.where(result[hemisphere].P['clusid'][posneg_idx][0] == clusid)[0]
                polys = result[hemisphere].surf.polys2D[np.isin(result[hemisphere].surf.polys2D, idx).all(axis=1)]

                area = 0
                for p in polys:
                    a = mni_coord[p[0]]
                    b = mni_coord[p[1]]
                    c = mni_coord[p[2]]

                    x = np.cross((a - b), (b - c))
                    A = np.sqrt(x.dot(x)) / 2

                    area += A

                cluster_summary = pd.concat([cluster_summary, pd.DataFrame({'clusid': [clusid], 
                                                                            'sign_t': [posneg],
                                                                            'Anatomical location (peak)': anatomical_loc, 
                                                                            'Hemisphere': hemisphere, 
                                                                            'Cluster area (mm2)': f'{area:.0f}', 
                                                                            'MNI coordinates (x,y,z)': str(peak_coord), 
                                                                            'Cluster FWE p-value': f'{clus_pval:.2g}'})], ignore_index=True)

    return cluster_summary