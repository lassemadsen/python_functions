"""This script contain useful funcitons when working with cortical surface data in python
"""

import numpy as np
import pandas as pd

PUBLIC_PATH='/public/lama'

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



