import subprocess
import os
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SURFACE_TEMPLATE = '/public/fristed/data/atlas/surface/mni_icbm152_t1_tal_nlin_sym_09c'
SURFACE_BLUR = 20

def map_to_surface(param_data, t1_to_param_transform, t1t2_pipeline, mr_id, timepoint, param_type, outdir, out_id=None, clean_surface=False):
    """Map parameter signals to mid surface 

    Parameters
    ----------
    param_data : str
        minc file of parameter data (in native space)
    t1_to_param_transform : str
        .xfm file with transformation from T1 native space to parameter native space
    t1t2pipeline : str
        Folder of t1t2pipeline containing FACE files
    mr_id : str
        MR ID of subject
    timepoint : str
        Timestamp of the MR image (where FACE is located)
    parameter_type: str
        Type of parameter
    outdir : str
        Location of outputs
    out_id : str | None
        Set alternative ID for output files (eg. collected pet_mr id)
    clean_surface : boolean | False
        If true, values equal -1 on the non-smoothed data and values less the 0 on the smoothed data are set to -1 (used for vertices outside FOV)
    """

    if out_id is None:
        out_id = mr_id

    mr_path = f'{t1t2_pipeline}/{mr_id}/{timepoint}'

    for hemisphere in ['left', 'right']:
        process_list = []
        # ----- Create surface -----
        # 1. Generate mid surface
        # 2. Invert transform matrix from native to mni space
        # 3. Move surface to native subject space using inverted transformation
        # 4. Move to parameter native space

        mid_surface = f'{outdir}/{out_id}_{timepoint}_mid_{hemisphere}_{param_type}.obj'

        process_list.extend([
            f'midsurface.bin {mr_path}/face/surfaces/world/inner_{hemisphere}.obj {mr_path}/face/surfaces/world/outer_{hemisphere}.obj {mr_path}/face/measurements/outer_{hemisphere}.corr {mr_path}/face/surfaces/world/mid_{hemisphere}.obj',
            f'xfminvert {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1.xfm {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm',
            f'transform_objects {mr_path}/face/surfaces/world/mid_{hemisphere}.obj {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm {mr_path}/face/surfaces/native/mid_{hemisphere}.obj',
            f'transform_objects {mr_path}/face/surfaces/native/mid_{hemisphere}.obj {t1_to_param_transform} {mid_surface}'])
    
        # ----- Map signals to surface -----
        # 1. Map parameter data onto surface
        # 2. Move surface to standard MNI space
        # 3. Blur signal on surface (along cortex) 

        mapping = f'{t1t2_pipeline}/{mr_id}/{timepoint}/face/mapping/{hemisphere}.corr'

        out_surface_prefix = f'{outdir}/{out_id}_{timepoint}_mid_{hemisphere}_{param_type}'
        surface_template = f'{SURFACE_TEMPLATE}_{hemisphere}_smooth.obj'

        process_list.extend([
            f'surfacesignals.bin {param_data} {mid_surface} {out_surface_prefix}.dat',
            f'map_measurements.bin {surface_template} {mid_surface} {mapping} {out_surface_prefix}.dat > {out_surface_prefix}_std.dat',
            f'blur_measurements.bin -iter {SURFACE_BLUR} {surface_template} {out_surface_prefix}_std.dat > {out_surface_prefix}_std_blur{SURFACE_BLUR}.dat'])

        succes = _run_process(process_list, out_id, timepoint, hemisphere, param_type)

        if succes:
            if clean_surface:
                _clean_surface_after_smoothing(f'{out_surface_prefix}_std.dat', f'{out_surface_prefix}_std_blur{SURFACE_BLUR}.dat')

def _run_process(process_list, sub_id, timepoint, hemisphere, measurement):
    """Run each command of procces list in bash 

    Parameters
    ----------
    process_list : list
        List of commands to be run in bash.
    sub_id : str
        ID of subject.
    timepoint : str
        Timestamp of measurement that is run.
    hemisphere : str
        Hemisphere (used for logging).
    measurement : str
        Type of measurement (used for logging).

    Return
    ------
    succes : bool
        Return 1 if all process were run succesfully, return 0 otherwise
    """

    output_file = process_list[-1].split(' ')[-1] # The final output file of the procces list.
    # If the ouput_file exists, the entire process list is skipped
    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        logger.info(f'{output_file} exists. Skipping...')
        return
    
    # Start processing    
    logger.info(f'{sub_id}, {timepoint}: mapping {measurement} to {hemisphere} surface')

    for process in process_list:
        if os.path.isfile(process.split(' ')[-1]) and os.path.getsize(process.split(' ')[-1]) > 0: # Check if output file of each process already exists
            continue

        try:
            subprocess.check_output(process, shell=True, stderr=subprocess.STDOUT)
            succes = 1 
        except:
            logger.error(f'Error when proccesing: {process}')
            succes = 0
            return succes
    
    return succes

def _clean_surface_after_smoothing(not_smoothed, smoothed):
    """Script to clean surface after smoothing to ensure that vertices outside FOV is set to -1
    Furthermore, all values under 0 is set to -1.
    
    Parameter
    ---------
    not_smoothed : str
        File location of parameter map before smoothing
    smoothed : str
        File location of parameter map after smoothing
    """

    ns = pd.read_csv(not_smoothed)
    s = pd.read_csv(smoothed)

    s[ns==-1] = -1
    s[s < 0] = -1

    s.to_csv(smoothed, index=False)
    