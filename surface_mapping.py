import subprocess
import os
import logging
import pandas as pd
import numpy as np
import glob
from brainspace.mesh.mesh_io import read_surface

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

SURFACE_GII = {'left': '/public/lama/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.gii',
               'right': '/public/lama/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.gii'}
SURFACE_OBJ = {'left': '/public/lama/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_left_smooth.obj',
               'right': '/public/lama/data/surface/mni_icbm152_t1_tal_nlin_sym_09c_right_smooth.obj'}
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

        if t1_to_param_transform is not None:
            process_list.extend([
                f'midsurface.bin {mr_path}/face/surfaces/world/inner_{hemisphere}.obj {mr_path}/face/surfaces/world/outer_{hemisphere}.obj {mr_path}/face/measurements/outer_{hemisphere}.corr {mr_path}/face/surfaces/world/mid_{hemisphere}.obj',
                f'xfminvert {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1.xfm {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm',
                f'transform_objects {mr_path}/face/surfaces/world/mid_{hemisphere}.obj {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm {mr_path}/face/surfaces/native/mid_{hemisphere}.obj',
                f'transform_objects {mr_path}/face/surfaces/native/mid_{hemisphere}.obj {t1_to_param_transform} {mid_surface}'])
        else:
            if param_type is 'thickness':
                # Cortical thickness data is already mapped to MNI surface
                pass
            else: 
                process_list.extend([
                    f'midsurface.bin {mr_path}/face/surfaces/world/inner_{hemisphere}.obj {mr_path}/face/surfaces/world/outer_{hemisphere}.obj {mr_path}/face/measurements/outer_{hemisphere}.corr {mr_path}/face/surfaces/world/mid_{hemisphere}.obj',
                    f'xfminvert {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1.xfm {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm',
                    f'transform_objects {mr_path}/face/surfaces/world/mid_{hemisphere}.obj {mr_path}/stx2/stx2_{mr_id}_{timepoint}_t1_inv.xfm {mr_path}/face/surfaces/native/mid_{hemisphere}.obj',
                    f'cp {mr_path}/face/surfaces/native/mid_{hemisphere}.obj {mid_surface}'])
        
        # ----- Map signals to surface -----
        # 1. Map parameter data onto surface
        # 2. Move surface to standard MNI space
        # 3. Blur signal on surface (along cortex) 

        mapping = f'{t1t2_pipeline}/{mr_id}/{timepoint}/face/mapping/{hemisphere}.corr'

        out_surface_prefix = f'{outdir}/{out_id}_{timepoint}_mid_{hemisphere}_{param_type}'

        if param_type is 'thickness':
            # Cortical thickness data is already in MNI space. Only needs blurring
            process_list.extend([
                f'cp {t1t2_pipeline}/{mr_id}/{timepoint}/face/mapping/{hemisphere}.dist {out_surface_prefix}_std.dat',
                f'blur_measurements.bin -iter {SURFACE_BLUR} {SURFACE_OBJ[hemisphere]} {out_surface_prefix}_std.dat > {out_surface_prefix}_std_blur{SURFACE_BLUR}.dat'])
        else:
            process_list.extend([
                f'surfacesignals.bin {param_data} {mid_surface} {out_surface_prefix}.dat',
                f'map_measurements.bin {SURFACE_OBJ[hemisphere]} {mid_surface} {mapping} {out_surface_prefix}.dat > {out_surface_prefix}_std.dat',
                f'blur_measurements.bin -iter {SURFACE_BLUR} {SURFACE_OBJ[hemisphere]} {out_surface_prefix}_std.dat > {out_surface_prefix}_std_blur{SURFACE_BLUR}.dat'])

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
    Furthermore, all values under 0.5 is set to -1. (Perfusion parameter values under 0.5 are very low and therfore very likely to be casued be interpolation with voxels outside FOV)
    
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
    s[s < 0.5] = -1

    s.to_csv(smoothed, index=False)


def clean_perfusion_surface_outside_fov(surface_dir):
    """ Surface mapping are using linear interpolation to extract values from each voxel.
    Thus, in the edge of FOV, the values are not representing true perfusion parametric. 

    To make up for this, this function finds these egde-vertices and set them as outside FOV (value = -1)
    The method is based on region growing such that vertices outside FOV "grow into" the egde until the value on the CBF maps is above 1.

    This new mask is used on all paramteric images for that particular scan.

    The CBF maps is selected because it is fairly stable across all subjects and is normalised to WM.
    A CBF below 1 is hence very low and not likely to be "true" in tissue not affacted by extensive vascular damage. 

    """


    logger.info('Cleaning surface outside FOV')

    for pwi_type in ['PARAMETRIC', 'SEPARAMETRIC']:
        if pwi_type is 'SEPARAMETRIC':
            cbf_thresh = 1
        else:
            cbf_thresh = 0.25
        
        for hemisphere in ['left', 'right']: 
            surf = read_surface(SURFACE_GII[hemisphere])
            faces_all = surf.polys2D
            vert_idx = np.arange(faces_all.max() + 1)

            cbf_files = glob.glob(f'{surface_dir}/*{hemisphere}_{pwi_type}_CBF*blur20.dat')

            for f in cbf_files:
                logger.info(f)

                cbf = pd.read_csv(f)
                sub_prefix = f.split(f'{hemisphere}_{pwi_type}_CBF')[0]

                outside_fov_clean = set()

                # -- Threshold data --
                outside_fov_not_used = set(vert_idx[cbf.values.ravel() == -1])
                below_thresh = set(vert_idx[cbf.values.ravel() < cbf_thresh])

                # --- Only used faces below cbf threshold ---
                faces = faces_all[np.isin(faces_all, list(below_thresh)).any(axis=1)]

                # ----- Find clusters -----
                while outside_fov_not_used:
                    outside = {outside_fov_not_used.pop()}

                    neighbours = set(faces[np.isin(faces, list(outside)).any(axis=1)].ravel()) & below_thresh
                    outside = outside | neighbours

                    while True:
                        neighbours = set(faces[np.isin(faces, list(neighbours)).any(axis=1)].ravel()) & below_thresh
                        below_thresh = below_thresh - neighbours
                        if len(neighbours) == 0:
                            break
                        else:
                            outside = outside | neighbours

                    outside_fov_clean.update(outside)
                    outside_fov_not_used = outside_fov_not_used - outside


                for param in ['CBF', 'CBV', 'MTT', 'CTH', 'RTH', 'PTO2']:
                    if pwi_type is 'PARAMETRIC' and param is 'PTO2':
                        continue

                    param_file = glob.glob(f'{sub_prefix}{hemisphere}_{pwi_type}_{param}*blur20.dat')
                    outfile = f'{param_file[0].split(".dat")[0]}_clean.dat'

                    if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
                        continue
                    
                    df = pd.read_csv(param_file[0])
                    df.iloc[list(outside_fov_clean)] = -1
                    df.to_csv(outfile, index=None)
                
def clean_VSI_outside_FOV(surface_dir):
    """ Remove vertices not is both the SE and GE FOV
    """

    for hemisphere in ['left', 'right']: 
        vsi_files = glob.glob(f'{surface_dir}/*{hemisphere}_VSI*blur20.dat')

        for vsi_file in vsi_files:
            vsi = pd.read_csv(vsi_file)
            sub_prefix = vsi_file.split(f'{hemisphere}_VSI')[0]

            se_cbv_file = glob.glob(f'{sub_prefix}{hemisphere}_SEPWI_CBV*blur20.dat')
            ge_cbv_file = glob.glob(f'{sub_prefix}{hemisphere}_PWI_CBV*blur20.dat')

            se_cbv = pd.read_csv(se_cbv_file[0])
            ge_cbv = pd.read_csv(ge_cbv_file[0])

            outside_fov_clean = list((se_cbv < 0) & (ge_cbv < 0))

            outfile = f'{vsi_file.split(".dat")[0]}_clean.dat'

            if os.path.isfile(outfile) and os.path.getsize(outfile) > 0:
                continue

            vsi.iloc[outside_fov_clean] = -1
            vsi.to_csv(outfile, index=None)
