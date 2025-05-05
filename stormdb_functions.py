from pathlib import Path
import os
import numpy as np
from tempfile import TemporaryDirectory
from glob import glob
import dicom2nifti
import shutil

if str(Path('__file__').resolve()).startswith('/Volumes'):
    path_prefix = '/Volumes'
else:
    path_prefix = ''

def convert_pwi(filtered_serie: dict, outfile: str, clobber: bool = False):
    if not os.path.isfile(outfile) or clobber:

        with TemporaryDirectory() as tmp_dir:
            # Convert iamge data
            dicom2nifti.convert_directory(path_prefix + filtered_serie['path'], tmp_dir, compression=False)

            # Rename output file
            out_file_temp = glob(tmp_dir + '/*.nii')[0]

            Path(os.path.dirname(outfile)).mkdir(parents=True, exist_ok=True)
            shutil.move(out_file_temp, outfile)
    else:
        print(f'{outfile} exits. Use clobber to overwrite.')

    # INFO
    dcm_file = f'{path_prefix}/{filtered_serie["path"]}/{filtered_serie["files"][0]}'
    info = get_info(dcm_file)

    return info


def get_info(dcm_file):
    import pydicom
    ds = pydicom.dcmread(dcm_file)

    info = {}

    # --- Echo time ---
    echoTime = ds.EchoTime

    # --- Repetition time ---
    repetitionTime = ds.RepetitionTime

    # --- Aquisition times ---
    for element in ds:
        if "MosaicRefAcqTimes" in element.name:
            acqtime = element.value

    acqorder = np.argsort(acqtime)

    uniq_acqtime = list(dict.fromkeys(acqtime))

    _, uniq_acqorder = np.unique(uniq_acqtime, return_index=True)

    for i, time in enumerate(uniq_acqtime):
        acqorder[np.array(acqtime) == time] = uniq_acqorder[i]
    
    # --- Return ---
    info['TE'] = echoTime * 1e-3 # In seconds
    info['TR'] = repetitionTime * 1e-3 # In seconds
    info['acqorder'] = acqorder
    
    return info