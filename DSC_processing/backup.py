import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from glob import glob
import nibabel as nib
from scipy.interpolate import interp1d
import sys
sys.path.append('/Users/au483096/Documents/Research/scripts/python_functions/DSC_processing')
from deconv_helperFunctions import mySvd, IntDcmTR
from levenberg_marquardt import LevenbergMarquardtReg

def calc_alpha_beta(mtt, cth):
    # Values are rounded to avoid issues with Floating Point Arithmetic, e.g.  0.1**2 = 0.010000000000000002 if not rounded.
    alpha = round(mtt**2/cth**2,10)
    beta = round(cth**2/mtt,10)

    return alpha, beta

# Calculate MTT and CTH
def calc_mtt_cth(a, b):
    mtt = (a * b)
    cth = (a**0.5) * b
    return mtt, cth

def read_aif_mat(AIFfile):
    # Read AIF mat
    try: # HDF format
        f = h5py.File(AIFfile,'r')
        AIFmat = np.array(f.get('val/AIFmat')).T # For converting to a NumPy array
        AIFs = AIFmat[:,7:]
    except: # If file has been save by scipy.savemat, it needs to read with scipy loadmat. 
        mat = loadmat(AIFfile)
        AIFs = mat['val']['AIFmat'][0][0][:,7:]
        AIFarea = mat['val']['AIFarea'][0][0][0][0]
    
    return AIFs, AIFarea

def read_data():

    conc_files = sorted(glob('/Volumes/projects/MINDLAB2021_MR-APOE-Microcirculation/scratch/dataAug24_perfusion/0004/20211007_082130/MR/SECONC/NATSPACE/*.nii'))
    imgs = []
    for f in conc_files:
        imgs.extend([nib.load(f)])
    n_frames = len(imgs)

    conc_data = np.zeros((imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2], n_frames))
    for i in range(n_frames):
        conc_data[:,:,:,i] = imgs[i].get_fdata()

    conc_data[conc_data == 0] = np.nan
    conc_data = np.transpose(np.flip(conc_data, axis=(0, 1)), (1,0,2,3))

    aif_mat, aifArea = read_aif_mat(glob(f'/Volumes/projects/MINDLAB2021_MR-APOE-Microcirculation/scratch/infoAug24_perfusion/0004/20211007_082130/MR/SEPWI/aifOwn.mat')[0])
    aif = np.mean(aif_mat, axis=0)

    return conc_data, aif, aifArea

conc_data, aif, aif_area = read_data()
baseline_end = 16
TimeBetweenVolumes = 1.56
conc_voxel = conc_data[51,10,0,:] # First voxel in mask slice 0 matlab

svd_cbf, svd_delay, svd_cbv, svd_rf = mySvd(conc_voxel, aif, baseline_end, TimeBetweenVolumes)
cbvbyC = np.trapz(np.clip(conc_voxel[baseline_end:], a_min=0, a_max=None), dx=TimeBetweenVolumes)/aif_area # aif area read from file. Could be calculated like cbvbyC
svd_mtt = cbvbyC/svd_cbf

sampling_factor = 8
dt = 1/sampling_factor*TimeBetweenVolumes

svd_delay = svd_delay/dt

# Upsample signal:
original_length = len(conc_voxel)
upsampled_length = original_length * sampling_factor

time_original = np.arange(original_length) * TimeBetweenVolumes
time_upsampled = np.linspace(0, time_original[-1], upsampled_length)

interp_func_conc = interp1d(time_original, conc_voxel, kind='linear')
conc_voxel_upsampled = interp_func_conc(time_upsampled).ravel()

interp_func_aif = interp1d(time_original, aif, kind='linear')
aif_upsampled = interp_func_aif(time_upsampled).ravel()

time_original = time_original.reshape(-1,1)
time_upsampled = time_upsampled.reshape(-1,1)

alpha = 1
beta = svd_mtt/alpha
theta = np.array([svd_cbf, alpha, svd_delay, beta])


lr = LevenbergMarquardtReg(model_fn = int_dcmTR.fit)

# Fit model
lr.fit(time_original, conc_voxel, theta_init = theta)

int_dcmTR = IntDcmTR(aif)
# int_dcmTR = Int_dcmTR(aif_upsampled)

lr = LevenbergMarquardtReg(model_fn = int_dcmTR.fit)

# Fit model
lr.fit(time_original, conc_voxel, theta_init = theta)
# lr.fit(time_upsampled, conc_mean_upsampled, theta_init = theta)
print(lr.theta)
fitted = lr.predict(time_upsampled)

plt.clf()
plt.plot(time_upsampled, conc_voxel_upsampled, label='conc curve')
plt.plot(time_upsampled, fitted, label='fitted curve')
# plt.plot(t, aif_reference, label='aif')
plt.legend()

class IntDcmTR_orig:
    def __init__(self, U):
        self.p_P = None
        self.p_aif = None
        self.p_rf = None
        self.p_y = None
        self.p_cs = None
        self.p_hits = 0
        self.p_calls = 0
        self.U = U
    
    def reset(self):
        self.p_hits = 0
        self.p_calls = 0
        self.p_P = None
        return [], [], [], 0

    def gamma_eval(self, x, alpha, beta):
        return stats.gamma.cdf(x, alpha, scale=beta)
    
    def int_dcmTR(self, P, M):
        if P is None or M is None or U is None:
            return self.reset()
        
        self.p_calls += 1
        calls = self.p_calls
        
        if U['leakage_correct']:
            k2 = P[4]
        
        P = np.exp(P)
        amp, alpha, delay, beta = P[:4]
        
        delay = min(delay, 1e3)
        
        eq_params = self.p_P is not None and np.array_equal(P, self.p_P)
        self.p_P = P
        
        if not eq_params:
            if self.p_aif is not None and P[2] == self.p_P[2]:
                aif = self.p_aif
                self.p_hits += 1
            else:
                d = delay % 1
                f = int(delay)
                aif = np.concatenate([np.zeros(f), M['pp'](M['time'] - d)])
                self.p_aif = aif
            
            if self.p_rf is not None and P[1] == self.p_P[1] and P[3] == self.p_P[3]:
                rf = self.p_rf
                self.p_hits += 1
            else:
                t = np.arange(0, max(M['time']) * U['dt'], U['dt'])
                if U['shape'] == 'gamma':
                    rf = amp * (1 - self.gamma_eval(t, alpha, beta))
                else:
                    rf = amp * (1 - stats.expon.cdf(t, scale=beta))
                self.p_rf = rf
            
            y = np.convolve(aif, rf) * U['dt']
            y = y[:max(M['s'])]
            self.p_y = y / amp
        else:
            y = self.p_y * amp
            self.p_hits += 3
        
        if U['leakage_correct']:
            if self.p_cs is not None and P[2] == self.p_P[2]:
                cs = self.p_cs
                self.p_hits += 1
            else:
                cs = np.zeros_like(aif)
                idx = max(int(delay), 1)
                cs[idx:] = np.cumsum(aif[idx:]) * U['dt']
                self.p_cs = cs
            
            leakage = k2 * cs[:max(M['s'])]
            y_noleak = y[M['s']] if 'y_noleak' in locals() else []
            y += leakage
        else:
            y_noleak = []
        
        y = y[M['s']]
        rf = rf[M['s']] if 'rf' in locals() else []
        
        hits = self.p_hits
        
        return y, rf, y_noleak, hits, calls


def dcmFit_voxel(cbf, delay, mtt, conc, leakage_correct, param_scale, U, Y, M):
    cbv = np.nan
    Ep = np.nan
    Cp = np.nan
    Ce = np.nan
    F = np.nan
    cbv_noleak = np.nan
    residuef = np.zeros_like(conc)
    hits = 0
    calls = 0
    imacro = []
    sumsq_trivial = 1000
    warnstat = 0
    sumsq = 1000
    rmse = 1000
    
    if not np.any(conc > 0):
        print("Warning: No positive values in concentration curve in this voxel")
        warnstat = 911
        return cbv, Ep, Cp, Ce, F, cbv_noleak, residuef, hits, calls, rmse, imacro, sumsq, warnstat, sumsq_trivial
    
    Y['y'] = conc.reshape(-1, 1)
    delay = min(delay, 5 / U['dt'])
    
    alpha = 1
    beta = mtt / alpha
    M['pE'] = np.log([abs(cbf), alpha, delay, beta])
    
    if leakage_correct:
        M['pE'] = np.append(M['pE'], 0.0)
    
    pC = np.array([0.1, 1, 10, 0.1])
    if leakage_correct:
        pC = np.append(pC, 0.1)
    
    M['pC'] = np.diag(pC) * param_scale
    old_rmse = 1000
    rmse = 0
    sumsq = 1000
    
    for imacro in range(2):
        int_dcmTR_instance = IntDcmTR(U)
        lm_regressor = LevenbergMarquardtReg(model_fn=int_dcmTR_instance.int_dcmTR)
        try:
            # lm_regressor.fit(U, Y['y'], M['pE'])
            lm_regressor.fit(M['aif'], Y['y'], M['pE'])
            Ep = lm_regressor.theta
            if imacro == 0:
                savedEp = Ep.copy()
        except Exception as e:
            print("Warning: Could not fit voxel!", str(e))
            warnstat = 999
            return cbv, Ep, Cp, Ce, F, cbv_noleak, residuef, hits, calls, rmse, imacro, sumsq, warnstat, sumsq_trivial
    
        approx = int_dcmTR_instance.int_dcmTR(Ep, M, U, len(Y['y']))
        sumsq = np.sum((Y['y'] - approx) ** 2) / len(Y['y'])
        rmse = np.sqrt(sumsq) / np.sum(np.abs(Y['y']))
    
        if rmse > 0.01 and not (rmse > old_rmse) and abs(rmse - old_rmse) / old_rmse > 0.10:
            savedEp = Ep.copy()
            M['pE'] = [Ep[0], 0, Ep[2], Ep[1] + Ep[3]]
            old_rmse = rmse
        elif abs(rmse - old_rmse) / old_rmse < 0.10 or rmse > old_rmse:
            rmse = old_rmse
            imacro -= 1
            Ep = savedEp
            break
        else:
            break
    
    imacro -= 1
    ytmp, residuef, y_noleak, hits, calls = int_dcmTR_instance.int_dcmTR(Ep, M, U)
    
    sumsq_trivial = np.sum(Y['y'] ** 2) / len(Y['y'])
    cbv = np.trapz(ytmp)
    cbv_noleak = np.trapz(y_noleak)
    
    return cbv, Ep, Cp, Ce, F, cbv_noleak, residuef, hits, calls, rmse, imacro, sumsq, warnstat, sumsq_trivial


theta = np.array([cbf, alpha, delay, beta])

int_dcmTR = Int_dcmTR(aif)
# int_dcmTR = Int_dcmTR(aif_upsampled)

lr = LevenbergMarquardtReg(model_fn = int_dcmTR.fit)    

# Fit model
lr.fit(time_original, conc_mean, theta_init = theta)
# lr.fit(time_upsampled, conc_mean_upsampled, theta_init = theta)
print(lr.theta)
fitted = lr.predict(time_upsampled)

plt.clf()
plt.plot(time_upsampled, conc_mean_upsampled, label='conc curve')
plt.plot(time_upsampled, fitted, label='fitted curve')
# plt.plot(t, aif_reference, label='aif')
plt.legend()

