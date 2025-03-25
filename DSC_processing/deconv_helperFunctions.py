import numpy as np
from scipy.stats import gamma
from scipy.signal import convolve
from scipy.interpolate import interp1d

class IntDcmTR:
    def __init__(self, aif, sampling_factor, TimeBetweenVolumes):
        self.aif = aif
        self.aif_upsampled = interp1d(np.arange(len(self.aif)) * TimeBetweenVolumes, aif, kind='linear', fill_value=(0, None), bounds_error=False)
        self.sampling_factor = sampling_factor
        self.TimeBetweenVolumes = TimeBetweenVolumes
        self.cbf = None
        self.delay = None
        self.alpha = None
        self.beta = None
        self.rf = None
        self.dt = 1 / self.sampling_factor * self.TimeBetweenVolumes 
    
    def fit_new(self, t, theta):
        # t: original
        cbf = theta[0,0]
        alpha = theta[1,0]
        delay = theta[2,0]
        beta = theta[3,0]

        t_upsampled = np.arange(len(t)*self.sampling_factor) * self.dt #np.linspace(0, t[-1], len(t)*self.sampling_factor)

        if alpha != self.alpha or beta != self.beta or cbf != self.cbf:
            self.cbf = cbf
            self.alpha = alpha
            self.beta = beta
            self.rf = self.cbf * (1 - gamma.cdf(t_upsampled, self.alpha, scale=self.beta))
            # self.rf = self.cbf * self.h(t, self.alpha, self.beta)

        if delay != self.delay:
            # Fit delay
            self.delay = delay

            aif = self.aif(t-self.delay) # np.concatenate([np.zeros(f).reshape(-1,1), self.aif(t-d)]).ravel()

            # d = int(delay % 1)
            # if f != 0:
        else:
            aif = self.aif(t)

        y = convolve(aif, self.rf)

        y = y[0::self.sampling_factor][:len(t_upsampled)]

        return y

    def fit(self, t, theta):
        theta = theta
        cbf = theta[0]
        alpha = theta[1]
        delay = theta[2]
        beta = theta[3]

        t_upsampled = np.arange(len(t)*self.sampling_factor) * self.dt #np.linspace(0, t[-1], len(t)*self.sampling_factor)

        if alpha != self.alpha or beta != self.beta or cbf != self.cbf:
            self.cbf = cbf
            self.alpha = alpha
            self.beta = beta
            self.rf = self.cbf * (1 - gamma.cdf(t_upsampled, self.alpha, scale=self.beta)).ravel()

        if delay != self.delay:
            # Fit delay
            self.delay = delay

            aif = self.aif_upsampled(t_upsampled-self.delay).ravel() # np.concatenate([np.zeros(f).reshape(-1,1), self.aif(t-d)]).ravel()

            # d = int(delay % 1)
            # if f != 0:
        else:
            aif = self.aif_upsampled(t_upsampled).ravel()

        y = convolve(self.rf, aif)*self.dt # why multiply by dt? 

        # Subsample
        y = y[0::self.sampling_factor][:len(t)]

        return y
    
    # def int_dcmTR(self, P, M):
    #     if P is None or M is None or U is None:
    #         return self.reset()
        
    #     self.p_calls += 1
    #     calls = self.p_calls
        
    #     if U['leakage_correct']:
    #         k2 = P[4]
        
    #     P = np.exp(P)
    #     amp, alpha, delay, beta = P[:4]
        
    #     delay = min(delay, 1e3)
        
    #     eq_params = self.p_P is not None and np.array_equal(P, self.p_P)
    #     self.p_P = P
        
    #     if not eq_params:
    #         if self.p_aif is not None and P[2] == self.p_P[2]:
    #             aif = self.p_aif
    #             self.p_hits += 1
    #         else:
    #             d = delay % 1
    #             f = int(delay)
    #             aif = np.concatenate([np.zeros(f), M['pp'](M['time'] - d)])
    #             self.p_aif = aif
            
    #         if self.p_rf is not None and P[1] == self.p_P[1] and P[3] == self.p_P[3]:
    #             rf = self.p_rf
    #             self.p_hits += 1
    #         else:
    #             t = np.arange(0, max(M['time']) * U['dt'], U['dt'])
    #             if U['shape'] == 'gamma':
    #                 rf = amp * (1 - self.gamma_eval(t, alpha, beta))
    #             else:
    #                 rf = amp * (1 - stats.expon.cdf(t, scale=beta))
    #             self.p_rf = rf
            
    #         y = np.convolve(aif, rf) * U['dt']
    #         y = y[:max(M['s'])]
    #         self.p_y = y / amp
    #     else:
    #         y = self.p_y * amp
    #         self.p_hits += 3
        
    #     if U['leakage_correct']:
    #         if self.p_cs is not None and P[2] == self.p_P[2]:
    #             cs = self.p_cs
    #             self.p_hits += 1
    #         else:
    #             cs = np.zeros_like(aif)
    #             idx = max(int(delay), 1)
    #             cs[idx:] = np.cumsum(aif[idx:]) * U['dt']
    #             self.p_cs = cs
            
    #         leakage = k2 * cs[:max(M['s'])]
    #         y_noleak = y[M['s']] if 'y_noleak' in locals() else []
    #         y += leakage
    #     else:
    #         y_noleak = []
        
    #     y = y[M['s']]
    #     rf = rf[M['s']] if 'rf' in locals() else []
        
    #     hits = self.p_hits
        
    #     return y, rf, y_noleak, hits, calls


def mySvd(conc_data, aif, baseline_end, TimeBetweenVolumes, threshold=0.2):
    """
    Calculate the residual function, CBF, CBV and delay using SVD
    """
    AIF_matrix = aifm_sim(aif[baseline_end:], TimeBetweenVolumes, 0)

    U, S, Vh = np.linalg.svd(AIF_matrix)

    # Invert and mask S matrix
    S_matrix = np.zeros_like(AIF_matrix)
    S[S<S[0]*threshold] = 0
    np.fill_diagonal(S_matrix, S)

    S_inv = np.where(S_matrix != 0, 1 / S_matrix, 0)

    # SVD matrix, truncated and inversed
    invAIF = Vh.T @ S_inv @ U.T

    # Calculate the residue function 
    rf = invAIF @ conc_data[baseline_end:].reshape(-1,1).ravel()

    # Calculate parameters CBF, CBV and delay
    cbf, max_sample = np.max(rf), np.argmax(rf)
    cbv = np.trapz(rf, dx=TimeBetweenVolumes) # To get exaclty the same as in the MATLAB implementation 
    delay = max_sample * TimeBetweenVolumes

    return cbf, delay, cbv, rf

def aifm_sim(aif, delt, dosmooth):
    """
    Constructs the AIF matrix for deconvolution.
    
    Parameters:
        aif (array-like): Arterial input function vector.
        delt (float): Time step.
        dosmooth (bool): If True, applies smoothing.

    Returns:
        np.ndarray: The AIF matrix.
    """
    aif = np.reshape(aif, (1, len(aif)))  # Ensure row vector
    
    if dosmooth:
        AIF = np.pad(aif, (1, 1), mode='constant', constant_values=0)  # Zero padding
        lengthA = len(AIF)
        A = np.zeros((lengthA - 2, lengthA - 2))
        
        for row in range(lengthA - 2):
            for column in range(row + 1):
                intoarraypos = row - column + 1  # Adjusted for zero padding
                A[row, column] = delt * (AIF[intoarraypos - 1] + 4 * AIF[intoarraypos] + AIF[intoarraypos + 1]) / 6
    else:
        lengthA = len(aif[0])  # Since aif is reshaped as (1, N)
        A = np.zeros((lengthA, lengthA))
        
        for row in range(lengthA):
            for column in range(row + 1):
                intoarraypos = row - column  # No zero padding needed
                A[row, column] = delt * aif[0, intoarraypos]
    
    return A