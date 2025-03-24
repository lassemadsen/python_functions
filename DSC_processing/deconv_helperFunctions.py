import numpy as np
from scipy.special import gamma
from scipy.signal import convolve

class IntDcmTR:
    def __init__(self, aif):
        self.aif = aif
        self.cbf = None
        self.delay = None
        self.alpha = None
        self.beta = None

    def h(self, t, alpha, beta):
        return 1 / (beta**alpha * gamma(alpha)) * t**(alpha-1) * np.exp(-t/beta)

    def fit(self, t, conc, theta):

        # Check equal parameters 
        if self.cbf and self.delay and self.alpha and self.beta:
            self.cbf = theta[0]
            self.alpha = theta[1]
            self.beta = theta[3]

        if theta[2] != self.delay:
            self.delay = theta[2]
            # Fit delay
            d = self.delay % 1
            f = int(self.delay)
            aif = np.concatenate([np.zeros(f), M['pp'](M['time'] - d)]) # TODO fix
            self.p_aif = aif
        if theta[1] != self.alpha or theta[3] != self.beta:
            rf = self.cbf * self.h(t, self.alpha, self.beta)

        y = convolve(rf, conc)

        y = y[:len(conc)]

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