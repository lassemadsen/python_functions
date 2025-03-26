import numpy as np
from scipy.stats import gamma
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.sparse import find
from numpy.linalg import det

class IntDcmTR:
    def __init__(self, aif, sampling_factor, TimeBetweenVolumes):
        self.aif = aif
        self.aif_upsampled_fn = interp1d(np.arange(len(self.aif)) * TimeBetweenVolumes, aif, kind='cubic', fill_value="extrapolate", bounds_error=False)
        self.aif_upsampled = None
        self.sampling_factor = sampling_factor
        self.TimeBetweenVolumes = TimeBetweenVolumes
        self.cbf = None
        self.delay = None
        self.alpha = None
        self.beta = None
        self.rf = None
        self.dt = 1 / self.sampling_factor * self.TimeBetweenVolumes 

    def fit(self, t, p):
        """
        t : Time
        p : parameters of the model
            p[0]: CBF (amplitude)
            p[1]: Alpha
            P[2]: Delay 
            P[3]: Beta
        """
        p = np.exp(p.ravel())
        cbf = p[0]
        alpha = p[1]
        delay = round(p[2],6) # Round,6 to prevent small rounding errors when converting between log and exp
        beta = p[3] 

        t_upsampled = np.arange(len(t)*self.sampling_factor) * self.dt #np.linspace(0, t[-1], len(t)*self.sampling_factor)

        if alpha != self.alpha or beta != self.beta or cbf != self.cbf:
            self.cbf = cbf
            self.alpha = alpha
            self.beta = beta
            self.rf = self.cbf * (1 - gamma.cdf(t_upsampled, self.alpha, scale=self.beta)).ravel()

        if delay != self.delay:
            # Fit delay
            self.delay = round(delay,5)
            delay_samples = int(np.ceil(delay/self.dt))-1
            self.aif_upsampled = self.aif_upsampled_fn(t_upsampled-self.delay).ravel() # np.concatenate([np.zeros(f).reshape(-1,1), self.aif(t-d)]).ravel()
            self.aif_upsampled[:delay_samples] = 0

        y = convolve(self.rf, self.aif_upsampled)*self.dt # why multiply by dt? 

        # Subsample
        y = y[0::self.sampling_factor][:len(t)]

        return y
    
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

def spm_nlso_gn_no_graphic(model_fn, pE, pC, y, t):
    pass

    Cp = pC
    Ep = pE

    ipC = np.linalg.inv(pC)

    C_F = -np.inf
    dv = 1/128
    lm = 0
    C_p = pE
    C_h = 0

    # Basis vecs
    ip = len(pE)
    np_ = ip
    B = np.eye(ip)
    h = 1 #np.array([1]).reshape(-1,1)
    ne = len(y)
    Q = np.eye(ne)
    dFdpp = np.eye(ip)
    dFdp = np.zeros(ip)
    p = np.zeros(ip)

    oldF = -np.inf

    for k in range(32):
    # Estep
    
    # M-Step: ReML estimator of variance components:  h = max{F(p,h)}
    # ===================================================================
    
    # prediction and errors
    # -------------------------------------------------------------------
        g = model_fn(t, Ep)
        r = y - g

        Jp = np.zeros((len(g), ip))

        for i in range(ip):
            dV = dv*np.sqrt(Cp[i,i])
            dg = model_fn(t, Ep + B[:,i]*dV) - g
            
            Jp[:,i] = dg/dV

        J = Jp

        # ------------
        S = h
        # iS = np.linalg.inv(S)
        if S == 0:
            iS = np.inf
        else:
            iS = 1/S
        Cp = np.linalg.inv(J.T @ (iS * J)+ ipC)

        A = np.trace((J.T @ J) @ Cp) + r.T @ r

        h = A/ne

        F = - r.T @ (iS * r)/2 - (pE-Ep).T @ ipC @ (pE-Ep)/2 - ne * np.log(8*np.arctan(1))/2 + ne * spm_logdet(iS)/2 + spm_logdet(ipC)/2 - spm_logdet(Cp)/2

        # if F has increased, update gradients and curvatures for E-Step
        # -------------------------------------------------------------------
        if F > C_F:
        
            # accept current estimates
            # ---------------------------------------------------------------
            C_p   = p
            C_h   = h
            C_F   = F

            # E-Step: Conditional update of gradients and curvature
            # ---------------------------------------------------------------

            dFdp  = J.T @ (iS * r) + ipC @ (pE-Ep)
            dFdpp = J.T @ (iS * J) + ipC

            # decrease regularization
            # ---------------------------------------------------------------
            lm = lm / 2
            # str_ = 'E-Step(-)'

        else:
            # reset expansion point
            # ---------------------------------------------------------------
            p = C_p
            h = C_h

            lm = np.max([lm*4,1/512])
        
        # E-Step: update
        # ==================================================================
        l = lm * np.linalg.norm(np.asarray(dFdpp)) * np.eye(np_) 

        # ----
        # dp    = (dFdpp + diag(diag(l)))\dFdp;
        # Extract diagonal of l and form a diagonal matrix
        l_diag = np.diag(np.diag(l.toarray())) if hasattr(l, "toarray") else np.diag(np.diag(l))

        # Solve the linear system
        dp = np.linalg.solve(dFdpp + l_diag, dFdp)
        # ---

        p = p + dp
        Ep = pE + p

        nmp = np.sqrt(dp.T @ dp)

        # convergence
        # -------------------------------------------------------------------
        if nmp < 1e-4 and k > 20:
            # mbh: It is probably not getting any further...    
            break
        
        oldF = F

    return Ep, Cp, S, F


def spm_logdet(C):
    """returns the log of the determinant of the positive definite matrix C
    FORMAT [H] = spm_logdet(C)
    H = log(det(C))

    spm_logdet is a computationally efficient operator that can deal with
    sparse matrices
    ___________________________________________________________________________
    Karl Friston

    assume diagonal form
    ---------------------------------------------------------------------------
    """
    if isinstance(C, (int, float)):
        C = np.reshape(C, (-1,1))

    H = np.sum(np.log(np.diag(C)))

    # invoke det if non-diagonal
    # ---------------------------------------------------------------------------

    # Get the nonzero indices of C
    i, j, _ = find(C)

    # Get the number of rows (or length equivalent in MATLAB)
    n = len(C)

    # Check if C is non-diagonal
    if np.any(i != j):
        a = np.exp(H / n)
        H = H + np.log(det(C / a))

    # invoke svd is rank deficient
    #---------------------------------------------------------------------------
    if np.any(np.imag(H)) or np.any(np.isinf(H)):  
        s = np.linalg.svd(np.asarray(C), compute_uv=False)  # Compute SVD singular values
        H = np.sum(np.log(s[s > len(s) * np.max(s) * np.finfo(s.dtype).eps]))
    
    return H