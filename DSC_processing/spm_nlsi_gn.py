import numpy as np

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
    h = 1
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
        g = model_fn(t, pE)
        r = y - g

        Jp = np.zeros((len(g), len(ip)))

        for i in range(ip):
            dV = dv*np.sqrt(Cp[i,i])
            dg = model_fn(t, pE + B[:,i]*dV) - g
            
            Jp[:,i] = dg/dV

        J = Jp

        # ------------
        S = h[0]
        iS = np.inv(S)
        Cp = np.inv(J.T @ iS @ J+ ipC)

        A=np.trace((J.T @ J) @ Cp) + r.T @ r

        h[0] = A/ne

        F = - r.T @ iS @ r/2 - (pE-Ep).T @ ipC @ (pE-Ep)/2 - ne @ np.log(8*np.atan(1))/2 + ne @ spm_logdet(iS)/2 + spm_logdet(ipC)/2 - spm_logdet(Cp)/2

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

            dFdp  = J.T @ iS @ r + ipC @ (pE-Ep)
            dFdpp = J.T @ iS @ J + ipC

            # decrease regularization
            # ---------------------------------------------------------------
            lm = lm/2
            # str_ = 'E-Step(-)'

        else:
            # reset expansion point
            # ---------------------------------------------------------------
            p = C_p
            h = C_h

            lm = np.max(lm*4,1/512)
        
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
    H = np.sum(np.log(np.diag(C)))

    # invoke det if non-diagonal
    # ---------------------------------------------------------------------------
    i,j = np.find(C)
    n = len(C)
    if any(i != j):
        a = np.exp(H/n)
        H = H + np.log(np.linalg.det(C/a))

    # invoke svd is rank deficient
    #---------------------------------------------------------------------------
    if np.any(np.imag(H)) or np.any(np.isinf(H)):  
        s = np.linalg.svd(np.asarray(C), compute_uv=False)  # Compute SVD singular values
        H = np.sum(np.log(s[s > len(s) * np.max(s) * np.finfo(s.dtype).eps]))
    
    return H