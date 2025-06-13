import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.integrate import quad

def calc_oef(mtt, cth, k=68, pto2=21.8, plot_h=False):
    # Function to calculate OEF from method described in Jespersen et al, 2021.
    # Using model paramters from Madsen et al, 2025. 

    # --- Constants for Q(t) ---
    aH = 3.1e-5
    sat = 0.95
    B = 0.1943
    C0 = sat * B
    P50 = 26
    h_val = 2.8
    Ct = pto2 * aH

    # --- Q(t) function ---
    def Q(t):
        def dC_dx(C, x):
            return -k * t * (aH * P50 * (C / (B - C))**(1 / h_val) - Ct)

        x = [0, 1]  # Only need endpoints
        # C = odeint(dC_dx, C0, x, args=(...), atol=1e-10, rtol=1e-10)
        C = odeint(dC_dx, C0, x)
        return 1 - C[-1, 0] / C[0, 0]
    
    # --- Integrand for OEF ---
    def integrand(t):
        return gamma_dist.pdf(t) * Q(t)

    if cth == 0:
        OEF = Q(mtt)
    else:
        # --- Calculate alpha and beta ---
        alpha = (mtt / cth) ** 2
        beta = (cth ** 2) / mtt

        # Precompute gamma distribution object for reuse
        gamma_dist = gamma(a=alpha, scale=beta)

        # --- Find integration upper bound using gamma.ppf (faster than cumulative quad) ---
        max_t = gamma_dist.ppf(0.9999)

        # --- Integrate to get OEF ---
        OEF, error = quad(integrand, 0, max_t)


    return OEF
