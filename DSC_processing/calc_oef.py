import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.integrate import quad

def calc_oef(k, pto2, alpha=5, beta=5, plot_h=False):
    # Function to calculate OEF from method described in Jespersen et al, 2021.
    
    def h(t):
        # Probability density function of transit time distribution
        return gamma.pdf(t, alpha, scale=beta)

    
    def Q(t):
        # Calculation of Q (OEF) for capillaries with transit time t

        # Define the differential equation as a function (Q)
        def dC_dx(C, x, k, t, aH, P50, B, h, Ct):
            return -k * t * (aH * P50 * (C / (B - C))**(1/h) - Ct)
        
        # Define parameters and initial conditions in dC_dx calcuations (from Jespersen et al, 2012)
        aH = 3.1e-5
        sat = 0.95
        B = 0.1943
        C0 = sat*B
        P50 = 26
        h = 2.8
        Ct = pto2*aH

        # Sampling of points along the vessel (fractional distance). We only need the first and last point in the vessel to caluclate OEF. 
        x = np.linspace(0, 1, 2)

        # Calculate C(x) (i.e. oxygen concentration along the capillary length)
        C = odeint(dC_dx, C0, x, args=(k, t, aH, P50, B, h, Ct))

        # Calculate Q (OEF) using the difference in oxygen concentration between the start end end of the vessel.  
        Q = 1-C[-1]/C[0]

        return Q[0]

    def integrand(t):
        # The integration term in eq. 1 (Jespersen et al, 2021). OEF=int_0_inf {dt h(t)*Q(t)}
        return h(t)*Q(t)

    # Make sure h(t) is small and the end of integration 
    # Ideally, the integration should go to infinity. This is ofcause infeasible, but this step makes sure that we have 99.99 % of all h(t) included. 
    max_t = 0
    for i in np.arange(10,1000, 10):
        P_X_x, _ = quad(h, 0, i) # P(X>x)
        if P_X_x > 0.9999:
            max_t = i
            break

    # Calculate OEF from eq. 1 (Jespersen et al, 2021). OEF=int_0_inf {dt h(t)*Q(t)}
    OEF, error = quad(integrand, 0, max_t)

    if plot_h:
        # Plot h(t)
        t = np.linspace(0, max_t+10, 100)
        # Plot h(t)
        plt.figure(figsize=(10,6))
        plt.plot(t, h(t))
        plt.axvline(x = max_t, color = 'r', linestyle = '-', label = 'End of integration') 
        plt.xlabel(r'$\tau$', fontsize=20)
        plt.ylabel(rf'h($\tau$)', fontsize=20)
        h_text = rf'h($\tau$;$\alpha=${alpha}, $\beta=${beta})'
        plt.title(f'Transit time distribution {h_text}\nMTT={alpha*beta:.2f}, CTH={np.sqrt(alpha)*beta:.2f}', fontsize=20)
        plt.legend(loc='upper right')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)

    return OEF
