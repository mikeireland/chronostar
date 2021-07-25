"""
Print data for table `componentsfit.tex` in the paper
"""

import numpy as np
from astropy.table import Table

from chronostar.component import SphereComponent

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
comps_filename = lib.comps_filename
bg_comps = lib.bg_comps
comps_multiple_pop = lib.comps_multiple_pop
data_filename_fit = lib.data_filename_fit
############################################
comps = Table.read(comps_filename)

# Data used for the fit
data_fit = Table.read(data_filename_fit)

# Create components
comps_raw = SphereComponent.load_raw_components(comps_filename)
print('Number of components: %d'%len(comps_raw))

# Labels
print('ID & $X_\mathrm{t}$ & $Y_\mathrm{t}$ & $Z_\mathrm{t}$ & $U_\mathrm{t}$ & $V_\mathrm{t}$ & $W_\mathrm{t}$ & $\sigma_{X_\mathrm{t}}$ & $\sigma_{Y_\mathrm{t}}$ & $\sigma_{Z_\mathrm{t}}$ & $\sigma_{U_\mathrm{t}}$ & $\sigma_{V_\mathrm{t}}$ & $\sigma_{W_\mathrm{t}}$ & $X_0$ & $Y_0$ & $Z_0$ & $U_0$ & $V_0$ & $W_0$ & $\sigma_{X_0}$ & $\sigma_{V_0}$ & Age & $\sigma_{\mathrm{Age}}$ & N$_\mathrm{fit}$ \\\\')

# Units
print(' & pc & pc & pc & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & pc & pc & pc & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & pc & pc & pc & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & $\mathrm{km\,s^{-1}}$ & pc & $\mathrm{km\,s^{-1}}$ & Myr & Myr & \\\\')


for c, c_raw in zip(comps, comps_raw):
    comp_id = c['comp_ID']
    age=c['Age']
    sigma_age = c['Crossing_time']
    
    if comp_id=='G':
        print(age, sigma_age)
    
    
    mean_now = c_raw.get_mean_now()
    covmatrix_now = c_raw.get_covmatrix_now()

    # Component at time NOW (today in the sky)
    Xt = mean_now[0]
    Yt = mean_now[1]
    Zt = mean_now[2]
    Ut = mean_now[3]
    Vt = mean_now[4]
    Wt = mean_now[5]

    sigmaXt = np.sqrt(covmatrix_now[0,0])
    sigmaYt = np.sqrt(covmatrix_now[1,1])
    sigmaZt = np.sqrt(covmatrix_now[2,2])
    sigmaUt = np.sqrt(covmatrix_now[3,3])
    sigmaVt = np.sqrt(covmatrix_now[4,4])
    sigmaWt = np.sqrt(covmatrix_now[5,5])


    # Component at time 0 (at birth)
    X0 = c['X']
    Y0 = c['Y']
    Z0 = c['Z']
    U0 = c['U']
    V0 = c['V']
    W0 = c['W']
    sigmaX0 = c['dX']
    sigmaV0 = c['dV']
    
    
    comment=''
    if comp_id in bg_comps:
        #~ comment = 'Background'
        comment = 'bg'
    
    if comp_id in comps_multiple_pop:
        #~ comment = 'MS+PMS'
        comment = 'bg*'
    

    # Number of members used in the fit
    pmin_membership = 0.5
    mask = data_fit['membership%s'%comp_id] >= pmin_membership
    Nfit = np.sum(mask)


    #~ print('%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f & %.2f & %.2f & %.2f & %.1f & %.2f & %.1f & %.1f & %d & %s & %s \\\\'%(comp_id, Xt, Yt, Zt, Ut, Vt, Wt, sigmaXt, sigmaYt, sigmaZt, sigmaUt, sigmaVt, sigmaWt, X0, Y0, Z0, U0, V0, W0, sigmaX0, sigmaV0, age, sigma_age, Nfit, comment, comp_id))
    
    # Less decimal places
    print('%s & %.0f & %.0f & %.0f & %.1f & %.1f & %.1f & %.0f & %.0f & %.0f & %.1f & %.1f & %.1f & %.0f & %.0f & %.0f & %.1f & %.1f & %.1f & %.0f & %.1f & %.0f & %.0f & %d \\\\'%(comp_id, Xt, Yt, Zt, Ut, Vt, Wt, sigmaXt, sigmaYt, sigmaZt, sigmaUt, sigmaVt, sigmaWt, X0, Y0, Z0, U0, V0, W0, sigmaX0, sigmaV0, age, sigma_age, Nfit))
    
