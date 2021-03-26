"""
Plot CMDs for each component.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
compnames = lib.compnames
colors = lib.colors
############################################

# Minimal probability required for membership
pmin_membership = 0.5
############################################
# how to split subplots
grid = [5, 5] 

# CMD limits
xlim = [-1, 5]
ylim = [17, -3]
############################################

# Read data
try:
    tab = tab0
    comps = comps0
except:
    tab0 = Table.read(data_filename)
    
    Gmag = tab0['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab0['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
    tab0['Gmag'] = Gmag
    
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0


# Main sequence parametrization
# fitpar for pmag, rpmag
fitpar = [0.17954163, -2.48748376, 12.9279348, -31.35434182, 38.31330583, -12.25864507]
poly = np.poly1d(fitpar)
x = np.linspace(1, 4, 100)
y = poly(x)
m = y > 4
yms = y[m]
xms = x[m]

def plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim):
    ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
    ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

    ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
    ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
    ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
    ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
        
    return ax

print('Plotting %d components.'%len(comps))

fig=plt.figure()
for i, c in enumerate(comps):
    ax = fig.add_subplot(grid[0], grid[1], i+1) # TODO: adjust this if needed
    
    comp_ID = c['comp_ID']
    col=tab['membership%s'%comp_ID]
    mask = col > pmin_membership
    t=tab[mask]

    if len(t)>100:
        alpha=0.5
    else:
        alpha=1

    t.sort('membership%s'%comp_ID)
    t.reverse()        
    #~ ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=alpha)
    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c=t['membership%s'%comp_ID], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)

    ax=plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim)
    
    age=c['Age']
    ax.set_title('%s (%.2f$\pm$%.2f Myr %s) %d'%(comp_ID, age, c['Crossing_time'], c['Age_reliable'], len(t)))



#~ plt.tight_layout()
plt.show()
