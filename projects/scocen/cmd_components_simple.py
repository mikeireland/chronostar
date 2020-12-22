"""
Author: Marusa Zerjal, 2020 - 01 - 20
"""

import numpy as np
from astropy.table import Table, join, unique
from astropy import units as u
from astropy.coordinates import SkyCoord # This is slow but used only for components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
plt.ion()

#~ import scocenlib2 as lib
#~ from fig_settings import *

import sys
sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
import coordinate

# Minimal probability required for membership
pmin_membership = 0.5
grid = [5, 5] # how to split subplots
################################

# Data
#~ d=lib.d
d = Table.read('members_from_5k_fit_and_overlaps.fits')

Gmag = d['phot_g_mean_mag'] - 5 * np.log10(1.0 / (d['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
d['Gmag'] = Gmag

# Read components (WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present!)
#~ comps = lib.comps
comps = Table.read('final_comps.fits')
comps.rename_column('age', 'Age')

# Crossing time. Only components with sigma<age have reliable ages.
crossing_time = comps['dX']/comps['dV'] * 0.977813106 # pc/km*s to Myr
comps['Crossing_time'] = crossing_time
mask = crossing_time < comps['Age'] # sigma < age
comps['Age_reliable'] = mask

"""
Plot CMD for members of each component
"""

# Minimal probability required for membership
pmin=0.5

xlim = [-1, 5]
ylim = [17, -3]

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
    age=c['Age']
    comp_ID = c['Name']
    col=d['membership%s'%comp_ID]
    mask=col>pmin
    t=d[mask]

    if len(t)>100:
        alpha=0.5
    else:
        alpha=1

    t.sort('membership%s'%comp_ID)
    t.reverse()        
    #~ ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=alpha)
    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c=t['membership%s'%comp_ID], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)

    ax=plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim)

    #~ ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
    #~ ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    #~ ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

    #~ ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
    #~ ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
    #~ ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
    #~ ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

    #~ ax.set_xlim(xlim[0], xlim[1])
    #~ ax.set_ylim(ylim[0], ylim[1])

    # Sum probability
    sump=np.nansum(d['membership%s'%comp_ID])
    
    ax.set_title('%s (%.2f$\pm$%.2f Myr %s) %d (%d)'%(comp_ID, age, c['Crossing_time'], c['Age_reliable'], len(t), sump))



#~ plt.tight_layout()
plt.show()
