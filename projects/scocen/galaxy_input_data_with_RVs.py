"""
Plot ScoCen in (l, b) coordinates, stars from the input data with RVs.
"""

import numpy as np
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib
from matplotlib import colors as cmpl
plt.ion()
import copy

from chronostar.component import SphereComponent as Component

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename_fit = lib.data_filename_fit
data_filename = lib.data_filename
good_comps = lib.good_comps
############################################

# Read data
try:
    tabf = tabf0
    tab = tab0
except:
    tabf0 = Table.read(data_filename_fit)
    tab0 = Table.read(data_filename)
    
    
    tabf = tabf0
    tab = tab0

# Shift data in galactic 'l' to put ScoCen in the middle of the plot
print('Shifting l by 100 to put ScoCen in the middle')
lshift=100
mask = np.where(tabf['l']<lshift)
tabf['l'][mask] = 360 + tabf['l'][mask]
mask = np.where(tab['l']<lshift)
tab['l'][mask] = 360 + tab['l'][mask]


def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    ax.set_ylim(-40, 40)
    #~ ax.set_xlim(380, 260)
    ax.set_xlim(380, 240)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    ax.set_aspect('equal')

    return ax



#### PLOTTING ###############################
# Data used in the fit
fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax=fig.add_subplot(111)
ax.scatter(tabf['l'], tabf['b'], s=1, c='k', marker='.', label='')
print('Total number of stars in this plot:', len(tabf))
gx_set_labels_and_ticks_over_360deg(ax)


# New data with RVs
fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax=fig.add_subplot(111)
mask = tab['radial_velocity_error']<100
tab=tab[mask]
ax.scatter(tab['l'], tab['b'], s=1, c='k', marker='.', label='')
print('Total number of stars in this plot:', len(tab))
gx_set_labels_and_ticks_over_360deg(ax)

# New data with RVs, non-bg stars
fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax=fig.add_subplot(111)
mask = tab['radial_velocity_error']>1e+10
for c in good_comps:
    mask = np.logical_or(mask, tab['membership%s'%c]>0.1)
ax.scatter(tab['l'][mask], tab['b'][mask], s=1, c='k', marker='.', label='')
print('Total number of stars in this plot:', len(tab[mask]))
gx_set_labels_and_ticks_over_360deg(ax)


# SAVE FIGURES
#~ plt.savefig('gx_input_params.pdf')
plt.show()
