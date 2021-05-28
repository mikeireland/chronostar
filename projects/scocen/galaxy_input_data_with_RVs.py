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

def plot_3_windows_gx(ax, labels=True, lw=2, ls='-', c='r', fontsize=12):
    """
    Plot lines designating USco, UCL, LCC
    """
    print('WINDOWS plotting...')
    
    def plot_window(ax, x1=None, x2=None, y1=None, y2=None, c=None, ls=None, lw=None):
        ax.plot([x1, x1], [y1, y2], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x2, x2], [y1, y2], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x1, x2], [y1, y1], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x1, x2], [y2, y2], c=c, linestyle=ls, linewidth=lw)
    
    # USco
    plot_window(ax, x1=342, x2=360, y1=10, y2=30, c=c, ls=ls, lw=lw)
    if labels:
        ax.annotate('USCO',
                xy=(355, 32), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=fontsize)
    
    # UCL
    ax.plot([350, 350], [0, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 312], [0, 25], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 350], [0, 0], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 312], [25, 25], c=c, linestyle=ls, linewidth=lw)
    if labels:
        ax.annotate('UCL',
                xy=(335, 27), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=fontsize)
    
    # LCC
    plot_window(ax, x1=312, x2=285, y1=-10, y2=23, c=c, ls=ls, lw=lw)
    if labels:
        ax.annotate('LCC',
                xy=(301, 25), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=fontsize)

    # Lupus complex (Hara et al. 1999)
    # Plot with windows from https://www.eso.org/~fcomeron/lupus_rev.pdf
    #~ ls = '--'
    #~ # Lupus 1
    #~ plot_window(ax, x1=337, x2=340, y1=14, y2=18, c=c, ls=ls, lw=lw)

    #~ # Lupus 2
    #~ plot_window(ax, x1=338, x2=340, y1=11, y2=13, c=c, ls=ls, lw=lw)

    #~ # Lupus 3
    #~ plot_window(ax, x1=338, x2=341, y1=8, y2=10, c=c, ls=ls, lw=lw)

    #~ # Lupus 4
    #~ plot_window(ax, x1=338, x2=335, y1=6, y2=8, c=c, ls=ls, lw=lw)

    #~ # Lupus 5
    #~ plot_window(ax, x1=344, x2=341, y1=7, y2=10, c=c, ls=ls, lw=lw)

    #~ # Lupus 6
    #~ plot_window(ax, x1=340, x2=342, y1=7, y2=5, c=c, ls=ls, lw=lw)

    #~ # Lupus 7
    #~ plot_window(ax, x1=336.5, x2=337.5, y1=4.5, y2=5.5, c=c, ls=ls, lw=lw)

    #~ # Lupus 8
    #~ plot_window(ax, x1=336, x2=334, y1=4.5, y2=3, c=c, ls=ls, lw=lw)

    #~ # Lupus 9
    #~ plot_window(ax, x1=346.5, x2=347.5, y1=6, y2=7, c=c, ls=ls, lw=lw)


    #~ # Damiani et al. 2019. UCL-1 is V1092 Sco
    #~ # UCL-2
    #~ plot_window(ax, x1=316, x2=322, y1=11, y2=15, c=c, ls=ls, lw=lw)
    
    #~ # LCC-1
    #~ plot_window(ax, x1=297, x2=303, y1=-4, y2=0, c=c, ls=ls, lw=lw)


    #~ # IC2602
    #~ ax.scatter([289.6014], [-04.9061], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('IC2602',
                #~ xy=(289.6014, -04.9061), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # V1062 Sco moving group (a newly discovered MG in ScoCen by Roser et al. 2018)
    #~ # (X, Y, Z, U, V, W) = (167.20, -49.14, 13.44, -3.80, -19.96, -4.06). from Roser et al. 2018
    #~ ax.scatter([343.6], [4.3], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('V1062 Sco',
                #~ xy=(343.6, 4.3), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # Corona Australis
    #~ CRA = [359.74400822, -17.51551102] # (l, b)
    #~ ax.scatter(CRA[0], CRA[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('CrA',
                #~ xy=(359.7, -17.5), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Esplin et al. 2018: rho Oph location (they are separating between rho Oph and USco)
    # Rho Ophiuci
    #~ ROPH = [353.22097900, 16.53342332] # (l, b)
    #~ ax.scatter(ROPH[0], ROPH[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate(r'$\rho$ Oph',
                #~ xy=(353, 16), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # IC2391
    #~ IC2391 = [270.36829815, -6.83062731] # (l, b)
    #~ ax.scatter(IC2391[0], IC2391[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('IC2391',
                #~ xy=(270, -7), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # Platais 8
    #~ PL8 = [277.6824, -07.6209] # (l, b)
    #~ ax.scatter(PL8[0], PL8[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('Platais 8',
                #~ xy=(277, -7), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # eps Chamaeleontis
    #~ EPSC = [300.20873944, -15.62481300] # (l, b)
    #~ ax.scatter(EPSC[0], EPSC[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate(r'$\epsilon$ Cha',
                #~ xy=(300, -15), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # eta Chamaeleontis
    #~ ETAC = [292.40233238, -21.65095171] # (l, b)
    #~ ax.scatter(ETAC[0], ETAC[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate(r'$\eta$ Cha',
                #~ xy=(292, -21), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)


    return ax

def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    ax.set_ylim(-20, 40)
    #~ ax.set_xlim(380, 260)
    ax.set_xlim(370, 270)

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
plot_3_windows_gx(ax, labels=True, lw=1, ls='-', c='k')

plt.tight_layout()

# SAVE FIGURES
plt.savefig('gx_input_data.pdf')


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

# New data with RVs, non-bg stars, stars between 100 and 200 pc
fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax=fig.add_subplot(111)
mask = (tab['radial_velocity_error']>1e+10)
for c in good_comps:
    mask = np.logical_or(mask, tab['membership%s'%c]>0.1)
mask = mask & (tab['parallax']>5) & (tab['parallax']<10) # DISTANCE CUT
ax.scatter(tab['l'][mask], tab['b'][mask], s=1, c='k', marker='.', label='')
print('Total number of stars in this plot:', len(tab[mask]))
gx_set_labels_and_ticks_over_360deg(ax)



plt.show()
