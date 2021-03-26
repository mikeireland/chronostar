"""
Plot CMD and gx for each component to see if they are healthy.
Add XU plot.....
"""

import numpy as np
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import os
plt.ion()

from chronostar.component import SphereComponent
from chronostar import tabletool

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
data_filename_fit = lib.data_filename_fit # Data with RV that was actually used in the fit
comps_filename = lib.comps_filename
good_comps = lib.good_comps
compnames = lib.compnames
fig_folder = 'fig'
############################################
# Minimal probability required for membership
pmin_membership = 0.8
############################################
# CMD limits
xlim = [-1, 5]
ylim = [17, -3]
############################################

# Read data
try:
    tab = tab0
    tab_fit = tab_fit0
    comps = comps0
    comps_raw = comps_raw0
except:
    tab0 = Table.read(data_filename)
    tab_fit0 = Table.read(data_filename_fit)

    tab0['Gmag'] = tab0['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab0['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
    
    tab_fit0['Gmag'] = tab_fit0['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab_fit0['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec

    comps0 = Table.read(comps_filename)
    comps_raw0 = SphereComponent.load_raw_components(comps_filename)

    tab = tab0
    tab_fit = tab_fit0
    comps = comps0
    comps_raw = comps_raw0
    

# Plotting purposes
# Shift in l to put ScoCen in the middle
print('Shifting l by 100 to put ScoCen in the middle')
lshift=100
mask = np.where(tab_fit['l']<lshift)
tab_fit['l'][mask] = 360 + tab_fit['l'][mask]
mask = np.where(tab['l']<lshift)
tab['l'][mask] = 360 + tab['l'][mask]


# Stars that are not background
mask_bg = tab['membership_bg']<0.5
mask_bg_fit = tab_fit['membership_bg']<0.5

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

def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    ax.set_ylim(-40, 60)
    ax.set_xlim(400, 220)
    #~ ax.set_xlim(380, 260)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    #~ ax.set_aspect('equal')

    return ax

for i, c in enumerate(comps):
    # Component data
    age=c['Age']
    comp_ID = c['comp_ID']
    membname = 'membership%s'%comp_ID
    col=tab[membname]
    mask=col>pmin_membership
    t=tab[mask]


    if len(t)<1:
        continue

    fig=plt.figure(figsize=(14.0, 8.0))

    plt.suptitle('%s (%.2f$\pm$%.2f Myr %s) %d'%(comp_ID, age, c['Crossing_time'], c['Age_reliable'], len(t)))

    t.sort(membname)
    #~ t.reverse() 
    
    # Plot CMD
    ax = fig.add_subplot(2, 2, 1)

    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c=t[membname], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)

    ax=plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim)
    
    ax.set_xlabel('BP-RP')
    ax.set_ylabel('G')
    
    # Plot XY: TODO: Only stars used in the fit
    ax = fig.add_subplot(2, 3, 4)
    dim1=0
    dim2=1
    ax.scatter(t['X'], t['Y'], s=1, c=t[membname], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)
    comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=False, color='k', alpha=1, ax=ax,
                       comp_orbit=False, orbit_color='red')
    ax.set_xlabel('X [pc]')
    ax.set_ylabel('Y [km/s]')
    
    # Plot XU
    #~ ax = fig.add_subplot(2, 2, 4)
    ax = fig.add_subplot(2, 3, 5)
    dim1=0
    dim2=3
    ax.scatter(t['X'], t['U'], s=1, c=t[membname], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)
    comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=False, color='k', alpha=1, ax=ax,
                       comp_orbit=False, orbit_color='red')
    ax.set_xlabel('X [pc]')
    ax.set_ylabel('U [km/s]')


    # Plot GX
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(tab['l'][mask_bg], tab['b'][mask_bg], s=1, c='k', alpha=0.1)
    cb=ax.scatter(t['l'], t['b'], s=1, c=t[membname], alpha=1, vmin=0.5, vmax=1, cmap=cm.jet)
    gx_set_labels_and_ticks_over_360deg(ax)
    #~ cbar=plt.colorbar(cb)
    #~ cbar.ax.set_ylabel('Membership')


    
    # Plot membership histogram
    if len(t)>1:
        ax = fig.add_subplot(2, 3, 6)
        #~ ax.hist(t[membname], bins=int(len(t)/20.0))
        mask_m = tab[membname]>0.1
        if len(t)>0:
            ax.hist(tab[membname][mask_m], bins=int(len(t)/20.0))
        ax.set_xlabel('Membership probability')


    #~ plt.tight_layout()
    
    plt.savefig(os.path.join(fig_folder, '%s.png'%comp_ID))
    
plt.show()
