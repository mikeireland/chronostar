"""
MZ - 2021 - Mar - 18
Basic plotting script for Chronostar results. Make a quick plot for
the final or partial results of the fit to see what's going on.
Plot CMD, gx plot and XU and XY plots.
"""

import numpy as np
from astropy.table import Table
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from chronostar.component import SphereComponent as Component


### Input info ####################
if len(sys.argv) != 3:
    raise UserWarning('Incorrect usage. \n'
                      '   > python quickplot.py path/to/4/C/final/ path/to/mytable_XYZUVW.fits')

folder = sys.argv[1] # e.g. 4/C/final/
table_filename = sys.argv[2] # tablename with XYZUVW


### Read data #####################
tab = Table.read(table_filename)
memberships = np.load(os.path.join(folder, 'final_membership.npy'))
comps = Component.load_raw_components(os.path.join(folder, 'final_comps.npy'))
###################################

# Get python's default(?) list of colors
#~ prop_cycle = plt.rcParams['axes.prop_cycle']
#~ colors = prop_cycle.by_key()['color']
#~ colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'yellow', ]

# Colormap
norm=plt.Normalize(vmin=vmin, vmax=vmax)
colors = [cm.viridis(norm(i)) for i in range(len(comps))]


def plot_xyzuvw():
    """
    In the velocity plot, filter out stars with very big RV errors.
    This is to remove RV-less stars.
    """

    # Choose which cartesian dimensions you wish to plot
    dims = [('X','Y'), ('U','V'), ('X','U'), ('Z','W')]

    figsize = 8
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize, figsize))
    fig.set_tight_layout(True)

    for ax, (dim1, dim2) in zip(axes.flatten(), dims):
        ax.tick_params(direction='in')
        print(dim1, dim2)

        if dim1 in ['X', 'Y', 'Z']:
            unit1 = 'pc'
        else:
            unit1 = 'km/s'
        if dim2 in ['X', 'Y', 'Z']:
            unit2 = 'pc'
        else:
            unit2 = 'km/s'
        ax.set_xlabel('{} [{}]'.format(dim1, unit1))
        ax.set_ylabel('{} [{}]'.format(dim2, unit2))

        
        
        # If the dimensions are velocity, mask out RV-less stars
        if np.any(np.in1d([dim1, dim2], ['U', 'V', 'W'])):
            mask = tab['radial_velocity_error']<100
            t = tab[mask]
            tmem = memberships[mask]
            
        else:
            t = tab
            tmem = memberships

        # Plot all stars in the table
        ax.scatter(t[dim1], t[dim2], c='grey', alpha=0.5, s=1, label='')
        
        # Plot individual components and their members
        i=0
        for comp in comps:
            comp.plot(dim1, dim2, comp_now=True, comp_then=False, color=colors[i], ax=ax,
                           comp_orbit=False, orbit_color=colors[i])
            
            # Plot only members
            mm=tmem[:,i]>0.5
            ax.scatter(t[dim1][mm], t[dim2][mm], c=colors[i], alpha=1, s=5, label = '%.2f Myr'%comp.get_age())
            
            print('color=', colors[i], comp.get_age())

            i+=1
        print('')
        if dim1=='X' and dim2=='U':
            ax.legend(loc=2)
            
            
    path = os.path.join(folder, 'comps_xyzuvw.png')
    fig.savefig(path)
    print('%s saved.'%path)

def plot_cmd_galaxy():
    ############################################
    # Minimal probability required for membership
    pmin_membership = 0.8
    ############################################
    # CMD limits
    xlim = [-1, 5]
    ylim = [17, -3]
    ############################################
    tab['Gmag'] = tab['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec


    # Plotting purposes
    # Shift in l to put ScoCen in the middle
    #~ print('Shifting l by 100 to put ScoCen in the middle')
    lshift=100
    mask = np.where(tab['l']<lshift)
    tab['l'][mask] = 360 + tab['l'][mask]


    # Stars that are not background
    #~ mask_bg = memberships[:,-1]<0.5


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
        ax.plot(xms, yms, c='brown', linewidth=1) # label='Median main sequence', 
        ax.plot(xms, yms - 1, c='brown', linewidth=1, linestyle='--') # label='1 mag above the median', 
        ax.plot(xms, yms - 1.5, c='brown', linewidth=1, linestyle='--') # label='1.5 mag above the median', 

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
        
        plt.gca().invert_xaxis()
        #~ ax.set_ylim(-40, 60)
        #~ ax.set_xlim(400, 220)
        #~ ax.set_xlim(380, 260)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        t = ax.get_xticks()
        xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
        ax.set_xticklabels(xtick_labels)

        ax.set_xlabel('l [deg]')
        ax.set_ylabel('b [deg]')
        #~ ax.set_aspect('equal')

        return ax

    fig=plt.figure(figsize=(14.0, 8.0))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)


    # Plot CMD for all stars
    ax1.scatter(tab['bp_rp'], tab['Gmag'], s=1, c='k', alpha=0.1, label='')
    
    # Plot all stars in (l, b)
    ax2.scatter(tab['l'], tab['b'], s=1, c='k', alpha=0.1)

    i=0
    for c in comps:
        # Component data
        age=c.get_age()
        comp_ID = i
        mask = memberships[:,i] > pmin_membership
        t=tab[mask]

        if len(t)<1:
            continue

        # Plot CMD
        ax1.scatter(t['bp_rp'], t['Gmag'], s=1, c=colors[i], alpha=1, label = 'Component %s (%.2f Myr) %d'%(comp_ID, age, len(t)))

        # Plot membership histogram
        if len(t)>1:
            mask_m = memberships[:,i]>0.1
            if len(t)>0:
                ax3.hist(memberships[mask_m, i], bins=int(len(t)/20.0), histtype='step', color=colors[i])
            
        # Plot GX
        cb=ax2.scatter(t['l'], t['b'], s=1, c=colors[i], alpha=1)

        i+=1


    ax1=plot_MS_parametrisation_and_spectral_types(ax1, xlim, ylim)
    ax1.set_xlabel('BP-RP')
    ax1.set_ylabel('G')
    ax1.legend(loc=1)
    
    gx_set_labels_and_ticks_over_360deg(ax2)
    
    # For some reason the galaxy thing reverses the ax3
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_xlabel('Membership probability')
    
    plt.tight_layout()
    
    path = os.path.join(folder, 'comps_cmd_gx.png')
    plt.savefig(path)
    print('%s saved.'%path)


plot_xyzuvw()
plot_cmd_galaxy()

# Cornerplot
#~ labels=['X', 'Y', 'Z', 'U', 'V', 'W', 'dX', 'dY', 'dU', 'dV', 'x_rot', 'y_rot', 'z_rot', 'cov_xv', 'age']
#~ chain2=chain.reshape(chain.shape[0]*chain.shape[1], chain.shape[2])
#~ fig = corner.corner(chain2, labels=labels, truths=true_comp.get_pars())
#~ fig.savefig(plt_dir+'corner.pdf')

