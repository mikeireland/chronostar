"""
Plot CMDs for CUT components and show that components with higher kinematic
age show less overluminosity in comparison to others.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
comps_to_plot = ['C', 'U', 'T'] # AGE SEQUENCE IS REAL!!! ['C', 'A', 'U', 'G', 'F', 'T']
############################################
# Minimal probability required for membership
pmin_membership = 0.9
############################################
# CMD limits
xlim = [-0.5, 5]
ylim = [16, -4]
############################################


# Read data
try:
    tab = tab0
    comps = comps0
except:
    # This table should include EW(Li) column!
    tab0 = Table.read(data_filename)
    
    Gmag = tab0['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab0['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
    tab0['Gmag'] = Gmag
    
    
    
    
    tab0.rename_column('EW(Li)', 'ewliii')
    
    
    # ADD LITHIUM INFO
    galah = Table.read('/Users/marusa/galah/lithium/ewli/GALAH_EW_Lithium_Zerjal_20181025.fits')
    lithium_galah = galah[['source_id', 'EW(Li)']]
    
    results_2m3 = Table.read('/Users/marusa/observing/23m/paper/data/2m3_results.fits')
    lithium_2m3 = results_2m3[['source_id', 'EW(Li)']]
    
    from astropy.table import vstack, unique, join
    lithium = vstack([lithium_galah, lithium_2m3])
    lithium = unique(lithium, keys=['source_id'], keep='last')  # Some are repeated!

    tab0 = join(tab0, lithium, keys='source_id', join_type='left')
    
    
    
    
    
    
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0

compsd = dict(zip(comps['comp_ID'], comps))


# TAKE ONLY GOOD COMPS, AND SORT THEM BY THEIR AGE
comps_good = []
for l in comps_to_plot:
    for c in comps:
        comp_ID = c['comp_ID']
        if comp_ID==l:
            comps_good.append(c)
            break


def isochrone(ax, plot_young_iso=True):
    """
    Padova isochrones
    """
    
    # 1 Gyr
    iso = np.loadtxt('data/padova_gaiadr2_evans_1gyr_met0.dat', comments='#')
    
    #Skip giants
    last = 80
    first = 2
    
    Gmag = iso[first:last,-8]
    BP = iso[first:last,-7]
    RP = iso[first:last,-6]
    
    ax.plot(BP-RP, Gmag, c='k', lw=0.8, label = '1 Gyr, [M/H]=0')
    
    
    # Young isochrones
    if plot_young_iso:
        iso = np.loadtxt('data/padova_gaiadr2_evans_4myr_met0.dat', comments='#')
        
        #Skip giants
        last = 150
        first = 2
        
        Gmag = iso[first:last,-8]
        BP = iso[first:last,-7]
        RP = iso[first:last,-6]
        
        ax.plot(BP-RP, Gmag, c='k', lw=0.8, ls='--', label = '4 Myr, [M/H]=0')
        
        
        iso = np.loadtxt('data/padova_gaiadr2_evans_15myr_met0.dat', comments='#')
        
        #Skip giants
        last = 150
        first = 2
        
        Gmag = iso[first:last,-8]
        BP = iso[first:last,-7]
        RP = iso[first:last,-6]
        
        ax.plot(BP-RP, Gmag, c='k', lw=0.8, ls=':', label = '15 Myr, [M/H]=0')

def spectral_types(ax, xlim, ylim, y = -2.4, ymin=0.9):

    # This is BP-RP
    lw=0.5
    ax.axvline(x=-0.037, ymin=ymin, linewidth=lw, color='k') # A0, according to Mamajek
    ax.axvline(x=0.369, ymin=ymin, linewidth=lw, color='k') # F
    ax.axvline(x=0.767, ymin=ymin, linewidth=lw, color='k') # G
    ax.axvline(x=0.979, ymin=ymin, linewidth=lw, color='k') # K
    ax.axvline(x=1.848, ymin=ymin, linewidth=lw, color='k') # M0
    #~ ax.axvline(x=4.86, ymin=0.9, linewidth=lw, color='k') # M9 (Mamajek)

    
    # Annotations
    #~ y = -2.4
    c='k'
    s=12
    ax.annotate('A', xy=(-0.037+0.15, y), xytext=(-0.037+0.15, y), color=c, size=s)
    ax.annotate('F', xy=(0.369+0.15, y), xytext=(0.369+0.15, y), color=c, size=s)
    ax.annotate('G', xy=(0.767+0.05, y), xytext=(0.767+0.05, y), color=c, size=s)
    ax.annotate('K', xy=(0.979+0.4, y), xytext=(0.979+0.4, y), color=c, size=s)
    #~ ax.annotate('M', xy=(1.848+0.5, y), xytext=(1.848+0.5, y), color=c, size=s)
    ax.annotate('M0', xy=(1.848+0.1, y), xytext=(1.848+0.1, y), color=c, size=s)
    ax.annotate('M5', xy=(3.35-0.05, y), xytext=(3.35-0.05, y), color=c, size=s)
    ax.annotate('M9', xy=(4.86-0.1, y), xytext=(4.86-0.1, y), color=c, size=s)

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
        
    return ax


fig = plt.figure(figsize=(figsize[1], figsize[0]*1.3))
gs = GridSpec(3, 1) # fig=fig
ax = fig.add_subplot(gs[1:, :])
ax2 = inset_axes(ax, width="100%", height="100%", loc=1, bbox_to_anchor=(0.7, 0.5, .3, .4), bbox_transform=ax.transAxes) 
ax3 = fig.add_subplot(gs[0, :])


for c2 in comps_to_plot:
    c = compsd[c2]

    comp_ID = c['comp_ID']        
    membname = 'membership%s'%comp_ID
    age=c['Age']
    
    mask=tab[membname]>pmin_membership
    t=tab[mask]

    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c=colors[comp_ID], label=r'%s %.1f $\pm$ %.1f Myr'%(comp_ID, age, c['Crossing_time']))
    
    # CUT
    ax2.scatter(t['bp_rp'], t['Gmag'], s=20, c=colors[comp_ID], label=r'%s %.1f $\pm$ %.1f Myr'%(comp_ID, age, c['Crossing_time']))
    
    # AG
    #~ if comp_ID=='G':
    #~ ax2.scatter(t['bp_rp'], t['Gmag'], s=1, c=colors[comp_ID], label=r'%s %.1f $\pm$ %.1f Myr'%(comp_ID, age, c['Crossing_time']))
    
    
    # Lithium
    ax3.scatter(t['bp_rp'], t['EW(Li)'], s=10, c=colors[comp_ID], label='')
    
    # PDS 70
    mask = np.in1d(t['source_id'], 6110141563309613056)
    edgecolor='lime'
    lw=2
    if np.sum(mask)==1:
        ax3.scatter(t['bp_rp'][mask], t['EW(Li)'][mask], s=50, c=colors[comp_ID], label='PDS 70', edgecolor=edgecolor, linewidth=lw)        
        
        ax.scatter(t['bp_rp'][mask], t['Gmag'][mask], s=50, c=colors[comp_ID], edgecolor=edgecolor, label='PDS 70', linewidth=lw)
        ax2.scatter(t['bp_rp'][mask], t['Gmag'][mask], s=50, c=colors[comp_ID], edgecolor=edgecolor, label='', linewidth=lw)



isochrone(ax, plot_young_iso=False)
isochrone(ax2, plot_young_iso=False)


### Make plots pretty


# INSET
# CUT
#~ ax2=spectral_types(ax2, xlim, ylim, y = 3)
xlim2 = [0.7, 1.8]
ylim2 = [8, 2]
ax2.set_xlim(xlim2[0], xlim2[1])
ax2.set_ylim(ylim2[0], ylim2[1])

#~ ax2.indicate_inset_zoom(axins)


# Region to zoom in
lw=0.5
ls='-'
c='grey'
ax.plot(xlim2, [ylim2[0], ylim2[0]], linewidth=lw, color=c, linestyle=ls)
ax.plot(xlim2, [ylim2[1], ylim2[1]], linewidth=lw, color=c, linestyle=ls)
ax.plot([xlim2[0], xlim2[0]], ylim2, linewidth=lw, color=c, linestyle=ls)
ax.plot([xlim2[1], xlim2[1]], ylim2, linewidth=lw, color=c, linestyle=ls)

# Lines connecting the subplot
ax.plot([xlim2[1], 4.95], [ylim2[0], 6.4], linewidth=lw, color=c, linestyle=ls)
ax.plot([xlim2[0], 3.31], [ylim2[1], -1.6], linewidth=lw, color=c, linestyle=ls)


#~ ax.axvline(x=xlim2[1], ymin=ylim2[0], ymax=ylim2[1], linewidth=lw, color=c)
#~ ax.axhline(y=ylim2[0], xmin=xlim2[0]/(xlim[1]-xlim[0]), xmax=xlim2[1]/(xlim[1]-xlim[0]), linewidth=lw, color=c)
#~ ax.axhline(y=ylim2[1], xmin=xlim2[0], xmax=xlim2[1], linewidth=lw, color=c)


# AG
#~ #ax2=spectral_types(ax2, xlim, ylim, y = 9)
#~ ax2.set_xlim(2.5, 4)
#~ ax2.set_ylim(14, 7)

ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position("top")
ax2.xaxis.tick_top()
#~ ax2.tick_params(direction='out', length=6, width=0.5)
ax2.tick_params(direction='out', length=0, width=0.5)
plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

   
# LEGEND
handles, labels = ax.get_legend_handles_labels()
labels = [labels[1], labels[2], labels[3], labels[4], labels[0]]
handles = [handles[1], handles[2], handles[3], handles[4], handles[0]]
legend=ax.legend(handles, labels, markerscale=5, frameon=False, loc='center right', bbox_to_anchor=(0.23, 0.23), title='Kinematic ages', prop={'size': 8})
plt.setp(legend.get_title(),fontsize=10)
legend.legendHandles[3]._sizes = [40]


#~ ax3.legend(loc='center left', frameon=False)

# Lithium axis
#~ ax3=spectral_types(ax3, xlim, [-0.2, 1], y = 0.65)
#~ ax3=spectral_types(ax3, xlim, [-0.2, 1], y = 0.8, ymin=0.8)
ax3.set_ylabel(r'EW(Li) [\AA]')
ax3.set_xlabel('Bp-Rp')
ax3.xaxis.set_label_position("top")
ax3.xaxis.tick_top()
ax3.tick_params(direction='out', length=6, width=0.5)
ax3.set_xlim(xlim[0], xlim[1])
ax3.set_ylim(-0.2, 0.8)
#~ ax3.set_ylim(-0.2, 1)
t = ax3.get_yticks()
ytick_labels = ['%.1f'%x if x>-0.1 else '' for x in t]
ax3.set_yticklabels(ytick_labels)


ax=spectral_types(ax, xlim, ylim)
#~ ax.set_xlim(xlim[0], xlim[1])
#~ ax.set_ylim(ylim[0], ylim[1])
ax.set_xlabel('Bp-Rp')
ax.set_ylabel(r'$M\mathrm{_G}$')

ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))

t = ax.get_yticks()
ytick_labels = [int(x) if x>-3 else '' for x in t]
ax.set_yticklabels(ytick_labels)

fig.subplots_adjust(bottom=0.1, top=0.9)
fig.subplots_adjust(hspace=0, wspace=0)

#~ plt.savefig('cmd_li_CUT_90percent_membership.pdf')
#~ plt.savefig('cmd_AGF.pdf')
#~ plt.savefig('cmd_li_AG.pdf')
plt.show()
