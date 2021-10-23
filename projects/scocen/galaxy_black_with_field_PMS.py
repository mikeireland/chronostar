"""
Plot ScoCen in (l, b) coordinates, stars for each component with different
colours.
"""

import numpy as np
from astropy.table import Table, join
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
exclude_components = lib.exclude_components
exclude_components.append('J')
exclude_components.append('B')
exclude_components.append('Q')
############################################

# Minimal probability required for membership
pmin_membership = 0.5 # nice
#~ pmin_membership = 0.9
#~ pmin_membership = 0.1
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

# Shift data in galactic 'l' to put ScoCen in the middle of the plot
lshift=100
mask = np.where(tab['l']<lshift)
tab['l'][mask] = 360 + tab['l'][mask]

# Take only stars between 100 and 200 pc
#~ mask = (tab['parallax']>5) & (tab['parallax']<10)
#~ tab = tab[mask]
#~ print('Only stars between 100 and 200 pc.')


#### PLOTTING ###############################
fig=plt.figure(figsize=(figsize[1], figsize[0]*1.3))
ax=fig.add_subplot(111)

total=0 # Total number of stars in the plot

# Plot components (stars) one by one with different colours
for c in comps:
    comp_id = c['comp_ID']
    if comp_id in exclude_components:
        continue

    # Take only members of this component
    mask = tab['membership%s'%comp_id] > pmin_membership 
    t=tab[mask]
    
    #~ print(comp_id, colors[comp_id])
    
    # PLOT STARS

    
    # Black
    cb=ax.scatter(t['l'], t['b'], s=1, c='r', marker='.', label='Sco-Cen members')
    
    # Colors: memberships
    #~ t.sort('membership%s'%comp_id)
    #~ cb=ax.scatter(t['l'], t['b'], s=1, c=t['membership%s'%comp_id], marker='.', vmin=0.5, vmax=1)
    
    total+=len(t)

    
print('Total number of stars in this plot:', total)




# Add JQ PMS M dwarfs
jq = Table.read('data/input_may2021/scocen_vac_DR2_distinct_JQ_7k_stars_for_a_new_fit.fits')
lshift=100
mask = np.where(jq['l']<lshift)
jq['l'][mask] = 360 + jq['l'][mask]
ax.scatter(jq['l'], jq['b'], s=1, c='k', marker='.')




# FIND GAIA PMS STARS NOT IN SCOCEN
mask_not_in_scocen = tab['l']<-1000 # all False
for c in comps:
    comp_id = c['comp_ID']
    if comp_id in exclude_components:
        continue

    # Take only members of this component
    mask_not_in_scocen = mask_not_in_scocen | (tab['membership%s'%comp_id] > pmin_membership)

mask_not_in_scocen = ~mask_not_in_scocen
    
# Take only PMS M stars
mask_mdwarf = (tab['bp_rp']>1.8) & (tab['Gmag']>5) # some giants there...

# Main sequence parametrization
# fitpar for pmag, rpmag
fitpar = [0.17954163, -2.48748376, 12.9279348, -31.35434182, 38.31330583, -12.25864507]
p = np.poly1d(fitpar)

mask_pms = tab['Gmag'] < p(tab['bp_rp'])-1

mask_ruwe = tab['ruwe']<1.4

mask_not = mask_not_in_scocen & mask_mdwarf & mask_pms & mask_ruwe


# Originally used this before reddening was computed. Now use the one below instead.
#~ ax.scatter(tab['l'][mask_not], tab['b'][mask_not], s=1, c='k', marker='.', alpha=1, label = 'PMS M dwarfs\n(photometric selection)\nnot in this Sco-Cen model')

#~ print('%d stars in black'%len(set(tab[mask_not]['source_id'])))


#~ tab[mask_not].write('black_stars_compute_reddening.fits')


# PMS stars not in scocen - reddening
pms_reddening = Table.read('data/black_stars_compute_reddening.fits')
black_stars = Table.read('black_stars_compute_reddening.fits')
black_stars.remove_columns(['Gmag_extinction_corrected', 'phot_g_mean_mag_extinction_corrected', 'phot_bp_mean_mag_extinction_corrected','phot_rp_mean_mag_extinction_corrected', 'Gmag'])
black_stars = join(black_stars, pms_reddening, keys='source_id')
print(black_stars.colnames)
mask_mdwarf = (black_stars['phot_bp_mean_mag_extinction_corrected']-black_stars['phot_rp_mean_mag_extinction_corrected']>1.8) & (black_stars['Gmag_extinction_corrected']>5) # some giants there...
mask_pms = black_stars['Gmag_extinction_corrected'] < p(black_stars['phot_bp_mean_mag_extinction_corrected']-black_stars['phot_rp_mean_mag_extinction_corrected'])-1
mask_not = mask_mdwarf & mask_pms
ax.scatter(black_stars['l'][mask_not], black_stars['b'][mask_not], s=1, c='k', marker='.', alpha=1, label = 'PMS M dwarfs\n(photometric selection)\nnot in this Sco-Cen model')

print('%d stars in black'%len(set(black_stars[mask_not]['source_id'])))


#~ tab[mask_not].write('data/field_PMS_stars.fits')


#~ # CMD limits
#~ xlim = [-1, 5]
#~ ylim = [17, -3]
#~ def plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim):
    #~ x = np.linspace(1, 4, 100)
    #~ y = p(x)
    #~ m = y > 4
    #~ yms = y[m]
    #~ xms = x[m]
    #~ ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
    #~ ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    #~ ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

    #~ ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
    #~ ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
    #~ ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
    #~ ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

    #~ ax.set_xlim(xlim[0], xlim[1])
    #~ ax.set_ylim(ylim[0], ylim[1])
        
    #~ return ax

#~ fig2=plt.figure()
#~ ax2=fig2.add_subplot(111)
#~ ax2.scatter(tab['bp_rp'][mask_not], tab['Gmag'][mask_not], s=1)
#~ ax2=plot_MS_parametrisation_and_spectral_types(ax2, xlim, ylim)





def plot_3_windows_gx(ax, labels=True, lw=2, ls='-', c='b'):
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
    
    # UCL
    ax.plot([350, 350], [0, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 312], [0, 25], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 350], [0, 0], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 312], [25, 25], c=c, linestyle=ls, linewidth=lw)
    
    # LCC
    plot_window(ax, x1=312, x2=285, y1=-10, y2=23, c=c, ls=ls, lw=lw)


    return ax

def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    #~ ax.set_ylim(-40, 60)
    ax.set_ylim(-40, 50)
    ax.set_xlim(400, 240)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    ax.set_aspect('equal')

    return ax




plot_3_windows_gx(ax, labels=True, lw=1, ls='-', c='k')
gx_set_labels_and_ticks_over_360deg(ax)


ax.legend()

# LEGEND with parameterised component T
handles, labels = ax.get_legend_handles_labels()
labels = [labels[0], labels[-1]]
handles = [handles[0], handles[-1]]
#~ legend=ax.legend(handles, labels, markerscale=5, frameon=False, loc='center right', bbox_to_anchor=(0.23, 0.23), title='Kinematic ages', prop={'size': 8})
legend=ax.legend(handles, labels, markerscale=5, frameon=True, loc=1, facecolor='white')
#~ legend.get_title().set_position((-35, 0))
legend.legendHandles[0]._sizes = [40]
legend.legendHandles[1]._sizes = [40]
#~ frame.set_facecolor('white')
legend.get_frame().set_alpha(None)

# SAVE FIGURES
#~ plt.savefig('gx_black_members_red_PMS_field_M_dwarfs.pdf', format='pdf')

ax.set_rasterized(True)
#~ plt.savefig('gx_black_members_red_PMS_field_M_dwarfs_rasterized.pdf', format='pdf')

plt.show()
