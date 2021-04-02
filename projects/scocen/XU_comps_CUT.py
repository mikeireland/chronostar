"""
Plot XU (to investigate age reliability) for the CUT components.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
plt.ion()

#~ import sys
#~ sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
from chronostar.component import SphereComponent
from chronostar import tabletool

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename_fit # Use only data with RV that was actually used in the fit
comps_filename = lib.comps_filename
good_comps = ['C', 'U', 'T']#, 'A', 'G']
compnames = lib.compnames
colors = lib.colors
############################################
# Minimal probability required for membership
pmin_membership = 0.9
############################################


# Read data
try:
    tab = tab0
    comps = comps0
    comps_raw = comps_raw0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)
    comps_raw0 = SphereComponent.load_raw_components(comps_filename)

    tab = tab0
    comps = comps0
    comps_raw = comps_raw0




fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax = fig.add_subplot(111)
dim1=0
dim2=3

for i, c in enumerate(comps):
    # Component data
    comp_ID = c['comp_ID']
    
    # Plot only good comps
    if comp_ID not in good_comps:
        continue
    
    # Take only members
    membname = 'membership%s'%comp_ID
    mask = tab[membname] > pmin_membership
    t=tab[mask]
    
    # Take only stars with RVs
    mask = t['radial_velocity_error']<100
    t = t[mask]

    
    age=c['Age']
    
    # Plot stars
    #~ ax.scatter(t['X'], t['U'], s=1, c=colors[comp_ID], label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    lw=0.5
    if comp_ID=='T':
        zorder=4
    else:
        zorder=1
        
    ax.errorbar(t['X'], t['U'], xerr=t['X_error'], yerr=t['U_error'], c=colors[comp_ID], fmt='o', markersize=3, lw=lw, zorder=zorder, label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))



    # PDS 70
    mask = np.in1d(t['source_id'], 6110141563309613056)
    edgecolor='lime'
    lw=2
    if np.sum(mask)==1:
        ax.scatter(t['X'][mask], t['U'][mask], c=colors[comp_ID], s=50, label = 'PDS 70', edgecolor=edgecolor, linewidth=lw, zorder=5)
        
    
    # Plot components
    comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=False, color=colors[comp_ID], alpha=0.5, ax=ax,
                       comp_orbit=False, orbit_color='red')



# Binaries in T?
#~ binariesT = Table.read('data/binariesTcandidates.fits')
#~ mask = binariesT['radial_velocity_error']<100
#~ ax.scatter(binariesT['X'][mask], binariesT['U'][mask], c='k', s=50, edgecolor='cyan', linewidth=lw, zorder=5, label='Binary T sequence?')


ax.set_xlim(30, 170)
ax.set_ylim(-10, 20)

ax.set_xlabel('X [pc]')
ax.set_ylabel('U [km/s]')

ax.legend(ncol=2, markerscale=5)
fig.subplots_adjust(bottom=0.15, top=0.9)

# LEGEND
handles, labels = ax.get_legend_handles_labels()
labels = [labels[1], labels[3], labels[2], labels[0]]
handles = [handles[1], handles[3], handles[2], handles[0]]
legend=ax.legend(handles, labels, markerscale=1, frameon=True, loc='center right', bbox_to_anchor=(0.28, 0.8), title='Kinematic ages', prop={'size': 8}, facecolor='white', edgecolor='none', framealpha=1)
plt.setp(legend.get_title(),fontsize=10)
#~ legend.legendHandles[3]._sizes = [40]
#~ frame = legend.get_frame()
#~ frame.set_facecolor('white')
#~ frame.set_edgecolor('white')







#~ plt.savefig('XU_comps_CUT.pdf')
    
plt.show()
