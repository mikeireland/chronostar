"""
XU for stars in the fit (stars with RV).
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
good_comps = lib.good_comps
compnames = lib.compnames
colors = lib.colors
############################################
# Minimal probability required for membership
pmin_membership = 0.9
############################################

# Remove the big components that contain a lot of MS stars
good_comps.remove('J')
good_comps.remove('B')
good_comps.remove('Q')



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
    col=tab[membname]
    mask=col>pmin_membership
    t=tab[mask]

    
    age=c['Age']
    
    # Plot stars
    ax.scatter(t['X'], t['U'], s=1, c=colors[comp_ID], label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    # Plot components
    comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=True, color=colors[comp_ID], alpha=0.5, ax=ax,
                       comp_orbit=False, orbit_color='red')


ax.set_xlabel('X [pc]')
ax.set_ylabel('U [km/s]')

ax.legend(ncol=2, markerscale=5)
fig.subplots_adjust(bottom=0.15, top=0.9)

plt.savefig('XU_comps.pdf')
    
plt.show()
