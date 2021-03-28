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

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
exclude_components = lib.exclude_components
comps_multiple_pop = lib.comps_multiple_pop
compnames = lib.compnames
colors = lib.colors
############################################

# Minimal probability required for membership
pmin_membership = 0.8
############################################

# Read data
try:
    tab = tab0
    comps = comps0
    comps_raw = comps_raw0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)
    comps_raw0 = Component.load_raw_components(comps_filename)

    tab = tab0
    comps = comps0
    comps_raw = comps_raw0


"""
In the velocity plot, filter out stars with very big RV errors.
This is to remove RV-less stars.
"""

# Choose which cartesian dimensions you wish to plot
dims = [('X','Y'), ('U','V'), ('X','U'), ('Z','W')]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize[1], figsize[0]))
fig.set_tight_layout(True)

for ax, (dim1, dim2) in zip(axes.flatten(), dims):
    # If the dimensions are velocity, mask out RV-less stars
    if np.any(np.in1d([dim1, dim2], ['U', 'V', 'W'])):
        mask = tab['radial_velocity_error']<100
        t = tab[mask]
        
    else:
        t = tab

    #~ print(t)

    # Plot all stars in the table
    #~ ax.scatter(t[dim1], t[dim2], c='grey', alpha=0.5, s=1, label='')
    
    # Plot individual components and their members
    i=0
    for c, c_raw in zip(comps, comps_raw):
        # Plot only members
        comp_id = c['comp_ID']
        if comp_id in exclude_components or comp_id in comps_multiple_pop:
            continue

        c_raw.plot(dim1, dim2, comp_now=True, comp_then=False, color=colors[comp_id], ax=ax,
                       comp_orbit=False, orbit_color=colors[comp_id])
        


        # Take only members of this component
        mask = t['membership%s'%comp_id] > pmin_membership 

        ax.scatter(t[dim1][mask], t[dim2][mask], c=colors[comp_id], alpha=1, s=5, label = '%s %.2f Myr'%(comp_id, c_raw.get_age()))

        i+=1



    # Pretty plot
    ax.tick_params(direction='in')
    
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


    if dim1=='X' and dim2=='U':
        ax.legend(loc=2)
#~ fig.savefig(folder+'comps_xyzuvw.png')
