"""
Take members of a component and write them in a separate table.
I would like to run chronostar just on this and split the data further
into at more components.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt


############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
############################################
# Minimal probability required for membership
#~ pmin_membership = 0.01
#~ pmin_membership = 0.2
#~ pmin_membership = 0.5
############################################

# Read data
try:
    tab=tab0
except:
    tab0 = Table.read(data_filename)
    tab=tab0
    

######################################################
# Component with PDS 70, there are many subcomponents in there
#~ mask = tab['membershipT'] > 0.01
#~ print(np.sum(mask))
#~ tab[mask].write('data/starsT_for_splitting_DR2.fits', overwrite=True)
#~ ######################################################
#~ # Component with MS and PMS sequence
#~ mask = tab['membershipB'] > 0.5
#~ print(np.sum(mask))
#~ tab[mask].write('data/starsB_for_splitting_DR2.fits', overwrite=True)
#~ ######################################################
# Component with MS and PMS sequence
mask = tab['membershipJ'] > 0.2
#~ #mask = tab['best_component_50']=='J'
print(np.sum(mask))
tab[mask].write('data/starsJ_for_splitting_DR2.fits', overwrite=True)
######################################################
#~ # Component with MS and PMS sequence
#~ mask = tab['membershipQ'] > 0.5
#mask = tab['best_component_50']=='Q'
#~ print(np.sum(mask))
#~ tab[mask].write('data/starsQ_for_splitting_DR2.fits', overwrite=True)
#~ ######################################################


#~ plt.scatter(tab['l'][mask], tab['b'][mask], s=1)
#~ plt.gca().invert_xaxis()





# Choose which cartesian dimensions you wish to plot
dims = [('X','Y'), ('U','V'), ('X','U'), ('Z','W')]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
fig.set_tight_layout(True)

tab=tab[mask]
for ax, (dim1, dim2) in zip(axes.flatten(), dims):
    # If the dimensions are velocity, mask out RV-less stars
    if np.any(np.in1d([dim1, dim2], ['U', 'V', 'W'])):
        maskrv = tab['radial_velocity_error']<100
        t = tab[maskrv]
        
    else:
        t = tab

    #~ print(t)

    # Plot all stars in the table
    ax.scatter(t[dim1], t[dim2], c='k', alpha=1, s=1, label='')
    
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



plt.show()
