"""
Take members of a component and write them in a separate table.
I would like to run chronostar just on this and split the data further
into at more components.
"""

import numpy as np
from astropy.table import Table


############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
############################################
# Minimal probability required for membership
pmin_membership = 0.01
############################################

# Read data
try:
    tab=tab0
except:
    tab0 = Table.read(data_filename)
    tab=tab0
    

####### Component with CrA  ##########################
#~ mask = tab['membershipB'] > pmin_membership
#~ print(np.sum(mask))
#~ tab[mask].write('starsB_for_splitting.fits')
######################################################
# Component with PDS 70, there are many subcomponents in there
mask = tab['membershipT'] > pmin_membership
print(np.sum(mask))
#~ tab[mask].write('starsT_for_splitting.fits')
######################################################
