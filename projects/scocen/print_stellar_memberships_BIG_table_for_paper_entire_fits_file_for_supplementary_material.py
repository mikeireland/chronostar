"""
Print stellar data (gaia ids, memberships etc.) for the paper.
Prepare two versions: One is the first few lines for the paper (stellar_memberships_example.tex), 
and the other one is an online supplementary material.


Print only members of ScoCen. Background stars are not important!

"""

import numpy as np
from astropy.table import Table, unique, join
import astropy.units as u

from chronostar.component import SphereComponent

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U'] # No B and E
############################################

# Read data
try:
    tab = tab0
except:
    tab0 = Table.read(data_filename)
    tab = tab0

#~ print('UNIQUE TABLE')
#~ tab = unique(tab, keys='source_id')
print(len(tab))

"""
Take only ScoCen members
"""
mask = [False]*len(tab)
for comp_ID in good_comps:
    m = tab['membership%s'%comp_ID]>0.5
    mask = mask | m

tab = tab[mask]
print('Number of stars in good comps with memb. prob. >0.5: ', len(tab))


# Rename some columns
tab.rename_column('bp_rp_extinction_corrected', '(Bp-Rp)0')
tab.rename_column('phot_g_mean_mag_extinction_corrected', 'G0')
tab.rename_column('best_component', 'comp')
tab.rename_column('best_component_membership', 'p')
tab.rename_column('EW(Li)_err', 'EW(Li)_error')


# Keys in the table
keys = ['source_id', '(Bp-Rp)0', 'G0', 'radial_velocity', 'radial_velocity_error', 'X', 'Y', 'Z', 'U', 'V', 'W', 'X_error', 'Y_error', 'Z_error', 'U_error', 'V_error', 'W_error', 'comp', 'p', 'EW(Li)', 'EW(Li)_error', 'Ref']

tab=tab[keys]

tab['radial_velocity'].unit = u.km/u.s
tab['radial_velocity_error'].unit = u.km/u.s
tab['X'].unit = u.pc
tab['Y'].unit = u.pc
tab['Z'].unit = u.pc
tab['X_error'].unit = u.pc
tab['Y_error'].unit = u.pc
tab['Z_error'].unit = u.pc
tab['U'].unit = u.km/u.s
tab['V'].unit = u.km/u.s
tab['W'].unit = u.km/u.s
tab['U_error'].unit = u.km/u.s
tab['V_error'].unit = u.km/u.s
tab['W_error'].unit = u.km/u.s
tab['EW(Li)'].unit = u.angstrom
tab['EW(Li)_error'].unit = u.angstrom


print(tab)

tab.write('scocen_kinematics_and_membership_probabilities_gaiaDR2.fits', overwrite=True)
