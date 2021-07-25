"""
Print stellar data (gaia ids, memberships etc.) for the paper.
Prepare two versions: One is the first few lines for the paper (stellar_memberships_example.tex), 
and the other one is an online supplementary material.


Print only members of ScoCen. Background stars are not important!

"""

import numpy as np
from astropy.table import Table, unique

from chronostar.component import SphereComponent

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
good_comps = ['A', 'C', 'D', 'F', 'G', 'H', 'I', 'T', 'U'] # No B and E
print('Revise component LIST!!! there is no E or B now!!!')
############################################

# Read data
try:
    tab = tab0
except:
    tab0 = Table.read(data_filename)
    tab = tab0

print('UNIQUE TABLE')
tab = unique(tab, keys='source_id')

"""
Take only ScoCen members
"""
mask = tab['source_id']<1000 # All False
for comp_ID in good_comps:
    m = tab['membership%s'%comp_ID]>0.5
    mask = mask | m

tab = tab[mask]
print(len(tab))



### EXAMPLE TABLE
# Print stars with some lithium and RV available
mask = (tab['EW(Li)']>0.1) & (tab['radial_velocity_error']<10)
tab=tab[mask]




# Labels. BP-RP is dereddened
print('source\_id & bp\_rp & BP-RP & g & g\_ext & RV & $\sigma_\mathrm{RV}$ & RV\_ref& best\_comp & best\_comp\_membership & EW(Li) & $\sigma\mathrm{EW(Li)}$ & EW(Li)\_ref \\\\')


# Units
print('Gaia\,DR2 & & & & & & & & & & & & \\\\')



for x in tab[:100]:
    line = '%d & '%x['source_id']
    line += '%.2f & '%x['bp_rp']
    line += '%.2f & '%x['bp_rp_extinction_corrected']
    line += '%.2f & '%x['phot_g_mean_mag']
    line += '%.2f & '%x['phot_g_mean_mag_extinction_corrected']
    line += '%.1f & '%x['radial_velocity']
    line += '%.1f & '%x['radial_velocity_error']
    line += '%s & '%x['radial_velocity_ref']
    line += '%s & '%x['best_component']
    line += '%.2f & '%x['best_component_membership']
    line += '%.2f & '%x['EW(Li)']
    line += '%.2f & '%x['EW(Li)_err']
    line += '%s  '%x['EW(Li)_ref']

    line += ' \\\\'
    
    line = line.replace('nan', '   ')
    line = line.replace('N/A', '   ')


    print(line)
