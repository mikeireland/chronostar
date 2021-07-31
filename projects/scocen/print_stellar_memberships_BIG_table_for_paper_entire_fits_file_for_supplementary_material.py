"""
Print stellar data (gaia ids, memberships etc.) for the paper.
Prepare two versions: One is the first few lines for the paper (stellar_memberships_example.tex), 
and the other one is an online supplementary material.


Print only members of ScoCen. Background stars are not important!

"""

import numpy as np
from astropy.table import Table, unique
import astropy.units as u

from chronostar.component import SphereComponent

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U'] # No B and E
print('Revise component LIST!!! there is no B!!!')
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
print('Number of stars in good comps with memb. prob. >0.5: ', len(tab))


from collections import Counter
c_li = Counter(tab['EW(Li)_ref'])
c_rv = Counter(tab['radial_velocity_ref'])


# References
# Merge together references for Lithium and RV as some of them are the same
ewli_ref = set(tab['EW(Li)_ref'])
rv_ref = set(tab['radial_velocity_ref'])
refs = ewli_ref.union(rv_ref)

# Banyan should be last because I should replace it with the actual references there!
refs=list(refs)
refs.append(refs.pop(refs.index('Banyan Sigma')))
refs.append(refs.pop(refs.index('')))
refs.append(refs.pop(refs.index('N/A')))

#~ refs.pop(refs.index(''))
#~ refs.pop(refs.index('N/A'))

ref_dict = dict(zip(refs, range(1, len(refs)+1)))
ref_dict['']=-99
ref_dict['N/A']=-99

print('ref_dict')
print(ref_dict)

tab['Ref'] = [('%d/%d'%(ref_dict[x['radial_velocity_ref']], ref_dict[x['EW(Li)_ref']])).replace('-99/-99', '').replace('-99', '') for x in tab]

ref_set_check = set(tab['Ref'])
print('ref_set_check')
print(ref_set_check)

cr = Counter(tab['Ref'])
print(cr)


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

tab.write('scocen_kinematics_and_membership_probabilities_gaiaDR2.fits')
