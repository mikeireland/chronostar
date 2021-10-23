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


### EXAMPLE TABLE
# Print stars with some lithium and RV available
mask = (tab['EW(Li)']>0.1) & (tab['radial_velocity_error']<10)
tab=tab[mask]


# References
# Merge together references for Lithium and RV as some of them are the same
ewli_ref = set(tab['EW(Li)_ref'])
rv_ref = set(tab['radial_velocity_ref'])
refs = ewli_ref.union(rv_ref)

# Banyan should be last because I should replace it with the actual references there!
refs=list(refs)
refs.append(refs.pop(refs.index('Banyan Sigma')))

ref_dict = dict(zip(refs, range(1, len(refs)+1)))
print('ref_dict')
print(ref_dict)


def table_with_basic_columns():
    # Labels. BP-RP is dereddened
    #~ print('source\_id & bp\_rp & BP-RP & G & G\_ext & RV & $\sigma_\mathrm{RV}$ & RV\_ref& best\_comp & best\_comp\_membership & EW(Li) & $\sigma\mathrm{EW(Li)}$ & EW(Li)\_ref \\\\')
    print('source\_id & bp\_rp & BP-RP & G & G\_ext & RV & $\sigma_\mathrm{RV}$ & best\_comp & best\_comp\_membership & EW(Li) & $\sigma\mathrm{EW(Li)}$ & Ref \\\\')


    # Units
    print('Gaia\,DR2 & & & & & & & & & & & & RV/Li \\\\')



    for x in tab[:50]:
        line = '%d & '%x['source_id']
        line += '%.2f & '%x['bp_rp']
        line += '%.2f & '%x['bp_rp_extinction_corrected']
        line += '%.2f & '%x['phot_g_mean_mag']
        line += '%.2f & '%x['phot_g_mean_mag_extinction_corrected']
        line += '%.1f & '%x['radial_velocity']
        line += '%.1f & '%x['radial_velocity_error']
        #~ line += '%d & '%rv_ref_dict[x['radial_velocity_ref']]
        line += '%s & '%x['best_component']
        line += '%.2f & '%x['best_component_membership']
        line += '%.2f & '%x['EW(Li)']
        line += '%.2f & '%x['EW(Li)_err']
        #~ line += '%d '%ewli_ref_dict[x['EW(Li)_ref']]
        line += '%d/%d '%(ref_dict[x['radial_velocity_ref']], ref_dict[x['EW(Li)_ref']])

        line += ' \\\\'
        
        line = line.replace('nan', '   ')
        line = line.replace('N/A', '   ')


        print(line)

def table_with_basic_columns_and_kinematics():
    print('source\_id & (BP-RP)$_0$ & G$_0$ & RV & $\sigma_\mathrm{RV}$ & X & Y & Z & U & V & W & $\sigma_\mathrm{X}$ & $\sigma_\mathrm{Y}$ & $\sigma_\mathrm{Z}$ & $\sigma_\mathrm{U}$ & $\sigma_\mathrm{V}$ & $\sigma_\mathrm{W}$ & comp & p & EW(Li) & $\sigma\mathrm{EW(Li)}$ & Ref \\\\')


    # Units
    print('Gaia\,DR2 & & & & & & & & & & & & & & & & & & & \AA & \AA & RV/Li \\\\')


    N=35
    i=0
    for x in tab:
        if x['X_error']>0.09 and x['Y_error']>0.09 and x['Z_error']>0.09 and x['U_error']>0.09 and x['V_error']>0.09 and x['W_error']>0.09:
            pass
        else:
            continue
        
        line = '%d & '%x['source_id']
        line += '%.2f & '%x['bp_rp_extinction_corrected']
        line += '%.2f & '%x['phot_g_mean_mag_extinction_corrected']
        line += '%.1f & '%x['radial_velocity']
        line += '%.1f & '%x['radial_velocity_error']
        
        line += '%.1f & '%x['X']
        line += '%.1f & '%x['Y']
        line += '%.1f & '%x['Z']
        line += '%.1f & '%x['U']
        line += '%.1f & '%x['V']
        line += '%.1f & '%x['W']
        
        line += '%.1f & '%x['X_error']
        line += '%.1f & '%x['Y_error']
        line += '%.1f & '%x['Z_error']
        #~ if '%.1f & '%x['Z_error']=='0.0 & ':
            #~ print('**********', x['Z_error'])
        line += '%.1f & '%x['U_error']
        line += '%.1f & '%x['V_error']
        line += '%.1f & '%x['W_error']
        
        line += '%s & '%x['best_component']
        line += '%.2f & '%x['best_component_membership']
        line += '%.2f & '%x['EW(Li)']
        line += '%.2f & '%x['EW(Li)_err']
        #~ line += '%d '%ewli_ref_dict[x['EW(Li)_ref']]
        line += '%d/%d '%(ref_dict[x['radial_velocity_ref']], ref_dict[x['EW(Li)_ref']])

        line += ' \\\\'
        
        line = line.replace('nan', '   ')
        line = line.replace('N/A', '   ')


        print(line)
        i+=1
        
        if i==N:
            break

table_with_basic_columns_and_kinematics()
