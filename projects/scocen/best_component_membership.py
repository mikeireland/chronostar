"""
Add columns to the table with best component memberships

"""

import numpy as np
from astropy.table import Table


############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
#~ data_filename = lib.data_filename
data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges3_li_XYZUVW_within200pc_overlaps_with_21_components.fits'
comps_filename = lib.comps_filename
comps_multiple_pop = lib.comps_multiple_pop
bg_comps = lib.bg_comps
############################################

# Minimal probability required for membership
pmin_membership = 0.5
############################################

# Read data
try:
    tab = tab0
    comps = comps0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0


membership_colnames = ['membership%s'%comp_ID for comp_ID in comps['comp_ID']]
membership_colnames.append('membership_bg')

memberships = np.array([tab[col] for col in membership_colnames]).T

ind = np.argwhere(memberships==np.amax(memberships,1, keepdims=True))[:,1]

best_group = np.array([membership_colnames[i].replace('membership', '').replace('_', '') for i in ind])
best_group_membership = np.array([tab[membership_colnames[i]][x] for x, i in enumerate(ind)])

tab['best_component'] = best_group
tab['best_component_membership'] = best_group_membership

# Comments on components that need further splitting
comments = ['Component needs further split' if comp_ID in comps_multiple_pop else '' for comp_ID in tab['best_component']]
tab['best_component_comment'] = comments

tab['best_component_80'] = ['  ']*len(tab)
mask = best_group_membership>0.8
tab['best_component_80'][mask] = best_group[mask]

tab['best_component_50'] = ['  ']*len(tab)
mask = best_group_membership>0.5
tab['best_component_50'][mask] = best_group[mask]

tab['best_component_90'] = ['  ']*len(tab)
mask = best_group_membership>0.9
tab['best_component_90'][mask] = best_group[mask]


compsd = dict(zip(comps['comp_ID'], comps))
# AGES
ages = [compsd[comp_ID]['Age'] if comp_ID!='bg' else np.nan for comp_ID in tab['best_component']]
tab['age_best_component'] = ages

crossing_time = [compsd[comp_ID]['Crossing_time'] if comp_ID!='bg' else np.nan for comp_ID in tab['best_component']]
tab['crossing_time_best_component'] = crossing_time

age_reliable = [compsd[comp_ID]['Age_reliable'] if comp_ID!='bg' else np.nan for comp_ID in tab['best_component']]
tab['age_reliable_best_component'] = age_reliable


tab.write(data_filename, overwrite=True)


# Exclude background stars
mask = (tab['best_component']!='bg') & (tab['best_component_membership']>0.5)
for c in bg_comps:
    mask = mask & (tab['best_component']!=c)
tab_members = tab[mask]
print(tab_members)

#~ tab_members.write('scocen_members.fits', overwrite=True)

