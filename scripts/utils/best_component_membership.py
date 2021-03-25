"""
Add columns to the table with best component memberships

"""

import numpy as np
from astropy.table import Table, vstack, unique, join

import sys
sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
import coordinate

############################################
data_filename = '../gaia_200_pc_for_scocen_with_overlaps.fits'
comps_filename = '../final_comps_21.fits'

good_comps_with_ages = ['C', 'U', 'T', 'A', 'G'] # F doesn't have enough members

# Components that are definitely background (based on CMD)
bg_comps = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S']

comps_multiple_pop = ['J', 'B', 'Q'] # Some background and very young stars

############################################

try:
    tab=tab0
except:
    tab0 = Table.read(data_filename)
    tab=tab0


# Read components (WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present!)
comps = Table.read(comps_filename)
comps.rename_column('age', 'Age')

# Crossing time. Only components with sigma<age have reliable ages.
crossing_time = comps['dX']/comps['dV'] * 0.977813106 # pc/km*s to Myr
comps['Crossing_time'] = crossing_time
mask = crossing_time < comps['Age'] # sigma < age
comps['Age_reliable'] = mask

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





# Exclude background stars
mask = (tab['best_component']!='bg') & (tab['best_component_membership']>0.5)
for c in bg_comps:
    mask = mask & (tab['best_component']!=c)
tab_members = tab[mask]
print(tab_members)

tab_members.write('scocen_members.fits', overwrite=True)

