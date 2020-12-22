"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '/Users/marusa/chronostar/')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join


# Create components
#c = np.load('all_nonbg_scocen_comps.npy') # including LCC
#comps = [SphereComponent(pars=x) for x in c]
#~ comps = SphereComponent.load_raw_components('data/all_nonbg_scocen_comps.npy')
comps_filename = 'final_comps.fits'
#~ comps = Table.read(comps_filename)
comps = SphereComponent.load_raw_components(comps_filename)
print('components', len(comps))

filename_output = 'members_from_5k_fit_and_overlaps.fits'
datafile = 'members_from_5k_fit_and_overlaps.fits'

# Read Gaia data including both stars with known and missing radial velocities
#datafile = 'data_table_cartesian_100k.fits'
data_table = tabletool.read(datafile)

# This table is masked. Unmask:
data_table=data_table.filled()

print('DATA READ', len(data_table))
historical = 'c_XU' in data_table.colnames


############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################

print('Create data dict')
# Create data dict for real
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        historical=historical,
)


# COMPONENT OVERLAPS
# Comps are given at time 0, but expectmax.get_all_lnoverlaps() does automatically get current day projection!
overlaps = expectmax.get_all_lnoverlaps(data_dict, comps)
print('overlaps.shape', overlaps.shape, len(comps))

# MEMBERSHIP PROBABILITIES (keep in mind that you DO NEED background overlaps for this)!!!
if 'background_log_overlap' not in data_table.colnames:
    print('WARNING: Please provide background overlaps first!!')
    #~ exit()
membership_probabilities = np.array([expectmax.calc_membership_probs(ol) for ol in overlaps])

comps_fits = Table.read(comps_filename.replace('.npy', '.fits'))

# Create a table
for i in range(membership_probabilities.shape[1]-1):
    comp_id = comps_fits[i]['Name']
    comp_id = comp_id.replace('comp', '')
    print(comp_id)
    #~ data_table['membership_%d' % (i + 1)] = membership_probabilities[:, i]
    data_table['membership%s' % comp_id] = membership_probabilities[:, i]
data_table['membership_bg'] = membership_probabilities[:, -1]

# Print data
data_table.write(filename_output, format='fits', overwrite=True)
print('%s written.'%filename_output)
print(data_table)
