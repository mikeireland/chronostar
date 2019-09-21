"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

"""

import numpy as np
import sys
sys.path.insert(0, '../../')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join

##### FILENAMES ############################################################
datafile = 'data/scocen_data_with_broken_radial_velocities.fits'
#datafile = 'data/scocen_data_with_original_radial_velocities.fits'

output_filename = datafile.replace('data/', 'results/').replace('.fits', '_with_membership.fits')
comps_filename = 'data/all_nonbg_scocen_comps_unique.npy'
############################################################################

data_table = tabletool.read(datafile)

# This table is masked. Unmask:
data_table = data_table.filled()

print('DATA READ', len(data_table))
historical = 'c_XU' in data_table.colnames

############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################

print('Create data dict')
# Create data dict
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        historical=historical,
)

# Create components
comps = SphereComponent.load_raw_components(comps_filename)

# COMPONENT OVERLAPS
overlaps = expectmax.get_all_lnoverlaps(data_dict, comps)
print('overlaps.shape', overlaps.shape, len(comps))

# MEMBERSHIP PROBABILITIES
membership_probabilities = np.array([expectmax.calc_membership_probs(ol) for ol in overlaps])

# Create a table
for i in range(membership_probabilities.shape[1]-1):
    data_table['membership_%d' % (i + 1)] = membership_probabilities[:, i]
data_table['membership_bg'] = membership_probabilities[:, -1]

# Print data
print('WRITE A TABLE WITH PROBABILITIES')
data_table.write(output_filename, format='fits', overwrite=True)
print('Table written.')
print(data_table)
