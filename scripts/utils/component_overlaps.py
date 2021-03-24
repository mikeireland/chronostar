"""
Author: Marusa Zerjal, 2019 - 07 - 15

Compute overlaps between given components and stars. This is useful if e.g.
there are stars that were not used in the fit, e.g. stars with no RVs.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '/Users/marusa/chronostar/')
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
from astropy.table import Table, vstack, join

##################################################
### INPUT #############################
comps_filename = 'final_comps_21.fits'


# Save output to this file. This is a copy of gaia_filename plus newly added memberships
filename_output = 'scocen_vac_eDR3.fits'


# Filename of data you want to compute overlaps for. It should include
# 'background_log_overlap'!!
gaia_filename = 'scocen_vac_eDR3.fits'

##################################################
### READ DATA ####################################

# Create components
comps = SphereComponent.load_raw_components(comps_filename)
print('components', len(comps))

data_table = tabletool.read(gaia_filename)

# This table is masked. Unmask:
data_table=data_table.filled()

print('DATA READ', len(data_table))
historical = 'c_XU' in data_table.colnames


############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################
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
#~ for i in range(membership_probabilities.shape[1]-1):
for i in range(membership_probabilities.shape[1]):
    #~ comp_id = comps_fits[i]['Name']
    comp_id = comps_fits[i]['comp_ID']
    comp_id = comp_id.replace('comp', '')
    print(comp_id)
    #~ data_table['membership_%d' % (i + 1)] = membership_probabilities[:, i]
    data_table['membership%s' % comp_id] = membership_probabilities[:, i]
data_table['membership_bg'] = membership_probabilities[:, -1]

### SAVE DATA #####################################
data_table.write(filename_output, format='fits', overwrite=True)
print('%s written.'%filename_output)
print(data_table)
