"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
with stars missing radial velocities in order to find more Sco-Cen candidates.

Note that this code reads in the Spherical Components (their covariance
matrices are diagonal - that's because currently, Chronostar's output
in the file are only diagonal elements).

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
comps_filename = 'data/final_comps_21.fits'


# Save output to this file. This is a copy of gaia_filename plus newly added memberships
filename_output = 'data/scocen_vac_DR2_with_21_overlaps.fits'
#~ filename_output = 'data/scocen_vac_EDR3_with_21_overlaps.fits'
#~ filename_output = 'data/scocen_vac_EDR3_SUBTABLE_with_bgols_need_to_update_bg_ols_with_21_overlaps_with_nplus1.fits'

# Filename of data you want to compute overlaps for. It should include
# 'background_log_overlap'!!
gaia_filename = 'data/scocen_vac_DR2.fits'
#~ gaia_filename = 'data/scocen_vac_EDR3.fits'
#~ gaia_filename = 'data/scocen_vac_EDR3_SUBTABLE_with_bgols_need_to_update_bg_ols.fits'

# Membership file from the fit. This is used for the component amplitudes. Hopefully works well.
#~ membership_fit_filename = 'data/final_membership.npy'

##################################################
print('Computing component overlaps for %s'%gaia_filename)
print('Output will be saved into %s'%filename_output)
### READ DATA ####################################

# Create components
comps = SphereComponent.load_raw_components(comps_filename)
print('Number of components: %d'%len(comps))

data_table = tabletool.read(gaia_filename)

# This table is masked. Unmask:
data_table=data_table.filled() # TODO: fill with a stupid value!!!!

print('DATA READ', len(data_table))

#~ memberships_fit = np.load(membership_fit_filename)

############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################
# Create data dict for real
historical = 'c_XU' in data_table.colnames
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True, # Must be set to True: This is to INSERT bg_ols into the data_table.
        historical=historical,
)

# COMPONENT OVERLAPS
# Comps are given at time 0, but expectmax.get_all_lnoverlaps() does automatically get current day projection!
# old_memb_probs are used to scale the amplitude of the components
# TODO: This gives all the components the same amplitude! This is not OK!!!
#~ old_memb_probs = np.ones((len(data_dict['means']), len(comps)+1))
#~ old_memb_probs = memberships_fit
nstars=len(data_dict['means'])
ncomps=len(comps)+1 # +1 for background
old_memb_probs = np.ones((nstars, ncomps)) / ncomps
overlaps = expectmax.get_all_lnoverlaps(data_dict, comps, old_memb_probs=old_memb_probs)
print('overlaps.shape', overlaps.shape, len(comps))

# MEMBERSHIP PROBABILITIES (keep in mind that you DO NEED background overlaps for this)!!!
if 'background_log_overlap' not in data_table.colnames:
    print('WARNING: Please provide background overlaps first!!')
membership_probabilities = np.array([expectmax.calc_membership_probs(ol) for ol in overlaps])

comps_fits = Table.read(comps_filename.replace('.npy', '.fits'))

# Create a table
for i in range(membership_probabilities.shape[1]-1):
    comp_id = comps_fits[i]['comp_ID']
    comp_id = comp_id.replace('comp', '')
    data_table['membership%s' % comp_id] = membership_probabilities[:, i]
data_table['membership_bg'] = membership_probabilities[:, -1]

### SAVE DATA #####################################
data_table.write(filename_output, format='fits', overwrite=True)
print('%s written.'%filename_output)
print(data_table)
