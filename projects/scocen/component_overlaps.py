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
######### INPUT ##################################
#~ comps_filename = 'data/final_comps_21.fits'
#~ comps_filename = 'data/final_comps_21_with_cra.fits'
#~ comps_filename = 'data/final_comps_8_CrA.fits'
#~ comps_filename = 'data/final_comps_CE_9G_not_converged_yet.fits'
comps_filename = 'data/final_comps_21_additional_comps.fits'

# Filename of data you want to compute overlaps for. Background overlaps are added later in this file!
gaia_filename = 'data/scocen_vac_DR2_distinct_XYZUVW.fits'

# Save output to this file. This is a copy of gaia_filename plus newly added memberships
#~ filename_output = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components.fits'
#~ filename_output = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_cra.fits'
#~ filename_output = 'data/scocen_vac_DR2_distinct_overlaps_with_cra_only.fits'
#~ filename_output = 'data/scocen_vac_DR2_distinct_overlaps_with_CE_only.fits'
filename_output = 'data/scocen_vac_DR2_distinct_overlaps_with_21_plus_additional_comps.fits'

##################################################
##################################################
print('Computing component overlaps for %s'%gaia_filename)
print('Output will be saved into %s'%filename_output)





### READ DATA ####################################

# Create components
comps = SphereComponent.load_raw_components(comps_filename)
print('Number of components: %d'%len(comps))

data_table = tabletool.read(gaia_filename)

# This table is masked. Unmask:
#~ data_table=data_table.filled() # TODO: fill with a stupid value!!!!

print('DATA READ', len(data_table))


#data_table.remove_column('background_log_overlap') # because we add it a few lines later


### Background overlaps ##########################
#~ bg_ols_filename = 'data/background_log_overlaps_gaia_DR2.fits'
bg_ols_filename = 'data/background_log_overlaps_scocen_DR2_updated_new_rvs.fits'
print('ADDING background overlaps from %s'%bg_ols_filename)
bg_ols = Table.read(bg_ols_filename)
#~ bg_ols.rename_column('source_id', 'dr2_source_id')
#~ data_table = join(data_table, bg_ols, keys='dr2_source_id', join_type='left')
data_table = join(data_table, bg_ols, keys='source_id', join_type='left')
mask = (data_table['background_log_overlap']<0)
if np.sum(np.logical_not(mask))>0:
    print('WARNING: %d stars do not have background overlaps.'%np.sum(np.logical_not(mask)))
else:
    print('ALL stars have background overlaps available!')
    
    
    
# This table is masked. Unmask:
data_table=data_table.filled(-999999) # TODO: fill with a stupid value!!!!
############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################
# Create data dict for real
historical = 'c_XU' in data_table.colnames
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True, # Must be set to True: This is to INSERT bg_ols into the Chronostar's internal data_table.
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
# Old memberships are updated here!
for i in range(membership_probabilities.shape[1]-1):
    comp_id = comps_fits[i]['comp_ID']
    comp_id = comp_id.replace('comp', '')
    data_table['membership%s' % comp_id] = membership_probabilities[:, i]
data_table['membership_bg'] = membership_probabilities[:, -1]

### SAVE DATA #####################################
data_table.write(filename_output, format='fits', overwrite=True)
print('%s written.'%filename_output)
print(data_table)
