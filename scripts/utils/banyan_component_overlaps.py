"""
Take components fitted to 6D data and make overlaps
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
from chronostar import likelihood
from astropy.table import Table, vstack, join


####### Input data ##############################
banyan_filename = 'banyan_sigma_parameters.fits'

# List of Banyan associations you want to include. Note that these names
# *MUST* match those from the Banyan table.
list_of_banyan_associations = ['BPMG  ', 'HYA   '] # Must include spaces!

# TODO: Check if BANYAN DATA IS IN LSR...

# Filename with stellar data in XYZUVW space
data_filename = 'beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits' # This is Gaia DR2

output_filename = 'banyan_component_overlaps.fits'
#################################################

banyan = Table.read(banyan_filename)
banyan_dict = dict(zip(banyan['NAME'], banyan))

data_table = tabletool.read(data_filename)

# This table is masked (missing values in the table are masked). Unmask:
data_table=data_table.filled()

############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################

historical = 'c_XU' in data_table.colnames
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        historical=historical,
)


# This part takes in a component at time TODAY. Everything needs to be in the XYZUVW space
"""
cov_now: component covariance matrix today
mean_now: component coordinates today in XYZUVW space
star_covs: a list of covariance matrices for stars today
star_means: a list of mean positions for stars today
star_count: how many stars are in this list
"""
star_means = data_dict['means']
star_covs = data_dict['covs']
star_count = len(star_means)

#~ tab = Table((data_table['source_id']), names=('source_id', ))
tab = Table((data_table['main_designation'],), names=('source_id', )) # FIX THIS

for assoc in list_of_banyan_associations:
    b = banyan_dict[assoc]
    
    cov_now = b['COVARIANCE_MATRIX']
    mean_now = b['CENTER_VEC']

    #~ lnols = likelihood.c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means, star_count) # This doesn't work...
    
    # log overlaps
    lnols = likelihood.slow_get_lnoverlaps(cov_now, mean_now, star_covs, star_means)

    # Save this somewhere...
    tab[assoc] = lnols


# Save results
tab.write(output_filename, format='fits', overwrite=True)
print('%s written.'%output_filename)
