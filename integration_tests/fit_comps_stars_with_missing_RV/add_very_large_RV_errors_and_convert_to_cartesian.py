"""
Add very large RV errors for stars with no known RVs.
Convert to cartesian.
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from chronostar import tabletool
from astropy.table import Table
import random

# Percentage of RV you want to be missing in the data
missingrv = 0.7

# beta Pic file that Tim used in the paper. Now spoil radial velocities here!
datafile = 'data/beta_Pictoris_with_gaia_small_everything_final.fits'
d = tabletool.read(datafile)

# Take care of nans
# Set missing radial velocities (nan) to 0 (8 stars in the bPic data which is nothing)
d['radial_velocity_best'] = np.nan_to_num(d['radial_velocity_best'])

# Set missing radial velocity errors (nan) to 1e+10
d['radial_velocity_error_best'][np.isnan(d['radial_velocity_error_best'])] = 1e+4

d['radial_velocity'] = d['radial_velocity_best']
d['radial_velocity_error'] = d['radial_velocity_error_best']
d.remove_column('radial_velocity_best')
d.remove_column('radial_velocity_error_best')
d.rename_column('gaia_dr2', 'source_id')

# Randomly select stars and break their RVs
selected = random.sample(d['source_id'], int(len(d)*missingrv))
mask = np.in1d(d['source_id'], selected)
d['radial_velocity'][mask] = 0
d['radial_velocity_error'][mask] = 1e+4


print('Convert to cartesian')
tabletool.convert_table_astro2cart(table=d, return_table=True)

d.write('data/data_with_%d_percent_missing_rvs.fits'%(missingrv*100), overwrite=True)
print('Cartesian written.', len(d))
