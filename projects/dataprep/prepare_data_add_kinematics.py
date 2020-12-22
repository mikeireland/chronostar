"""
Add very large RV errors for stars with no known RVs.
Convert to cartesian.
"""

import numpy as np
import sys
sys.path.insert(0, '..')
from chronostar import tabletool
from astropy.table import Table

datafile = '../data/ScoCen_box_result.fits'
d = tabletool.read(datafile)

# Set missing radial velocities (nan) to 0
d['radial_velocity'] = np.nan_to_num(d['radial_velocity'])

# Set missing radial velocity errors (nan) to 1e+10
d['radial_velocity_error'][np.isnan(d['radial_velocity_error'])] = 1e+4

print('Convert to cartesian')
tabletool.convert_table_astro2cart(table=d, return_table=True)

d.write('/priv/mulga1/marusa/chronostar/data/ScoCen_box_result_15M_ready_for_bg_ols.fits')
print('Cartesian written.', len(d))