"""
MZ: Test build_data_dict_from_table: see how covariance matrices are built

"""

import numpy as np
import sys
sys.path.insert(0, '/Users/marusa/chronostar/')
from chronostar import tabletool

gaia_filename = 'example2_XYZUVW.fits'
#~ gaia_filename = '/Users/marusa/chronostar/cfastfit/example2_XYZUVW.dat'

data_table = tabletool.read(gaia_filename)

historical = 'c_XU' in data_table.colnames
data_dict = tabletool.build_data_dict_from_table_marusa_testing(
        data_table,
        get_background_overlaps=False, # Must be set to True: This is to INSERT bg_ols into the Chronostar's internal data_table.
        historical=historical,
)
