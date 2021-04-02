"""
Find stars that need new background overlaps, compute them separately,
and add them to the table here with the keyword 'background_log_overlap'.
"""

import numpy as np
from astropy.table import Table, join

table_with_bg_overlaps_filename = 'data/background_log_overlaps_gaia_DR2.fits'

# This table includes source_id from EDR3, and 'dr2_source_id' from DR2
gaia_table_filename = 'data/scocen_vac_EDR3.fits'


bgols = Table.read(table_with_bg_overlaps_filename)
tab = Table.read(gaia_table_filename)


bgols.rename_column('source_id', 'dr2_source_id')
tab = join(tab, bgols, keys='dr2_source_id', join_type='left')

# How many stars are missing bgols
tab=tab.filled(10000000000)
mask = tab['background_log_overlap']<0
print(np.sum(mask))
print(np.sum(~mask))


tab[mask].write('data/scocen_vac_EDR3_SUBTABLE_with_bgols_need_to_update_bg_ols.fits', overwrite=True)
