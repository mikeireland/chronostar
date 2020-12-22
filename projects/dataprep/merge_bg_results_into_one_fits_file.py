"""
Merge results from the bg_ols_multiprocessing into one fits file.
"""

from __future__ import print_function, division

import numpy as np
from astropy.table import Table, vstack
import glob

#~ datafile = 'solar_neighbourhood_determine_bg_ols_for_these_stars.fits'
datafile = 'last_200k_to_compute_background_overlaps.fits'
d = Table.read(datafile)
N=2 # that many chunks in total
    
#~ filenames = glob.glob('*securecopy.dat')
filenames = glob.glob('*round3*')

for f in filenames:
    n = int(f.split('_')[-2])
    print(f, n)

    NI=n # take this chunk #TODO: update this number for every run!

    # TAKE ONLY the i-th part of the data
    indices_chunks = np.array_split(range(len(d)), N)
    d_tmp=d[indices_chunks[NI]]

    data = np.loadtxt(f)
    print(len(data), len(d_tmp))

    tab_tmp = Table((d_tmp['source_id'], data), names=('source_id', 'background_log_overlap'))

    try:
        tab = vstack((tab, tab_tmp))
    except:
        tab=tab_tmp

print(tab)
#~ tab.write('solar_neighbourhood_determine_bg_ols_for_these_stars_bgols_computed.fits', format='fits')
tab.write('last_200k_to_compute_background_overlaps_bgols_computed.fits', format='fits')
