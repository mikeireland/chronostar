"""
Check what component PDS 70 belongs to.
"""

import numpy as np
from astropy.table import Table, join, unique
from astropy import units as u
import string
import matplotlib.pyplot as plt
plt.ion()

import sys
sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
import coordinate

d = Table.read('members_from_5k_fit_and_overlaps.fits')

Gmag = d['phot_g_mean_mag'] - 5 * np.log10(1.0 / (d['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
d['Gmag'] = Gmag

# Read components (WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present!)
comps = Table.read('final_comps.fits')
comps.rename_column('age', 'Age')

# Crossing time. Only components with sigma<age have reliable ages.
crossing_time = comps['dX']/comps['dV'] * 0.977813106 # pc/km*s to Myr
comps['Crossing_time'] = crossing_time
mask = crossing_time < comps['Age'] # sigma < age
comps['Age_reliable'] = mask


source_id_pds70 = 6110141563309613056

mask = d['source_id'] == source_id_pds70

print(d[mask])

abc=string.ascii_uppercase
compnames = ['membership%s'%abc[i] for i in range(len(comps))]

component_id=''
for c in compnames:
    prob = d[mask][c]
    if prob>0.5:
        print(c, prob)
        component_id=c # Assumes that only one component has prob>0.5

mask = comps['Name']==component_id.replace('membership', '')
comp_pds70 = comps[mask]
print(comp_pds70)
