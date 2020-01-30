"""
MZ (2020 - 01 - 20)
Take files with results of the fit and put them in a fits file.
Some results from the naivefit are left in the npy format as they are
a supplementary results.

Columns:
object_id
membership_1
...
membership_N for N components
membership_bg.

Do the same for the components.
Apart from XYZUVW for the components,
add additional coordinates: (l, b, dist), and (ra, dec, dist)
"""
# TODO: This could later be incorporated into naivefit itself and be a default option

import numpy as np
from astropy.table import Table
from astropy import units as u
import string

import os
import sys
#sys.path.insert(0, '../')
sys.path.insert(0, '/home/marusa/chronostar/chronostar/') # TODO: hardcoded
import readparam

if len(sys.argv)<3:
    print('Usage:')
    print('python prepare_fits_file_with_results.py path_to_the_results_folder component_folder')
    print('E.g. python prepare_fits_file_with_results.py results/ 15/')

root = sys.argv[1]
root_comp = sys.argv[2]

filename_components = os.path.join(root, root_comp, 'final_comps.npy') # The best step of walkers. Lnprob can oscillate when converged.
filename_membership =  os.path.join(root, root_comp, 'final_membership.npy')
fit_pars = readparam.readParam(os.path.join(root, 'fit_pars.log'))
filename_raw_input_data = fit_pars['data_table'] # Star IDs


### COMPONENTS
# Components (EXcludes background, but background must be 1-sum(memberships for all components))
#TODO: It would be great to add background membership as well, because then it is much more simple (using one column) to determine stars that are members of any of the components as opposed to the background.
comp = np.load(filename_components)
ncomps = len(comp)

# Compnames
abc=string.ascii_uppercase
compnames = [abc[i] for i in range(ncomps)]

tabcomps = Table([compnames, comp[:,0], comp[:,1], comp[:,2], comp[:,3], comp[:,4], comp[:,5], comp[:,6], comp[:,7], comp[:,8]], names=('comp_ID', 'X', 'Y', 'Z', 'U', 'V', 'W', 'dX', 'dV', 'Age'))
tabcomps['X'].unit = u.pc
tabcomps['Y'].unit = u.pc
tabcomps['Z'].unit = u.pc
tabcomps['dX'].unit = u.pc
tabcomps['U'].unit = u.km/u.s
tabcomps['V'].unit = u.km/u.s
tabcomps['W'].unit = u.km/u.s
tabcomps['dV'].unit = u.km/u.s
tabcomps['Age'].unit = u.Myr

tabcomps.write('final_comps.fits', format='fits')

### MEMBERSHIPS
# Raw data with IDs
tab_raw = Table.read(filename_raw_input_data)
ids = list(tab_raw['source_id'])
tab = Table((ids,), names=('source_id',))

# Membership
memb = np.load(filename_membership)
for i, c in enumerate(compnames):
    tab['membership%s'%c.replace('comp', '')] = memb[:,i]

tab.write('final_membership.fits', format='fits')

# TODO: what is this
#final_comps_bak.npy # get some pickle error
#final_med_and_spans.npy # related to components: these are actual estimates from the walkers with uncertainties!