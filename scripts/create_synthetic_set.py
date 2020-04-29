"""
Create a synthetic set for testing purposes.
"""

import logging
import numpy as np
import os.path
import sys
from distutils.dir_util import mkpath

sys.path.insert(0, '..')

from chronostar.naivefit import NaiveFit
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax

PY_VERS = sys.version[0]


"""
Synthesise a file with negligible error, retrieve initial
parameters

Takes a while... maybe this belongs in integration unit_tests

Performance of test is a bit tricky to callibrate. Since we are skipping
any temporal evolution for speed reasons, we model two
isotropic Gaussians. Now if these Gaussians are too far apart, NaiveFit
will gravitate to one of the Gaussians during the 1 component fit, and then
struggle to discover the second Gaussian.

If the Gaussians are too close, then both will be characteresied by the
1 component fit, and the BIC will decide two Gaussians components are
overkill.

I think I've addressed this by having the two groups have
large number of stars.
"""
using_bg = True

run_name = '3comps_and_background'

logging.info(60 * '-')
logging.info(15 * '-' + '{:^30}'.format('TEST: ' + run_name) + 15 * '-')
logging.info(60 * '-')

savedir = 'temp_data/{}_naive_{}/'.format(PY_VERS, run_name)
mkpath(savedir)
data_filename = savedir + '{}_naive_{}_data.fits'.format(PY_VERS,
                                                             run_name)
log_filename = 'temp_data/{}_naive_{}/log.log'.format(PY_VERS,
                                                          run_name)

logging.basicConfig(level=logging.INFO, filemode='w',
                    filename=log_filename)

### INITIALISE SYNTHETIC DATA ###

uniform_age = 1e-10
# Warning: if peaks are too far apart, it will be difficult for
# chronostar to identify the 2nd when moving from a 1-component
# to a 2-component fit.
#~ sphere_comp_pars = np.array([
    #~ #  X,  Y, Z, U, V, W, dX, dV,  age,
    #~ [  0,  0, 0, 0, 0, 0, 10., 5, uniform_age],
    #~ [ 30,  0, 0, 0, 5, 0, 10., 5, uniform_age],
#~ ])
sphere_comp_pars = np.array([
    #  X,  Y, Z, U, V, W, dX, dV,  age,
    [  0,  0, 0, 0, 0, 0, 10., 5, 10],
    [ 30,  0, 0, 0, 5, 0, 10., 5, 30],
    [ 10,  0, 0, 0, 2, 0, 10., 5, 20],
])
starcounts = [100, 150, 80]
ncomps = sphere_comp_pars.shape[0]
nstars = np.sum(starcounts)

background_density = 1e-9

# initialise z appropriately
true_memb_probs = np.zeros((np.sum(starcounts), ncomps))
start = 0
for i in range(ncomps):
    true_memb_probs[start:start + starcounts[i], i] = 1.0
    start += starcounts[i]

# # Initialise some random membership probablities
# # Normalising such that each row sums to 1
# init_memb_probs = np.random.rand(np.sum(starcounts), ncomps)
# init_memb_probs = (init_memb_probs.T / init_memb_probs.sum(axis=1)).T

try:
    data_dict = tabletool.build_data_dict_from_table(data_filename)
except:
    synth_data = SynthData(pars=sphere_comp_pars, starcounts=starcounts,
                           Components=SphereComponent,
                           background_density=background_density,
                           )
    synth_data.synthesise_everything()

    tabletool.convert_table_astro2cart(synth_data.table,
                                       write_table=True,
                                       filename=data_filename)

    background_count = len(synth_data.table) - np.sum(starcounts)
    # insert background densities
    synth_data.table['background_log_overlap'] =\
        len(synth_data.table) * [np.log(background_density)]

    synth_data.table.write(data_filename, overwrite=True)

