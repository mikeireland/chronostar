"""
Create a synthetic dataset
"""

import logging
import numpy as np
import os
import sys

sys.path.insert(0, '..')

from chronostar.synthdata import SynthData
from chronostar.component import EllipComponent as Component
from chronostar import tabletool

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

run_name = 'ellip_synt'

savedir = 'ellipsyn/{}_naive_{}/'.format(PY_VERS, run_name)
os.makedirs(savedir)
filename_output = savedir + '{}_naive_{}_data.fits'.format(PY_VERS,
                                                             run_name)

### INITIALISE SYNTHETIC DATA ###

uniform_age = 1e-10
# Warning: if peaks are too far apart, it will be difficult for
# chronostar to identify the 2nd when moving from a 1-component
# to a 2-component fit.
comp_pars = np.array([
    # X, Y, Z, U, V, W, dX, dU, dV, quat, quat, quat, quat, corr, age,
    [  0,  0, 0, 0, 0, 0, 10., 5, 7, 0.2, 0.1, 0.7, 0.4, 0.7, 10],
    #[  30,  0, 0, 0, 0, 0, 10., 15, 5, 7, 0.2, 0.1, 0.7, 0.4, 0.3, 20],
    #[  15,  20, 0, 0, 0, 0, 10., 15, 5, 7, 0.2, 0.1, 0.7, 0.4, 0.3, 50],
])

print(comp_pars.shape)
starcounts = [200] #[100, 150, 300]
ncomps = comp_pars.shape[0]
nstars = np.sum(starcounts)

background_density = 1e-9
#~ background_density = 1e-7 # 50 bg stars
#~ background_density = 1e-12

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

print('Start creating data...')
synth_data = SynthData(pars=comp_pars, starcounts=starcounts,
                       Components=Component,
                       background_density=background_density,
                       #span_factor=5
                       )
print('Init done...')
#~ print(synth_data)
synth_data.synthesise_everything()
print('Synthesisation done...')
tabletool.convert_table_astro2cart(synth_data.table,
                                   write_table=True,
                                   filename=filename_output)

background_count = len(synth_data.table) - np.sum(starcounts)
print('starcounts', starcounts)
print('background_count', background_count)
# insert background densities
synth_data.table['background_log_overlap'] =\
    len(synth_data.table) * [np.log(background_density)]

synth_data.table.write(filename_output, overwrite=True)
