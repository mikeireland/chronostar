#! /usr/bin/env python
"""Script to run an exhaustive analysis of synthetic fits and store
their results in a hierarchical directory structure.
"""

import logging
import numpy as np
import sys
from distutils.dir_util import mkpath
import pdb

sys.path.insert(0, '..')

import chronostar.investigator as iv

SAVE_DIR = '../results/synth_results/'
NTIMES = 21
NFIXED_FITS = 11

#precs = ['perf']#, 'gaia']
precs = ['gaia']
ages = [10, 20]
spreads = [5]#, 10]
v_disps = [2]#, 5]
sizes   = [50]#, 100]

base_group_pars = [
    -80, 80, 50, 10, -20, -5, None, None, None, None,
    0.0, 0.0, 0.0, None, None
]

prec_val = {'perf':1e-5, 'gaia':1.0}

def do_something(age,spread,v_disp,size,prec):
    path_name = SAVE_DIR + "{}_{}_{}_{}_{}/".format(
        age, spread, v_disp, size, prec
    )
    logging.basicConfig(
        filename=path_name + 'investigator_demo.log',
        level=logging.DEBUG, filemode='w'
    )
    group_pars = list(base_group_pars)
    group_pars[6:9] = [spread, spread, spread]
    group_pars[9] = v_disp
    group_pars[13] = age
    group_pars[14] = size

    times = np.linspace(0, 2*age, NTIMES)

    mkpath(path_name)
    sf = iv.SynthFit(init_group_pars=group_pars, save_dir=path_name,
                times=times, nfixed_fits=NFIXED_FITS)
    sf.investigate()

if __name__ == '__main__':
    [do_something(age, spread, v_disp, size, prec)
     for age in ages
     for spread in spreads
     for v_disp in v_disps
     for size in sizes
     for prec in precs
     ]
