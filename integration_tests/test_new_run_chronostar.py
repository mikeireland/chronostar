"""
test_new_run_chronsotar.py

Integration test, testing some simple scenarios for NaiveFit
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

# if len(sys.argv) != 2:
#     raise UserWarning('Incorrect usage. Path to parameter file is required'
#                       ' as a single command line argument. e.g.\n'
#                       '   > python new_run_chronostar.py path/to/parsfile.par')

# fit_par_file = sys.argv[1]

# if not os.path.isfile(fit_par_file):
#     raise UserWarning('Provided file does not exist')

def dummy_trace_orbit_func(loc, times=None):
    """Dummy trace orbit func to skip irrelevant computation"""
    if times is not None:
        if np.all(times > 1.0):
            return loc + 1000.
    return loc

def test_2comps_and_background():
    """
     Synthesise a file with negligible error, retrieve initial
     parameters

     Takes a while... maybe this belongs in integration unit_tests
     """

    run_name = '2comps_and_background'

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
    sphere_comp_pars = np.array([
        #   X,  Y,  Z, U, V, W, dX, dV,  age,
        [-20, -20,  0, 0, 0, 0, 2., 2, uniform_age],
        [ 20,  20,  0, 0, 0, 0, 2., 2, uniform_age],
    ])
    starcounts = [20, 50]
    ncomps = sphere_comp_pars.shape[0]

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

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    ### SET UP PARAMETER FILE ###
    fit_pars = {
        'results_dir':savedir,
        'data_table':data_filename,
        'trace_orbit_func':'dummy_trace_orbit_func',
        'return_results':True,
        'par_log_file':savedir + 'fit_pars.log',
        'overwrite_prev_run':True,
    }

    ### INITIALISE AND RUN A NAIVE FIT ###
    naivefit = NaiveFit(fit_pars=fit_pars)
    naivefit.setup()
    result = naivefit.run_fit()
    best_comps, med_and_spans, memb_probs, lnlike, lnpost, bic = result

    ### CHECK RESULT ###
    # No guarantee of order, so check if result is permutated
    perm = expectmax.get_best_permutation(memb_probs, true_memb_probs)

    logging.info('Best permutation is: {}'.format(perm))

    assert np.allclose(true_memb_probs, memb_probs[:, perm])

    for origin, best_comp in zip(origins, np.array(best_comps)[perm,]):
        assert (isinstance(origin, SphereComponent) and
                isinstance(best_comp, SphereComponent))
        o_pars = origin.get_pars()
        b_pars = best_comp.get_pars()

        logging.info("origin pars:   {}".format(o_pars))
        logging.info("best fit pars: {}".format(b_pars))
        assert np.allclose(origin.get_mean(),
                           best_comp.get_mean(),
                           atol=5.)
        assert np.allclose(origin.get_sphere_dx(),
                           best_comp.get_sphere_dx(),
                           atol=2.)
        assert np.allclose(origin.get_sphere_dv(),
                           best_comp.get_sphere_dv(),
                           atol=2.)
        assert np.allclose(origin.get_age(),
                           best_comp.get_age(),
                           atol=1.)


