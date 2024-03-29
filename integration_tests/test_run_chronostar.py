"""
test_run_chronostar.py

Integration test, testing some simple scenarios for NaiveFit
"""

import logging
import numpy as np
import sys
from distutils.dir_util import mkpath

sys.path.insert(0, '..')

from chronostar.naivefit import NaiveFit
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax

PY_VERS = sys.version[0]


def dummy_trace_orbit_func(loc, times=None):
    """
    Integrating orbits takes a long time. So we can run this test quickly,
    we enforce the age of all components to be ~0, and then use this
    "dummy" trace orbit function.

    This function doesn't do anything, and it definitely should not be
    used in any actual comuptation. It is merely a place holder to
    skip irrelevant computation when running integration tests.
    """
    if times is not None:
        if np.all(times > 1.0):
            return loc + 1000.
    return loc

def test_2comps_and_background():
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

    # DON'T CHANGE THE AGE! BECAUSE THIS TEST DOESN'T USE ANY ORBIT INTEGRATION!!!
    # Note: if peaks are too far apart, it will be difficult for
    # chronostar to identify the 2nd when moving from a 1-component
    # to a 2-component fit.
    uniform_age = 1e-10
    sphere_comp_pars = np.array([
        #  X,  Y, Z, U, V, W, dX, dV,  age,
        [  0,  0, 0, 0, 0, 0, 5., 2, uniform_age],
        [ 30,  0, 0, 0, 5, 0, 5., 2, uniform_age],
    ])
    starcounts = [100, 150]
    ncomps = sphere_comp_pars.shape[0]
    nstars = np.sum(starcounts)

    background_density = 1e-9

    # initialise z appropriately
    true_memb_probs = np.zeros((np.sum(starcounts), ncomps))
    start = 0
    for i in range(ncomps):
        true_memb_probs[start:start + starcounts[i], i] = 1.0
        start += starcounts[i]

    try:
        # Check if the synth data has already been constructed
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

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    ### SET UP PARAMETER FILE ###
    fit_pars = {
        'results_dir':savedir,
        'data_table':data_filename,
        'trace_orbit_func':'dummy_trace_orbit_func',
        'return_results':True,
        'par_log_file':'fit_pars.log',
        'overwrite_prev_run':True,
        # 'nthreads':18,
        'nthreads':3,
    }

    ### INITIALISE AND RUN A NAIVE FIT ###
    naivefit = NaiveFit(fit_pars=fit_pars)
    result, score = naivefit.run_fit()

    best_comps = result['comps']
    memb_probs = result['memb_probs']

    # Check membership has ncomps + 1 (bg) columns
    n_fitted_comps = memb_probs.shape[-1] - 1
    assert ncomps == n_fitted_comps

    ### CHECK RESULT ###
    # No guarantee of order, so check if result is permutated
    #  also we drop the bg memberships for permutation reasons
    perm = expectmax.get_best_permutation(memb_probs[:nstars,:ncomps], true_memb_probs)

    memb_probs = memb_probs[:nstars]

    logging.info('Best permutation is: {}'.format(perm))

    n_misclassified_stars = np.sum(np.abs(true_memb_probs - np.round(memb_probs[:,perm])))

    # Check fewer than 15% of association stars are misclassified
    try:
        assert n_misclassified_stars / nstars * 100 < 15
    except AssertionError:
        import pdb; pdb.set_trace()

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
                           atol=2.5)
        assert np.allclose(origin.get_sphere_dv(),
                           best_comp.get_sphere_dv(),
                           atol=2.5)
        assert np.allclose(origin.get_age(),
                           best_comp.get_age(),
                           atol=1.)

if __name__ == '__main__':
    test_2comps_and_background()
