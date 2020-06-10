"""
test_run_chronostar.py

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
from chronostar.traceorbit import trace_epicyclic_orbit

PY_VERS = sys.version[0]

# if len(sys.argv) != 2:
#     raise UserWarning('Incorrect usage. Path to parameter file is required'
#                       ' as a single command line argument. e.g.\n'
#                       '   > python run_chronostar.py path/to/parsfile.par')

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

    run_name = '3comps_and_background_w_orbits'

    logging.info(60 * '-')
    logging.info(15 * '-' + '{:^30}'.format('TEST: ' + run_name) + 15 * '-')
    logging.info(60 * '-')

    savedir = 'temp_data/{}_break_scipy_even_even_even_more_{}/'.format(PY_VERS, run_name)
    mkpath(savedir)
    data_filename = savedir + '{}_break_scipy_{}_data.fits'.format(PY_VERS,
                                                                 run_name)
    log_filename = savedir + 'log.log'.format(PY_VERS,
                                                              run_name)
    print('Saving in ', savedir)
    print('Logging in ', log_filename)

    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename=log_filename)
    logging.info('Beginning fit')

    # At
    # Want final result to be something like:
    # #  X,  Y, Z, U, V, W, dX, dV,  age,
    # [  0,  0, 0, 0, 0, 0,100., 2, 30.],
    # [200,  0, 0, 0, 0, 0, 50., 2, 15.],
    c1_age = 40
    c2_age = 15.
    c3_age =  1e-5
    c1_mean_now = np.array([0., 0., 0., 0., 0., 0.,])
    c2_mean_now = np.array([180., 0., 0., 0., 0., 0.,])
    # c3_mean_now = np.array([0., 0., 0., 0., 0., 0.,])

    c1_mean_then = trace_epicyclic_orbit(c1_mean_now, -c1_age)
    c2_mean_then = trace_epicyclic_orbit(c2_mean_now, -c2_age)
    c3_mean_then = np.array([50., 0., 0., 0., 0., 0.,])

    dx = 25.
    sphere_comp_pars = np.array([
        #  X,  Y, Z, U, V, W, dX, dV,  age,
        np.hstack((c1_mean_then, dx,   3., c1_age)),
        np.hstack((c2_mean_then, dx,   5., c2_age)),
        np.hstack((c3_mean_then, 100., 5., c3_age)),
    ])
    np.save('init_comps.npy', sphere_comp_pars)
    # init_comps = [SphereComponent(pars=pars) for pars in sphere_comp_pars]
    # SphereComponent.st
    starcounts = [50, 50, 200]
    ncomps = sphere_comp_pars.shape[0]
    nstars = np.sum(starcounts)

    background_density = 3e-10
    # background_density = 5e-11

    # initialise z appropriately such that all stars begin as members
    true_memb_probs = np.zeros((np.sum(starcounts), ncomps))
    start = 0
    for i in range(ncomps):
        true_memb_probs[start:start + starcounts[i], i] = 1.0
        start += starcounts[i]

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

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    ### SET UP PARAMETER FILE ###
    fit_pars = {
        'results_dir':savedir,
        'data_table':data_filename,
        'trace_orbit_func':trace_epicyclic_orbit,
        'return_results':True,
        'par_log_file':'fit_pars.log',
        'overwrite_prev_run':True,
        # 'nthreads':3,
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

    #TODO: Need to rethink this logic for ncomps > 1
    # Identify where calculated membership probs differ from true
    memb_discrepancy = true_memb_probs - np.round(memb_probs[:,perm])

    # Ensure we only count each misclassified star once
    # This is done by identifying the membership rows with any non-zero entries
    # Note that this approach will include background stars erroneously assigned
    # membership to components
    misclassified_stars = np.where(np.sum(np.abs(memb_discrepancy), axis=1))
    n_misclassified_stars = len(misclassified_stars[0])

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
        # TODO: tolerance
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


if __name__=='__main__':
    test_2comps_and_background()