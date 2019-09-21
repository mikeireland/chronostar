#!/usr/bin/env python -W ignore
"""
test_expectmax
-----------------------------

Tests for `expectmax` module
"""
from __future__ import division, print_function

import logging
import numpy as np
import sys
from distutils.dir_util import mkpath


sys.path.insert(0, '..')  # hacky way to get access to module
from chronostar.component import SphereComponent
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import expectmax

PY_VERS = sys.version[0]

def dummy_trace_orbit_func(loc, times=None):
    """Dummy trace orbit func to skip irrelevant computation"""
    if times is not None:
        if np.all(times > 1.0):
            return loc + 1000.
    return loc

SPHERE_COMP_PARS = np.array([
    # X, Y, Z, U, V, W, dX, dV, age,nstars
    [ 0, 0, 0, 0, 0, 0, 10,  5,  10],
    [50,50, 0, 0, 0, 0, 10,  5,  10],
])
STARCOUNTS = [200, 200]

# run_name = 'stationary'
# savedir = 'temp_data/{}_expectmax_{}/'.format(PY_VERS, run_name)
# mkpath(savedir)
# data_filename = savedir + '{}_expectmax_{}_data.fits'.format(PY_VERS,
#                                                             run_name)
# log_filename = 'logs/{}_expectmax_all_tests.log'.format(PY_VERS)
# plot_dir = 'temp_plots/{}_expectmax_{}'.format(PY_VERS, run_name)
# print('logging in {}'.format(log_filename))
# logging.basicConfig(level=logging.INFO, filemode='w',
                    # filename=log_filename)


def test_execution_simple_fit():
    """
    Don't test for correctness, but check that everything actually executes
    """
    run_name = 'quickdirty'
    logging.info(50*'-')
    logging.info(15*'-' + '{:^20}'.format(run_name) + 15*'-')
    logging.info(50*'-')

    savedir = 'temp_data/{}_expectmax_{}/'.format(PY_VERS, run_name)
    mkpath(savedir)
    data_filename = savedir + '{}_expectmax_{}_data.fits'.format(PY_VERS,
                                                                 run_name)
    log_filename = 'temp_data/{}_expectmax_{}/log.log'.format(PY_VERS,
                                                              run_name)
    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename=log_filename)

    uniform_age = 1e-10
    sphere_comp_pars = np.array([
        # X, Y, Z, U, V, W, dX, dV,  age,
        [ 0, 0, 0, 0, 0, 0, 10.,  5, uniform_age],
    ])
    starcount = 100

    background_density = 1e-9

    ncomps = sphere_comp_pars.shape[0]

    # true_memb_probs = np.zeros((starcount, ncomps))
    # true_memb_probs[:,0] = 1.

    synth_data = SynthData(pars=sphere_comp_pars, starcounts=[starcount],
                           Components=SphereComponent,
                           background_density=background_density,
                           )
    synth_data.synthesise_everything()

    tabletool.convert_table_astro2cart(synth_data.table,
                                       write_table=True,
                                       filename=data_filename)
    background_count = len(synth_data.table) - starcount

    # insert background densities
    synth_data.table['background_log_overlap'] =\
        len(synth_data.table) * [np.log(background_density)]

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    best_comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(data=synth_data.table,
                                 ncomps=ncomps,
                                 rdir=savedir,
                                 trace_orbit_func=dummy_trace_orbit_func,
                                 use_background=True,
                                 burnin=10,
                                 sampling_steps=10,

                                 )


def test_fit_one_comp_with_background():
    """
    Synthesise a file with negligible error, retrieve initial
    parameters

    Takes a while... maybe this belongs in integration unit_tests
    """
    run_name = 'background'

    logging.info(50*'-')
    logging.info(15*'-' + '{:^20}'.format(run_name) + 15*'-')
    logging.info(50*'-')

    savedir = 'temp_data/{}_expectmax_{}/'.format(PY_VERS, run_name)
    mkpath(savedir)
    data_filename = savedir + '{}_expectmax_{}_data.fits'.format(PY_VERS,
                                                                 run_name)
    log_filename = 'temp_data/{}_expectmax_{}/log.log'.format(PY_VERS,
                                                              run_name)

    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename=log_filename)
    uniform_age = 1e-10
    sphere_comp_pars = np.array([
        # X, Y, Z, U, V, W, dX, dV,  age,
        [ 0, 0, 0, 0, 0, 0, 10.,  5, uniform_age],
    ])
    starcount = 200

    background_density = 1e-9

    ncomps = sphere_comp_pars.shape[0]

    # true_memb_probs = np.zeros((starcount, ncomps))
    # true_memb_probs[:,0] = 1.

    synth_data = SynthData(pars=sphere_comp_pars, starcounts=[starcount],
                           Components=SphereComponent,
                           background_density=background_density,
                           )
    synth_data.synthesise_everything()

    tabletool.convert_table_astro2cart(synth_data.table,
                                       write_table=True,
                                       filename=data_filename)
    background_count = len(synth_data.table) - starcount

    # insert background densities
    synth_data.table['background_log_overlap'] =\
        len(synth_data.table) * [np.log(background_density)]

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    best_comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(data=synth_data.table,
                                 ncomps=ncomps,
                                 rdir=savedir,
                                 trace_orbit_func=dummy_trace_orbit_func,
                                 burnin=500,
                                 sampling_steps=5000,
                                 use_background=True)

    # return best_comps, med_and_spans, memb_probs

    # Check parameters are close
    assert np.allclose(sphere_comp_pars, best_comps[0].get_pars(),
                       atol=1.5)

    # Check most assoc members are correctly classified
    recovery_count_threshold = 0.95 * starcount
    recovery_count_actual =  np.sum(memb_probs[:starcount,0] > 0.5)
    assert recovery_count_threshold < recovery_count_actual

    # Check most background stars are correctly classified
    # Number of bg stars classified as members should be less than 5%
    # of all background stars
    contamination_count_threshold = 0.05 * len(memb_probs[starcount:])
    contamination_count_actual = np.sum(memb_probs[starcount:, 0] > 0.5)
    assert contamination_count_threshold > contamination_count_actual

    # Check reported membership probabilities are consistent with recovery
    # rate (within 5%)
    mean_membership_confidence = np.mean(memb_probs[:starcount,0])
    assert np.isclose(recovery_count_actual/starcount., mean_membership_confidence,
                      atol=0.05)


def test_fit_many_comps():
    """
    Synthesise a file with negligible error, retrieve initial
    parameters

    Takes a while... maybe this belongs in integration unit_tests
    """

    run_name = 'stationary'

    logging.info(50*'-')
    logging.info(15*'-' + '{:^20}'.format(run_name) + 15*'-')
    logging.info(50*'-')

    savedir = 'temp_data/{}_expectmax_{}/'.format(PY_VERS, run_name)
    mkpath(savedir)
    data_filename = savedir + '{}_expectmax_{}_data.fits'.format(PY_VERS,
                                                                 run_name)
    log_filename = 'temp_data/{}_expectmax_{}/log.log'.format(PY_VERS,
                                                              run_name)

    logging.basicConfig(level=logging.INFO, filemode='w',
                        filename=log_filename)
    uniform_age = 1e-10
    sphere_comp_pars = np.array([
        #  X,  Y,  Z, U, V, W, dX, dV,  age,
        [-50,-50,-50, 0, 0, 0, 10.,  5, uniform_age],
        [ 50, 50, 50, 0, 0, 0, 10.,  5, uniform_age],
    ])
    starcounts = [200,200]
    ncomps = sphere_comp_pars.shape[0]

    # initialise z appropriately
    # start = 0
    # for i in range(ngroups):
    #     nstars_in_group = int(group_pars[i,-1])
    #     z[start:start+nstars_in_group,i] = 1.0
    #     start += nstars_in_group

    true_memb_probs = np.zeros((np.sum(starcounts), ncomps))
    true_memb_probs[:200,0] = 1.
    true_memb_probs[200:,1] = 1.

    # Initialise some random membership probablities
    # Normalising such that each row sums to 1
    init_memb_probs = np.random.rand(np.sum(starcounts), ncomps)
    init_memb_probs = (init_memb_probs.T / init_memb_probs.sum(axis=1)).T

    synth_data = SynthData(pars=sphere_comp_pars, starcounts=starcounts,
                           Components=SphereComponent,
                           )
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table,
                                       write_table=True,
                                       filename=data_filename)

    origins = [SphereComponent(pars) for pars in sphere_comp_pars]

    best_comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(data=synth_data.table,
                                 ncomps=ncomps,
                                 rdir=savedir,
                                 init_memb_probs=init_memb_probs,
                                 trace_orbit_func=dummy_trace_orbit_func, )

    # compare fit with input
    try:
        assert np.allclose(true_memb_probs, memb_probs)
    except AssertionError:
        # If not close, check if flipping component order fixes things
        memb_probs = memb_probs[:,::-1]
        best_comps = best_comps[::-1]
        assert np.allclose(true_memb_probs, memb_probs)
    for origin, best_comp in zip(origins, best_comps):
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



"""
def test_expectation(self):
    ngroups = self.groups_pars_ex.shape[0]
    nstars = np.sum(self.groups_pars_ex[:, -1])

    groups_pars_in = utils.internalise_multi_pars(self.groups_pars_ex)

    # neligible error - smaller vals lead to problems with matrix inversions
    error = 1e-5
    ntimes = 20

    tb_file = "tmp_expectmax_tb_file.pkl"

    # to save time, check if tb_file is already created
    try:
        with open(tb_file):
            pass
    # if not created, then create it. Careful though! May not be the same
    # as group_pars. So if test fails try deleting tb_file from
    # directory
    except IOError:
        # generate synthetic data
        syn.synthesise_data(
            ngroups, self.groups_pars_ex, error, savefile=self.synth_file
        )
        with open(self.synth_file, 'r') as fp:
            t = pickle.load(fp)

        max_age = np.max(groups_pars_in[:, -1])
        times = np.linspace(0, 2 * max_age, ntimes)
        tb.traceback(t, times, savefile=tb_file)

    star_pars = gf.read_stars(tb_file)

    z = em.expectation(star_pars, groups_pars_in)

    # check membership list totals to nstars in group
    self.assertTrue(np.isclose(np.sum(z), nstars))
    self.assertTrue(np.allclose(np.sum(z, axis=1), 1.0))
    self.assertTrue(
        np.allclose(np.sum(z, axis=0), self.groups_pars_ex[:,-1], atol=0.1)
    )

    nstars1 = int(self.groups_pars_ex[0,-1])
    nstars2 = int(self.groups_pars_ex[1,-1])
    self.assertTrue( (z[:nstars1,0] > z[:nstars1,1]).all() )
    self.assertTrue( (z[nstars1:,0] < z[nstars1:,1]).all() )

"""

if __name__ == '__main__':
    res = test_fit_one_comp_with_background()
