from __future__ import print_function, division, unicode_literals

import logging
import numpy as np
import os
import sys

sys.path.insert(0, '..')
from chronostar.component import EllipComponent
from chronostar.component import SphereComponent
from chronostar.synthdata import SynthData
from chronostar.traceorbit import trace_cartesian_orbit
from chronostar.traceorbit import trace_epicyclic_orbit
from chronostar import tabletool
import chronostar.compfitter as cf

import matplotlib.pyplot as plt

PY_VERS = sys.version[0]

def plot_results(true_comp, best_fit_comp, star_pars, plt_dir=''):

    labels = 'XYZUVW'
    # labels = ['xi', 'eta', 'zeta', 'xi dot', 'eta dot', 'zeta dot']
    units = 3*['pc'] + 3*['km/s']
    # units = ['units'] * 6

    # <--!!! Choose which cartesian dimensions you wish to plot !!!--> #
    # <--!!! 0 -> X, 1-> Y, 2 -> Z, 3 -> U etc.                 !!!--> #
    dims = [(0,1), (3,4), (0,3), (1,4)]

    figsize = 10
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize, figsize))
    fig.set_tight_layout(True)

    for ax, (dim1, dim2) in zip(axes.flatten(), dims):
        ax.tick_params(direction='in')
        print(dim1, dim2)

        ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
        ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

        means = star_pars['means']

        ax.scatter(means[:, dim1], means[:, dim2])
        true_comp.plot(dim1, dim2, comp_now=True, comp_then=True, color='blue', ax=ax,
                       comp_orbit=True, orbit_color='blue')
        best_fit_comp.plot(dim1, dim2, comp_now=True, comp_then=True, color='red', ax=ax,
                           comp_orbit=True, orbit_color='red')

    fig.savefig(plt_dir+'ellip_comps.pdf')

    return

def dummy_trace_orbit_func(loc, times=None):
    """Dummy trace orbit func to skip irrelevant computation"""
    if times is not None:
        if np.all(times > 1.0):
            return loc + 10.
    return loc

def run_fit_helper(true_comp, starcounts, measurement_error,
                   burnin_step=None,
                   run_name='default',
                   trace_orbit_func=None,
                   Component=EllipComponent,
                   init_pars=None
                   ):
    py_vers = sys.version[0]
    save_dir = 'temp_data/%s_compfitter_%s/'%(py_vers, run_name)
    data_filename = save_dir + 'synth_data.fits'.format(py_vers, run_name)
    plot_dir = save_dir
    print("---------", save_dir)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    log_filename = save_dir + 'log.log'.format(py_vers, run_name)

    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w')

    synth_data = SynthData(pars=true_comp.get_pars(),
                           starcounts=starcounts,
                           measurement_error=measurement_error,
                           Components=Component)

    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table,
                                       write_table=True,
                                       filename=data_filename)

    print("newPars ------------------------------ \n",init_pars)
    if init_pars is None:
        internal_pars = None
    else:
        internal_pars = Component.internalise(init_pars)

    res = cf.fit_comp(
            data=synth_data.table,
            plot_it=True,
            burnin_steps=burnin_step,
            store_burnin_chains=True,
            plot_dir=plot_dir,
            save_dir=save_dir,
            trace_orbit_func=trace_orbit_func,
            optimisation_method='emcee',
            Component=Component,
            init_pars=internal_pars
    )

    comps_filename = save_dir + 'true_and_best_comp.py'
    best_comp = res[0]
    EllipComponent.store_raw_components(comps_filename, [true_comp, best_comp])

    star_pars = tabletool.build_data_dict_from_table(synth_data.table)
    plot_results(true_comp, best_fit_comp=res[0], star_pars=star_pars,
                 plt_dir=save_dir)

    return res

def test_stationary_comp():
    """
    Tests an ellip comp fit with no traceforward, i.e. no evolution of stars or
    component
    """
    # log_filename = 'logs/compfitter_stationary.log'
    # synth_data_savefile = 'temp_data/compfitter_stationary_synthdata.fits'

    burnin_step = 1000

    true_comp_mean = np.zeros(6)
    true_comp_dx = 10.
    true_comp_dy = 5.
    true_comp_du = 2.
    true_comp_dv = 4.
    true_comp_roll  = 0.
    true_comp_pitch = 0.
    true_comp_yaw   = 0.
    true_comp_cov_xv = 0.4
    true_comp_age = 1e-10

    true_comp_pars = np.hstack([
            true_comp_mean,
            true_comp_dx,
            true_comp_dy,
            true_comp_du,
            true_comp_dv,
            true_comp_roll,
            true_comp_pitch,
            true_comp_yaw,
            true_comp_cov_xv,
            true_comp_age,
    ])

    true_comp = EllipComponent(true_comp_pars)

    nstars = 100
    measurement_error = 1e-10

    best_comp, chain, lnprob = run_fit_helper(
            true_comp=true_comp, starcounts=nstars,
            measurement_error=measurement_error,
            run_name='stationary',
            burnin_step=burnin_step,
            trace_orbit_func=dummy_trace_orbit_func,
    )

    assert np.allclose(true_comp.get_mean(), best_comp.get_mean(),
                       atol=1.0)
    assert np.allclose(true_comp.get_age(), best_comp.get_age(),
                       atol=1.0)
    assert np.allclose(true_comp.get_covmatrix(),
                       best_comp.get_covmatrix(),
                       atol=2.0)


def test_any_age_comp(age=32):
    """
    Tests an ellip component fit with age as a parameter
    """
    # log_filename = 'logs/compfitter_stationary.log'
    # synth_data_savefile = 'temp_data/compfitter_stationary_synthdata.fits'

    burnin_step = 2000

    true_comp_mean = [0., 0., 0., 2., 2., 2.]
    true_comp_dx = 8.
    true_comp_dy = 3.
    true_comp_du = 6.
    true_comp_dv = 3.
    true_comp_roll  = 0.
    true_comp_pitch = 0.5
    true_comp_yaw   = 1.
    true_comp_cov_xv = 2.5
    true_comp_age = age

    true_comp_pars = np.hstack([
            true_comp_mean,
            true_comp_dx,
            true_comp_dy,
            true_comp_du,
            true_comp_dv,
            true_comp_roll,
            true_comp_pitch,
            true_comp_yaw,
            true_comp_cov_xv,
            true_comp_age,
    ])

    true_comp = EllipComponent(true_comp_pars)

    nstars = 100
    measurement_error = 1e-10

    best_comp, chain, lnprob = run_fit_helper(
            true_comp=true_comp, starcounts=nstars,
            measurement_error=measurement_error,
            run_name='priorTest_11_age_%.1e'%true_comp_age,
            burnin_step=burnin_step,
            trace_orbit_func=trace_epicyclic_orbit,
    )
    print("Age: --~~~~~~~~~~~~~~~~", best_comp.get_age())

    # old_pars = best_comp.get_pars()
    # edited_init_pars = np.copy(old_pars)
    # # edited_init_pars[10] = old_pars[10]+(np.pi/2)
    # # edited_init_pars[11] = old_pars[11]+(np.pi/2)
    # edited_init_pars[12] = old_pars[12]+(np.pi/2)
    #
    # edited_best_comp, edited_chain, edited_lnprob = run_fit_helper(
    #         true_comp=true_comp, starcounts=nstars,
    #         measurement_error=measurement_error,
    #         run_name='priorTest_01_Edited_age_%.1e'%true_comp_age,
    #         burnin_step=burnin_step,
    #         trace_orbit_func=trace_epicyclic_orbit,
    #         init_pars=edited_init_pars
    # )
    # if lnprob.max() < edited_lnprob.max():
    #     print("lnProb - max", lnprob.max())
    #     print("edited _ lnProb - max", edited_lnprob.max())
    #
    #     print("-------------------------------------------------------")
    #     print("Pi/2 Error, A second fit was run and is set as best fit")
    #     print("-------------------------------------------------------")
    # print("New Age: --~~~~~~~~~~~~~~~~", edited_best_comp.get_age())

    assert np.allclose(true_comp.get_mean(), best_comp.get_mean(),
                       atol=1.0)
    assert np.allclose(true_comp.get_age(), best_comp.get_age(),
                       atol=1.0)
    assert np.allclose(true_comp.get_covmatrix(),
                       best_comp.get_covmatrix(),
                       atol=2.0)


if __name__ == '__main__':
    res = test_stationary_comp()
