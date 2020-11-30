from __future__ import print_function, division

import numpy as np
import pytest
import sys

import chronostar.likelihood

sys.path.insert(0,'..')

from chronostar import expectmax as em
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool
from chronostar import expectmax
import chronostar.synthdata as syn
# import chronostar.retired2.measurer as ms
# import chronostar.retired2.converter as cv

#
# def test_calcMedAndSpan():
#     """
#     Test that the median, and +- 34th percentiles is found correctly
#     """
#     dx = 10.
#     dv = 5.
#     dummy_mean = np.array([10,10,10, 5, 5, 5,np.log(dx),np.log(dv),20])
#     dummy_std =  np.array([1.,1.,1.,1.,1.,1.,0.5,       0.5,       3.])
#     assert len(dummy_mean) == len(dummy_std)
#     npars = len(dummy_mean)
#
#     nsteps = 10000
#     nwalkers = 18
#
#     dummy_chain = np.array([np.random.randn(nsteps)*std + mean
#                             for (std, mean) in zip(dummy_std, dummy_mean)]).T
#     np.repeat(dummy_chain, 18, axis=0).reshape(nwalkers,nsteps,npars)
#
#     med_and_span = em.calcMedAndSpan(dummy_chain)
#     assert np.allclose(dummy_mean, med_and_span[:,0], atol=0.1)
#     approx_stds = 0.5*(med_and_span[:,1] - med_and_span[:,2])
#     assert np.allclose(dummy_std, approx_stds, atol=0.1)

def test_calcMembershipProbs():
    """
    Even basicer. Checks that differing overlaps are
    correctly mapped to memberships.
    """
    # case 1
    star_ols = [10, 10]
    assert np.allclose([.5,.5], em.calc_membership_probs(np.log(star_ols)))

    # case 2
    star_ols = [10, 30]
    assert np.allclose([.25,.75], em.calc_membership_probs(np.log(star_ols)))

    # case 3
    star_ols = [10, 10, 20]
    assert np.allclose([.25, .25, .5],
                       em.calc_membership_probs(np.log(star_ols)))


def test_expectation():
    """
    Super basic, generates some association stars along
    with some background stars and checks membership allocation
    is correct
    """

    age = 1e-5
    ass_pars1 = np.array([0, 0, 0, 0, 0, 0, 5., 2., age])
    comp1 = SphereComponent(ass_pars1)
    ass_pars2 = np.array([100., 0, 0, 20, 0, 0, 5., 2., age])
    comp2 = SphereComponent(ass_pars2)
    starcounts = [100,100]
    synth_data = SynthData(pars=[ass_pars1, ass_pars2],
                           starcounts=starcounts)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    true_memb_probs = np.zeros((np.sum(starcounts), 2))
    true_memb_probs[:starcounts[0], 0] = 1.
    true_memb_probs[starcounts[0]:, 1] = 1.

    # star_means, star_covs = tabletool.buildDataFromTable(synth_data.astr_table)
    # all_lnols = em.getAllLnOverlaps(
    #         synth_data.astr_table, [comp1, comp2]
    # )

    fitted_memb_probs = em.expectation(
            tabletool.build_data_dict_from_table(synth_data.table),
            [comp1, comp2]
    )

    assert np.allclose(true_memb_probs, fitted_memb_probs, atol=1e-10)

'''
@pytest.mark.skip
def test_fit_many_comps_gradient_descent_with_multiprocessing():
    """
    Added by MZ 2020 - 07 - 13
    
    Test if maximisation works when using gradient descent and multiprocessing.
    """
    
    
    age = 1e-5
    ass_pars1 = np.array([0, 0, 0, 0, 0, 0, 5., 2., age])
    comp1 = SphereComponent(ass_pars1)
    starcounts = [100,]
    synth_data = SynthData(pars=[ass_pars1,],
                           starcounts=starcounts)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    true_memb_probs = np.zeros((np.sum(starcounts), 2))
    true_memb_probs[:starcounts[0], 0] = 1.
    true_memb_probs[starcounts[0]:, 1] = 1.
    
    
    ncomps = len(starcounts)
    
    best_comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(synth_data.table, ncomps, 
                   rdir='test_gradient_descent_multiprocessing', 
                   #~ init_memb_probs=None,
                   #~ init_comps=None,
                   trace_orbit_func=None,
                   optimisation_method='Nelder-Mead', 
                   nprocess_ncomp = True,
                   )
'''

@pytest.mark.skip(reason='Too long for unit tests. Put this in integration instead')
def test_maximisation_gradient_descent_with_multiprocessing_tech():
    """
    Added by MZ 2020 - 07 - 13
    
    Test if maximisation works when using gradient descent and multiprocessing.
    NOTE: this is not a test if maximisation returns appropriate results but
    it only tests if the code runs withour errors. This is mainly to test
    multiprocessing.
    """
    
    
    age = 1e-5
    ass_pars1 = np.array([0, 0, 0, 0, 0, 0, 5., 2., age])
    comp1 = SphereComponent(ass_pars1)
    starcounts = [100,]
    synth_data = SynthData(pars=[ass_pars1,],
                           starcounts=starcounts)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    true_memb_probs = np.zeros((np.sum(starcounts), 1))
    true_memb_probs[:starcounts[0], 0] = 1.
    #~ true_memb_probs[starcounts[0]:, 1] = 1.
    
    ncomps = len(starcounts)
    
    noise = np.random.rand(ass_pars1.shape[0])*5
    
    all_init_pars = [ass_pars1 + noise]

    new_comps, all_samples, _, all_init_pos, success_mask =\
        expectmax.maximisation(synth_data.table, ncomps, 
                true_memb_probs, 100, 'iter00',
                all_init_pars,
                optimisation_method='Nelder-Mead',
                nprocess_ncomp=True,
                )

    # TODO: test if new_comps, all_samples, _, all_init_pos, success_mask are of the right format.



# def test_background_overlaps():
#     """
#     Author: Marusa Zerjal, 2019 - 05 - 26

    # Compare background overlap with KDE and background overlap with tiny covariance matrix

    # :return:
    # """

    # background_means = tabletool.build_data_dict_from_table(kernel_density_input_datafile,
    #     only_means=True,
    # )

    # ln_bg_ols_kde = em.get_kernel_densities(background_means,
    # #                                           star_means, )


if __name__=='__main__':
    test_maximisation_gradient_descent_with_multiprocessing_tech()
