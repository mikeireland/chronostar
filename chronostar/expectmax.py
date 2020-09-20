"""
expectmax.py

Implementation of the expectation-maximisation algorithm used to fit
a multivariate gaussian mixture model of moving groups' origins
to a data set of stars, measured in Cartesian space, centred on and
co-rotating with the local standard of rest.

This module is in desperate need of a tidy. The entry point
`fit_many_comps` is particularly messy and clumsy.
"""
from __future__ import print_function, division

from distutils.dir_util import mkpath
import itertools
import logging
import numpy as np
import multiprocessing

# python3 throws FileNotFoundError that is essentially the same as IOError
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# The placement of logsumexp varies wildly between scipy versions
import scipy
_SCIPY_VERSION= [int(v.split('rc')[0])
                 for v in scipy.__version__.split('.')]
if _SCIPY_VERSION[0] == 0 and _SCIPY_VERSION[1] < 10:
    from scipy.maxentropy import logsumexp
elif ((_SCIPY_VERSION[0] == 1 and _SCIPY_VERSION[1] >= 3) or
    _SCIPY_VERSION[0] > 1):
    from scipy.special import logsumexp
else:
    from scipy.misc import logsumexp
from scipy import stats

import os

try:
    import matplotlib as mpl
    # prevents displaying plots from generation from tasks in background
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")

from .component import SphereComponent
from . import likelihood
from . import compfitter
from . import tabletool
try:
    print('Using C implementation in expectmax')
    from ._overlap import get_lnoverlaps
except:
    print("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    logging.info("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    from .likelihood import slow_get_lnoverlaps as get_lnoverlaps

#from functools import partial

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


def get_best_permutation(memb_probs, true_memb_probs):
    n_comps = memb_probs.shape[1]
    perms = itertools.permutations(np.arange(n_comps))
    best_perm = None
    min_diff = np.inf
    for perm in perms:
        diff = np.sum(np.abs(memb_probs[:,perm] - true_memb_probs))
        if diff < min_diff:
            min_diff = diff
            best_perm = perm
    return best_perm


def get_kernel_densities(background_means, star_means, amp_scale=1.0):
    """
    Build a PDF from `data`, then evaluate said pdf at `points`

    The Z and W value of points (height above, and velocity through the plane,
    respectively) are inverted in an effort to make the inferred background
    phase-space density independent of over-densities caused by suspected
    moving groups/associations. The idea is that the Galactic density is
    vertically symmetric about the plane, and any deviations are temporary.


    Parameters
    ----------
    background_means: [nstars,6] float array_like
        Phase-space positions of some star set that greatly envelops points
        in question. Typically contents of gaia_xyzuvw.npy.
    star_means: [npoints,6] float array_like
        Phase-space positions of stellar data that we are fitting components to
    amp_scale: float {1.0}
        One can optionally weight the background density so as to make over-densities
        more or less prominent. For e.g., amp_scale of 0.1 will make background
        overlaps an order of magnitude lower.

    Returns
    -------
    bg_lnols: [nstars] float array_like
        Background log overlaps of stars with background probability density
        function.
    """
    if type(background_means) is str:
        background_means = np.load(background_means)
    nstars = amp_scale * background_means.shape[0]

    kernel = stats.gaussian_kde(background_means.T)
    star_means = np.copy(star_means)
    star_means[:, 2] *= -1
    star_means[:, 5] *= -1

    bg_lnols = np.log(nstars)+kernel.logpdf(star_means.T)
    return bg_lnols


def get_background_overlaps_with_covariances(background_means, star_means,
                                             star_covs):
    """
    author: Marusa Zerjal 2019 - 05 - 25

    Determine background overlaps using means and covariances for both
    background and stars.
    Covariance matrices for the background are Identity*bandwidth.

    Takes about 3 seconds per star if using whole Gaia DR2 stars with 6D
    kinematics as reference.

    Parameters
    ----------
    background_means: [nstars,6] float array_like
        Phase-space positions of some star set that greatly envelops points
        in question. Typically contents of gaia_xyzuvw.npy, or the output of
        >> tabletool.build_data_dict_from_table(
                   '../data/gaia_cartesian_full_6d_table.fits',
                    historical=True)['means']
    star_means: [npoints,6] float array_like
        Phase-space positions of stellar data that we are fitting components to
    star_covs: [npoints,6,6] float array_like
        Phase-space covariances of stellar data that we are fitting components to

    Returns
    -------
    bg_lnols: [nstars] float array_like
        Background log overlaps of stars with background probability density
        function.

    Notes
    -----
    We invert the vertical values (Z and U) because the typical background
    density should be symmetric along the vertical axis, and this distances
    stars from their siblings. I.e. association stars aren't assigned
    higher background overlaps by virtue of being an association star.

    Edits
    -----
    TC 2019-05-28: changed signature such that it follows similar usage as
                   get_kernel_densitites
    """
    # Inverting the vertical values
    star_means = np.copy(star_means)
    star_means[:, 2] *= -1
    star_means[:, 5] *= -1

    # Background covs with bandwidth using Scott's rule
    d = 6.0 # number of dimensions
    nstars = background_means.shape[0]
    bandwidth = nstars**(-1.0 / (d + 4.0))
    background_cov = np.cov(background_means.T) * bandwidth ** 2
    background_covs = np.array(nstars * [background_cov]) # same cov for every star

    # shapes of the c_get_lnoverlaps input must be: (6, 6), (6,), (120, 6, 6), (120, 6)
    # So I do it in a loop for every star
    bg_lnols=[]
    for i, (star_mean, star_cov) in enumerate(zip(star_means, star_covs)):
        print('bgols', i)
        #print('{} of {}'.format(i, len(star_means)))
        #print(star_cov)
        #print('det', np.linalg.det(star_cov))
        #bg_lnol = get_lnoverlaps(star_cov, star_mean, background_covs,
        #                         background_means, nstars)
        try:
            #print('***********', nstars, star_cov, star_mean, background_covs, background_means)
            bg_lnol = get_lnoverlaps(star_cov, star_mean, background_covs,
                                     background_means, nstars)
            #print('intermediate', bg_lnol)
            # bg_lnol = np.log(np.sum(np.exp(bg_lnol))) # sum in linear space
            bg_lnol = logsumexp(bg_lnol) # sum in linear space

        # Do we really want to make exceptions here? If the sum fails then
        # there's something wrong with the data.
        except:
            # TC: Changed sign to negative (surely if it fails, we want it to
            # have a neglible background overlap?
            print('bg ln overlap failed, setting it to -inf')
            bg_lnol = -np.inf
        bg_lnols.append(bg_lnol)
        #print(bg_lnol)
        #print('')

    # This should be parallelized
    #bg_lnols = [np.sum(get_lnoverlaps(star_cov, star_mean, background_covs, background_means, nstars)) for star_mean, star_cov in zip(star_means, star_covs)]
    #print(bg_lnols)

    return bg_lnols


def check_convergence(old_best_comps, new_chains, perc=40):
    """Check if the last maximisation step yielded is consistent to new fit

    Convergence is achieved if previous key values fall within +/-"perc" of
    the new fits. With default `perc` value of 40, the previous best fits
    must be within the 80% range (i.e. not fall outside the bottom or top
    10th percentiles in any parameter) of the current chains.

    Parameters
    ----------
    old_best_fits: [ncomp] Component objects
        List of Components that represent the best possible fits from the
        previous run.
    new_chain: list of ([nwalkers, nsteps, npars] float array_like)
        The sampler chain from the new runs of each component
    perc: int
        the percentage distance that previous values must be within current
        values. Must be within 0 and 50

    Returns
    -------
    converged : bool
        If the runs have converged, return true
    """
    # Handle case where input is bad (due to run just starting out for e.g.)
    if old_best_comps is None:
        return False
    if old_best_comps[0] is None:
        return False

    # Check each run in turn
    each_converged = []
    for old_best_comp, new_chain in zip(old_best_comps, new_chains):
        med_and_spans = compfitter.calc_med_and_span(new_chain, perc=perc)
        upper_contained =\
            old_best_comp.get_emcee_pars() < med_and_spans[:,1]
        lower_contained = \
            old_best_comp.get_emcee_pars() > med_and_spans[:,2]
        each_converged.append(
            np.all(upper_contained) and np.all(lower_contained))

    return np.all(each_converged)


def calc_membership_probs(star_lnols):
    """Calculate probabilities of membership for a single star from overlaps

    Parameters
    ----------
    star_lnols : [ncomps] array
        The log of the overlap of a star with each group

    Returns
    -------
    star_memb_probs : [ncomps] array
        The probability of membership to each group, normalised to sum to 1
    """
    ncomps = star_lnols.shape[0]
    star_memb_probs = np.zeros(ncomps)

    for i in range(ncomps):
        star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))

    return star_memb_probs


def get_all_lnoverlaps(data, comps, old_memb_probs=None,
                       inc_posterior=False, amp_prior=None):
    """
    Get the log overlap integrals of each star with each component

    Parameters
    ----------
    data: dict -or- astropy.table.Table -or- path to astrop.table.Table
        if dict, should have following structure:
            'means': [nstars,6] float array_like
                the central estimates of star phase-space properties
            'covs': [nstars,6,6] float array_like
                the phase-space covariance matrices of stars
            'bg_lnols': [nstars] float array_like (opt.)
                the log overlaps of stars with whatever pdf describes
                the background distribution of stars.
        if table, see tabletool.build_data_dict_from_table to see
        table requirements.
    comps: [ncomps] syn.Group object list
        a fit for each comp (in internal form)
    old_memb_probs: [nstars, ncomps] float array {None}
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each comp. Each
        element is between 0.0 and 1.0 such that each row sums to 1.0
        exactly.
        If bg_hists are also being used, there is an extra column for the
        background (but note that it is not used in this function)
    inc_posterior: bool {False}
        If true, includes prior on groups into their relative weightings
    amp_prior: int {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`

    Returns
    -------
    lnols: [nstars, ncomps (+1)] float array
        The log overlaps of each star with each component, optionally
        with the log background overlaps appended as the final column
    """
    # Tidy input, infer some values
    if not isinstance(data, dict):
        data = tabletool.build_data_dict_from_table(data)
    nstars = len(data['means'])
    ncomps = len(comps)
    using_bg = 'bg_lnols' in data.keys()

    lnols = np.zeros((nstars, ncomps + using_bg))

    # Set up old membership probabilities
    if old_memb_probs is None:
        old_memb_probs = np.ones((nstars, ncomps)) / ncomps
    # 'weigths' is the same as 'amplitudes', amplitudes for components
    weights = old_memb_probs[:, :ncomps].sum(axis=0)

    # [ADVANCED/dodgy] Optionally scale each weight by the component prior, then rebalance
    # such that total expected stars across all components is unchanged
    if inc_posterior:
        comp_lnpriors = np.zeros(ncomps)
        for i, comp in enumerate(comps):
            comp_lnpriors[i] = likelihood.ln_alpha_prior(
                    comp, memb_probs=old_memb_probs
            )
        comp_starcount = weights.sum()
        weights *= np.exp(comp_lnpriors)
        weights = weights / weights.sum() * comp_starcount

    # Optionally scale each weight such that the total expected stars
    # is equal to or greater than `amp_prior`
    if amp_prior:
        if weights.sum() < amp_prior:
            weights *= amp_prior / weights.sum()

    # For each component, get log overlap with each star, scaled by
    # amplitude (weight) of each component's PDF
    for i, comp in enumerate(comps):
        lnols[:, i] = \
            np.log(weights[i]) + \
            likelihood.get_lnoverlaps(comp, data)

    # insert one time calculated background overlaps
    if using_bg:
        lnols[:,-1] = data['bg_lnols']
    return lnols


def calc_bic(data, ncomps, lnlike, memb_probs=None, Component=SphereComponent):
    """Calculates the Bayesian Information Criterion

    A simple metric to judge whether added components are worthwhile.
    The number of 'data points' is the expected star membership count.
    This way the BIC is (mostly) independent of the overall data set,
    if most of those stars are not likely members of the component fit.

    Parameters
    ----------
    data: dict
        See fit_many_comps
    ncomps: int
        Number of components used in fit
    lnlike: float
        the overall log likelihood of the fit
    memb_probs: [nstars,ncomps {+1}] float array_like
        See fit_many_comps
    Component:
        See fit_many_comps

    Returns
    -------
    bic: float
        A log likelihood score, scaled by number of free parameters. A
        lower BIC indicates a better fit. Differences of <4 are minor
        improvements.
    """
    if memb_probs is not None:
        nstars = np.sum(memb_probs[:, :ncomps])
    else:
        nstars = len(data['means'])
    ncomp_pars = len(Component.PARAMETER_FORMAT)
    n = nstars * 6                      # 6 for phase space origin
    k = ncomps * (ncomp_pars)           # parameters for each component model
                                        #  -1 for age, +1 for amplitude
    return np.log(n)*k - 2 * lnlike


def expectation(data, comps, old_memb_probs=None,
                inc_posterior=False, amp_prior=None):
    """Calculate membership probabilities given fits to each group

    Parameters
    ----------
    data: dict
        See fit_many_comps
    comps: [ncomps] Component list
        The best fit for each component from previous runs
    old_memb_probs: [nstars, ncomps (+1)] float array
        Memberhsip probability of each star to each component. Only used here
        to set amplitudes of each component.
    inc_posterior: bool {False}
        Whether to rebalance the weighting of each component by their
        relative priors
    amp_prior: float {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`

    Returns
    -------
    memb_probs: [nstars, ncomps] float array
        An array designating each star's probability of being a member to
        each component. It is populated by floats in the range (0.0, 1.0) such
        that each row sums to 1.0, each column sums to the expected size of
        each component, and the entire array sums to the number of stars.
    """
    # Tidy input and infer some values
    if not isinstance(data, dict):
        data = tabletool.build_data_dict_from_table(data)
    ncomps = len(comps)
    nstars = len(data['means'])
    using_bg = 'bg_lnols' in data.keys()

    # TODO: implement interation till convergence

    # if no memb_probs provided, assume perfectly equal membership
    if old_memb_probs is None:
        old_memb_probs = np.ones((nstars, ncomps+using_bg)) / (ncomps+using_bg)

    logging.info('start with get_allnoverlaps')

    # Calculate all log overlaps
    lnols = get_all_lnoverlaps(data, comps, old_memb_probs,
                               inc_posterior=inc_posterior, amp_prior=amp_prior)

    # Calculate membership probabilities, tidying up 'nan's as required
    memb_probs = np.zeros((nstars, ncomps + using_bg))
    for i in range(nstars):
        memb_probs[i] = calc_membership_probs(lnols[i])
    if np.isnan(memb_probs).any():
        log_message('AT LEAST ONE MEMBERSHIP IS "NAN"', symbol='!')
        memb_probs[np.where(np.isnan(memb_probs))] = 0.
    return memb_probs


def get_overall_lnlikelihood(data, comps, return_memb_probs=False,
                             old_memb_probs=None,
                             inc_posterior=False):
    """
    Get overall likelihood for a proposed model.

    Evaluates each star's overlap with every component and background
    If only fitting one group, inc_posterior does nothing

    Parameters
    ----------
    data: (dict)
        See fit_many_comps
    comps: [ncomps] list of Component objects
        See fit_many_comps
    return_memb_probs: bool {False}
        Along with log likelihood, return membership probabilites

    Returns
    -------
    overall_lnlikelihood: float
    """
    logging.info('here00')
    memb_probs = expectation(data, comps,
                             old_memb_probs=old_memb_probs,
                             inc_posterior=inc_posterior)
    
    logging.info('here0')
    
    all_ln_ols = get_all_lnoverlaps(data, comps,
                                    old_memb_probs=memb_probs,
                                    inc_posterior=inc_posterior)

    # multiplies each log overlap by the star's membership probability
    # (In linear space, takes the star's overlap to the power of its
    # membership probability)
    weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)
    
    if return_memb_probs:
        return np.sum(weighted_lnols), memb_probs
    else:
        return np.sum(weighted_lnols)

def maximise_one_comp(data, memb_probs, i, idir, all_init_pars=None, all_init_pos=None,
                ignore_stable_comps=False, ignore_dead_comps=False,
                DEATH_THRESHOLD=2.1, unstable_comps=None,
                burnin_steps=None, plot_it=False,
                pool=None, convergence_tol=0.25,
                plot_dir=None, save_dir=None,
                Component=SphereComponent,
                trace_orbit_func=None,
                store_burnin_chains=False,
                nthreads=1, 
                optimisation_method=None,
                nprocess_ncomp=False,
                ):

    """
    Performs the 'maximisation' step of the EM algorithm for 1 component
    at a time.

    all_init_pars must be given in 'internal' form, that is the standard
    deviations must be provided in log form.

    Parameters
    ----------
    data: dict
        See fit_many_comps
    memb_probs: [nstars, ncomps {+1}] float array_like
        See fit_many_comps
    i: int
        Perform optimisation for the i-th component of the model.
    DEATH_THRESHOLD: float {2.1}
        ...
    burnin_steps: int
        The number of steps for each burnin loop
    idir: str
        The results directory for this iteration
    all_init_pars: [ncomps, npars] float array_like
        The initial parameters around which to initialise emcee walkers
    all_init_pos: [ncomps, nwalkers, npars] float array_like
        The actual exact positions at which to initialise emcee walkers
        (from, say, the output of a previous emcee run)
    plot_it: bool {False}
        Whether to plot lnprob chains (from burnin, etc) as we go
    pool: MPIPool object {None}
        pool of threads to execute walker steps concurrently
    convergence_tol: float {0.25}
        How many standard devaitions an lnprob chain is allowed to vary
        from its mean over the course of a burnin stage and still be
        considered "converged". Default value allows the median of the
        final 20 steps to differ by 0.25 of its standard deviations from
        the median of the first 20 steps.
    ignore_dead_comps : bool {False}
        if componennts have fewer than 2(?) expected members, then ignore
        them
    ignore_stable_comps : bool {False}
        If components have been deemed to be stable, then disregard them
    Component: Implementation of AbstractComponent {Sphere Component}
        The class used to convert raw parametrisation of a model to
        actual model attributes.
    trace_orbit_func: function {None}
        A function to trace cartesian oribts through the Galactic potential.
        If left as None, will use traceorbit.trace_cartesian_orbit (base
        signature of any alternate function on this ones)
    optimisation_method: str {'emcee'}
        Optimisation method to be used in the maximisation step to fit
        the model. Default: emcee. Available: scipy.optimise.minimize with
        the Nelder-Mead method. Note that in case of the gradient descent,
        no chain is returned and meds and spans cannot be determined.
    nprocess_ncomp: bool {False}
        Compute maximisation in parallel? This is relevant only in case
        Nelder-Mead method is used: This method computes optimisation
        many times with different initial positions. The result is the 
        one with the best likelihood. These optimisations are computed
        in parallel if nprocess_ncomp equals True.
        
    Returns
    -------
    best_comp:
        The best fitting component.
    chain:
        
    lnprob:
        
    final_pos:
        The final positions of walkers for this maximisation. 
        Useful for restarting the next emcee run.
    """


    log_message('Fitting comp {}'.format(i), symbol='.', surround=True)
    gdir = idir + "comp{}/".format(i)
    mkpath(gdir)

    #~ # If component has too few stars, skip fit, and use previous best walker
    #~ if ignore_dead_comps and (np.sum(memb_probs[:, i]) < DEATH_THRESHOLD):
        #~ logging.info("Skipped component {} with nstars {}".format(
                #~ i, np.sum(memb_probs[:, i])
        #~ ))
    #~ elif ignore_stable_comps and not unstable_comps[i]:
        #~ logging.info("Skipped stable component {}".format(i))
    # Otherwise, run maximisation and sampling stage
    #~ else:

    best_comp, chain, lnprob = compfitter.fit_comp(
            data=data, memb_probs=memb_probs[:, i],
            burnin_steps=burnin_steps, plot_it=plot_it,
            pool=pool, convergence_tol=convergence_tol,
            plot_dir=gdir, save_dir=gdir, init_pos=all_init_pos[i],
            init_pars=all_init_pars[i], Component=Component,
            trace_orbit_func=trace_orbit_func,
            store_burnin_chains=store_burnin_chains,
            nthreads=nthreads, 
            optimisation_method=optimisation_method,
            nprocess_ncomp=nprocess_ncomp,
    )
    logging.info("Finished fit")
    logging.info("Best comp pars:\n{}".format(
            best_comp.get_pars()
    ))
    
    if optimisation_method=='emcee':
        final_pos = chain[:, -1, :]
        logging.info("With age of: {:.3} +- {:.3} Myr".
                     format(np.median(chain[:,:,-1]),
                            np.std(chain[:,:,-1])))
    elif optimisation_method=='Nelder-Mead':
        final_pos = chain
        logging.info("With age of: {:.3} Myr".
                     format(np.median(chain)))


    best_comp.store_raw(gdir + 'best_comp_fit.npy')
    np.save(gdir + "best_comp_fit_bak.npy", best_comp) # can remove this line when working
    np.save(gdir + 'final_chain.npy', chain)
    np.save(gdir + 'final_lnprob.npy', lnprob)
    
    return best_comp, chain, lnprob, final_pos
    

def maximisation(data, ncomps, memb_probs, burnin_steps, idir,
                 all_init_pars, all_init_pos=None, plot_it=False, pool=None,
                 convergence_tol=0.25, ignore_dead_comps=False,
                 Component=SphereComponent,
                 trace_orbit_func=None,
                 store_burnin_chains=False,
                 unstable_comps=None,
                 ignore_stable_comps=False,
                 nthreads=1, optimisation_method=None,
                 nprocess_ncomp=False,
                 ):
    """
    Performs the 'maximisation' step of the EM algorithm

    all_init_pars must be given in 'internal' form, that is the standard
    deviations must be provided in log form.

    Parameters
    ----------
    data: dict
        See fit_many_comps
    ncomps: int
        Number of components being fitted
    memb_probs: [nstars, ncomps {+1}] float array_like
        See fit_many_comps
    burnin_steps: int
        The number of steps for each burnin loop
    idir: str
        The results directory for this iteration
    all_init_pars: [ncomps, npars] float array_like
        The initial parameters around which to initialise emcee walkers
    all_init_pos: [ncomps, nwalkers, npars] float array_like
        The actual exact positions at which to initialise emcee walkers
        (from, say, the output of a previous emcee run)
    plot_it: bool {False}
        Whehter to plot lnprob chains (from burnin, etc) as we go
    pool: MPIPool object {None}
        pool of threads to execute walker steps concurrently
    convergence_tol: float {0.25}
        How many standard devaitions an lnprob chain is allowed to vary
        from its mean over the course of a burnin stage and still be
        considered "converged". Default value allows the median of the
        final 20 steps to differ by 0.25 of its standard deviations from
        the median of the first 20 steps.
    ignore_dead_comps : bool {False}
        if componennts have fewer than 2(?) expected members, then ignore
        them
    ignore_stable_comps : bool {False}
        If components have been deemed to be stable, then disregard them
    Component: Implementation of AbstractComponent {Sphere Component}
        The class used to convert raw parametrisation of a model to
        actual model attributes.
    trace_orbit_func: function {None}
        A function to trace cartesian oribts through the Galactic potential.
        If left as None, will use traceorbit.trace_cartesian_orbit (base
        signature of any alternate function on this ones)
    optimisation_method: str {'emcee'}
        Optimisation method to be used in the maximisation step to fit
        the model. Default: emcee. Available: scipy.optimise.minimize with
        the Nelder-Mead method. Note that in case of the gradient descent,
        no chain is returned and meds and spans cannot be determined.
    nprocess_ncomp: bool {False}
        How many processes to use in the maximisation of ncomps with
        python's multiprocessing library in case Nelder-Mead is used.
        
    Returns
    -------
    new_comps: [ncomps] Component array
        For each component's maximisation, we have the best fitting component
    all_samples: [ncomps, nwalkers, nsteps, npars] float array
        An array of each component's final sampling chain
    all_lnprob: [ncomps, nwalkers, nsteps] float array
        An array of each components lnprob
    all_final_pos: [ncomps, nwalkers, npars] float array
        The final positions of walkers from each separate Compoment
        maximisation. Useful for restarting the next emcee run.
    success_mask: np.where mask
        If ignoring dead components, use this mask to indicate the components
        that didn't die
    """
    # Set up some values
    DEATH_THRESHOLD = 2.1       # The total expected stellar membership below
                                # which a component is deemed 'dead' (if
                                # `ignore_dead_comps` is True)

    new_comps = []
    all_samples = []
    all_lnprob = []
    success_mask = []
    all_final_pos = ncomps * [None]

    # Ensure None value inputs are still iterable
    if all_init_pos is None:
        all_init_pos = ncomps * [None]
    if all_init_pars is None:
        all_init_pars = ncomps * [None]
    if unstable_comps is None:
        unstable_comps = ncomps * [True]

    log_message('Ignoring stable comps? {}'.format(ignore_stable_comps))
    log_message('Unstable comps are {}'.format(unstable_comps))


    ### MULTIPROCESSING
    if nprocess_ncomp and ncomps>1:
        logging.info("Maximising components with multiprocessing")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        def worker(i, return_dict):

            best_comp, chain, lnprob, final_pos = maximise_one_comp(data,
                memb_probs, i, all_init_pars=all_init_pars, 
                all_init_pos=all_init_pos, idir=idir, 
                ignore_stable_comps=ignore_stable_comps, 
                ignore_dead_comps=ignore_dead_comps,
                DEATH_THRESHOLD=DEATH_THRESHOLD, unstable_comps=unstable_comps,
                burnin_steps=burnin_steps, plot_it=plot_it,
                pool=pool, convergence_tol=0.25,
                Component=Component,
                trace_orbit_func=trace_orbit_func,
                store_burnin_chains=store_burnin_chains,
                nthreads=nthreads, 
                optimisation_method=optimisation_method,
                )

            return_dict[i] = {'best_comp': best_comp, 'chain': chain, 'lnprob': lnprob, 'final_pos': final_pos}

        jobs = []
        for i in range(ncomps):
            # If component has too few stars, skip fit, and use previous best walker
            if ignore_dead_comps and (np.sum(memb_probs[:, i]) < DEATH_THRESHOLD):
                logging.info("Skipped component {} with nstars {}".format(
                        i, np.sum(memb_probs[:, i])
                ))
            elif ignore_stable_comps and not unstable_comps[i]:
                logging.info("Skipped stable component {}".format(i))
            else:
                process = multiprocessing.Process(target=worker, args=(i, return_dict))
                jobs.append(process)

        # Start the threads (i.e. calculate the random number lists)
        for j in jobs:
            j.start()

        # Ensure all of the threads have finished
        for j in jobs:
            j.join()

        keys = return_dict.keys()
        keys = sorted(keys)
        
        for i in keys:
            v = return_dict[i]
            best_comp = v['best_comp']
            chain = v['chain']
            lnprob = v['lnprob']
            final_pos = v['final_pos']

            new_comps.append(best_comp)
            all_samples.append(chain)
            all_lnprob.append(lnprob)

        # Keep track of the components that weren't ignored
            success_mask.append(i)

        # record the final position of the walkers for each comp
            all_final_pos[i] = final_pos

    else:
        logging.info("Maximising components in a for loop")
        for i in range(ncomps):
            
            # If component has too few stars, skip fit, and use previous best walker
            if ignore_dead_comps and (np.sum(memb_probs[:, i]) < DEATH_THRESHOLD):
                logging.info("Skipped component {} with nstars {}".format(
                        i, np.sum(memb_probs[:, i])
                ))
            elif ignore_stable_comps and not unstable_comps[i]:
                logging.info("Skipped stable component {}".format(i))
            else:
                best_comp, chain, lnprob, final_pos = maximise_one_comp(data,
                    memb_probs, i, all_init_pars=all_init_pars,
                    all_init_pos=all_init_pos, idir=idir,
                    ignore_stable_comps=ignore_stable_comps,
                    ignore_dead_comps=ignore_dead_comps,
                    DEATH_THRESHOLD=DEATH_THRESHOLD, unstable_comps=unstable_comps,
                    burnin_steps=burnin_steps, plot_it=plot_it,
                    pool=pool, convergence_tol=0.25,
                    Component=Component,
                    trace_orbit_func=trace_orbit_func,
                    store_burnin_chains=store_burnin_chains,
                    nthreads=nthreads,
                    optimisation_method=optimisation_method,
                    )

                new_comps.append(best_comp)
                all_samples.append(chain)
                all_lnprob.append(lnprob)

                # Keep track of the components that weren't ignored
                success_mask.append(i)

                # record the final position of the walkers for each comp
                all_final_pos[i] = final_pos

    # # TODO: Maybe need to this outside of this call, so as to include
    # # reference to stable comps
    # Component.store_raw_components(idir + 'best_comps.npy', new_comps)
    # np.save(idir + 'best_comps_bak.npy', new_comps)

    return new_comps, all_samples, all_lnprob, \
           all_final_pos, success_mask


def check_stability(data, best_comps, memb_probs):
    """
    Checks if run has encountered problems

    Common problems include: a component losing all its members, lnprob
    return nans, a membership listed as nan

    Paramters
    ---------
    star_pars: dict
        See fit_many_comps
    best_comps: [ncomps] list of Component objects
        The best fits (np.argmax(chain)) for each component from the most
        recent run
    memb_probs: [nstars, ncomps] float array
        The membership array from the most recent run

    Returns
    -------
    stable: bool
        Whether or not the run is stable or not

    Notes
    -----
    TODO: For some reason runs are continuing past less than 2 members...
    """
    ncomps = len(best_comps)
    logging.info('DEBUG: memb_probs shape: {}'.format(memb_probs.shape))
    if np.min(np.sum(memb_probs[:, :ncomps], axis=0)) <= 2.:
        logging.info("ERROR: A component has less than 2 members")
        return False
    if not np.isfinite(get_overall_lnlikelihood(data, best_comps)):
        logging.info("ERROR: Posterior is not finite")
        return False
    if not np.isfinite(memb_probs).all():
        logging.info("ERROR: At least one membership is not finite")
        return False
    return True


def check_comps_stability(z, unstable_flags_old, ref_counts, using_bg, thresh=0.02):
    """
    Compares current total member count of each component with those
    from the last time it was deemed stable, and see if membership has
    changed strongly enough to warrant a refit of a component model

    TODO: maybe worth investigating if run can be deemed converged if all
    components are "stable". Tim think better safe than sorry.

    Parameters
    ----------
    z : [nstars,ncomps] float array
        Membership probability of each star with each component
    ref_counts : [ncomps] float array
        Stored expected membership of each component, when the component was
        last refitted.
    thresh : float {0.02}
        The threshold fractional difference within which the component
        is considered stable
    """
    ncomps = z.shape[1] - using_bg

    memb_counts = z.sum(axis=0)
    # Handle first call
    if ref_counts is None:
        unstable_flags = np.array(ncomps * [True])
        ref_counts = memb_counts

    else:
        # Update instability flag
        unstable_flags = np.abs((memb_counts - ref_counts)/ref_counts) > thresh
        # Disregard column for background memberships
        if using_bg:
            unstable_flags = unstable_flags[:-1]

        # Only update reference counts for components that have just been
        # refitted
        ref_counts[unstable_flags_old] = memb_counts[unstable_flags_old]

    return unstable_flags, ref_counts


def fit_many_comps(data, ncomps, rdir='', pool=None, init_memb_probs=None,
                   init_comps=None, inc_posterior=False, burnin=1000,
                   sampling_steps=5000, ignore_dead_comps=False,
                   Component=SphereComponent, trace_orbit_func=None,
                   use_background=False, store_burnin_chains=False,
                   ignore_stable_comps=False, max_em_iterations=100,
                   record_len=30, bic_conv_tol=0.1, min_em_iterations=30,
                   nthreads=1, optimisation_method=None, 
                   nprocess_ncomp = False,
                   **kwargs):
    """

    Entry point: Fit multiple Gaussians to data set

    This is where we apply the expectation maximisation algorithm.

    There are two ways to initialise this function, either:
    membership probabilities -or- initial components.
    If only fitting with one component (and a background) this function
    can initilialise itself.

    Parameters
    ----------
    data: dict -or- astropy.table.Table -or- path to astrop.table.Table
        if dict, should have following structure:
            'means': [nstars,6] float array_like
                the central estimates of star phase-space properties
            'covs': [nstars,6,6] float array_like
                the phase-space covariance matrices of stars
            'bg_lnols': [nstars] float array_like (opt.)
                the log overlaps of stars with whatever pdf describes
                the background distribution of stars.
        if table, see tabletool.build_data_dict_from_table to see
        table requirements.
    ncomps: int
        the number of components to be fitted to the data
    rdir: String {''}
        The directory in which all the data will be stored and accessed
        from
    pool: MPIPool object {None}
        the pool of threads to be passed into emcee
    init_memb_probs: [nstars, ngroups] array {None} [UNIMPLEMENTED]
        If some members are already known, the initialsiation process
        could use this.
    init_comps: [ncomps] Component list
        Initial components around whose parameters we can initialise
        emcee walkers.
    inc_posterior: bool {False}
        Whether to scale the relative component amplitudes by their priors
    burnin: int {1000}
        The number of emcee steps for each burnin loop
    sampling_steps: int {5000}
        The number of emcee steps for sampling a Component's fit
    ignore_dead_comps: bool {False}
        DEPRECATED FOR NOW!!!
        order groupfitter to skip maximising if component has less than...
        2..? expected members
    Component: Implementation of AbstractComponent {Sphere Component}
        The class used to convert raw parametrisation of a model to
        actual model attributes.
    trace_orbit_func: function {None}
        A function to trace cartesian oribts through the Galactic potential.
        If left as None, will use traceorbit.trace_cartesian_orbit (base
        signature of any alternate function on this ones)
    use_background: bool {False}
        Whether to incorporate a background density to account for stars
        that mightn't belong to any component.
    ignore_stable_comps: bool {False}
        Set to true if components that barely change should only be refitted
        every 5 iterations. Component stability is determined by inspecting
        whether the change in total star member count is less than 2% as
        compared to previous fit.
    optimisation_method: str {'emcee'}
        Optimisation method to be used in the maximisation step to fit
        the model. Default: emcee. Available: scipy.optimise.minimize with
        the Nelder-Mead method. Note that in case of the gradient descent,
        no chain is returned and meds and spans cannot be determined.
    nprocess_ncomp: bool {False}
        How many processes to use in the maximisation of ncomps with
        python's multiprocessing library in case Nelder-Mead is used.
        

    Return
    ------
    final_comps: [ncomps] list of synthesiser.Group objects
        the best fit for each component
    final_med_errs: [ncomps, npars, 3] array
        the median, -34 perc, +34 perc values of each parameter from
        each final sampling chain
    memb_probs: [nstars, ncomps] array
        membership probabilities

    """
    # Tidying up input
    if not isinstance(data, dict):
        data = tabletool.build_data_dict_from_table(
                data, get_background_overlaps=use_background
        )
    if rdir == '':                      # Ensure results directory has a
        rdir = '.'                      # trailing '/'
    rdir = rdir.rstrip('/') + '/'
    if not os.path.exists(rdir):
        mkpath(rdir)

    if use_background:
        assert 'bg_lnols' in data.keys()

    # filenames
    init_comp_filename = 'init_comps.npy'

    # setting up some constants
    nstars = data['means'].shape[0]
    C_TOL = 0.5

    logging.info("Fitting {} groups with {} burnin steps with cap "
                 "of {} iterations".format(ncomps, burnin, max_em_iterations))

    # INITIALISE RUN PARAMETERS

    # If initialising with components then need to convert to emcee parameter lists
    if init_comps is not None:
        logging.info('Initialised by components')
        all_init_pars = [ic.get_emcee_pars() for ic in init_comps]
        skip_first_e_step = False
        memb_probs_old = np.ones((nstars, ncomps+use_background))\
                         / (ncomps+use_background)

    # If initialising with membership probabilities, we need to skip first
    # expectation step, but make sure other values are iterable
    elif init_memb_probs is not None and init_comps is None: # MZ added and init_comps is None
        logging.info('Initialised by memberships')
        skip_first_e_step = True
        all_init_pars = ncomps * [None]
        init_comps = ncomps * [None]
        memb_probs_old = init_memb_probs

    # MZ
    # We need all_init_pars for scipy as a starting point
    elif init_memb_probs is not None and init_comps is not None:
        logging.info('Initialised by memberships')
        skip_first_e_step = True
        all_init_pars = np.array([c.get_emcee_pars() for c in init_comps])
        init_comps = ncomps * [None]
        memb_probs_old = init_memb_probs

    # If no initialisation provided, assume each star is equally probable to belong
    # to each component, but 0% likely to be part of the background
    # Currently only implemented blind initialisation for one component
    else:
        assert ncomps == 1, 'If no initialisation set, can only accept ncomp==1'
        logging.info('No specificed initialisation... assuming equal memberships')
        init_memb_probs = np.ones((nstars, ncomps)) / ncomps

        if use_background:
            init_memb_probs = np.hstack((init_memb_probs, np.zeros((nstars,1))))
        memb_probs_old    = init_memb_probs
        skip_first_e_step = True
        all_init_pars     = ncomps * [None]
        init_comps        = ncomps * [None]

    # Store the initial components if available
    if init_comps[0] is not None:
        Component.store_raw_components(rdir + init_comp_filename, init_comps)
    # np.save(rdir + init_comp_filename, init_comps)

    # Initialise values for upcoming iterations
    old_comps          = init_comps
    # old_overall_lnlike = -np.inf
    all_init_pos       = ncomps * [None]
    all_med_and_spans  = ncomps * [None]
    all_converged      = False
    stable_state       = True         # used to track issues

    # Keep track of all fits for convergence checking
    list_prev_comps        = []
    list_prev_memberships  = []
    list_all_init_pos      = []
    list_all_med_and_spans = []
    list_prev_bics         = []

    # Keep track of ALL BICs, so that progress can be observed
    all_bics = []

    # Keep track of unstable components, which will require
    # extra iterations
    ref_counts = None
    if ignore_stable_comps:
        unstable_comps = np.array(ncomps * [True])
    else:
        unstable_comps = None

    logging.info("Search for previous iterations")
    
    # Look for previous iterations and update values as appropriate
    prev_iters       = True
    iter_count       = 0
    found_prev_iters = False
    while prev_iters:
        try:
            idir = rdir+"iter{:02}/".format(iter_count)
            memb_probs_old = np.load(idir + 'membership.npy')
            try:
                old_comps = Component.load_raw_components(idir + 'best_comps.npy')
            # End up here if components aren't loadable due to change in module
            # So we rebuild from chains
            except AttributeError:
                old_comps = ncomps * [None]
                for i in range(ncomps):
                    chain   = np.load(idir + 'comp{}/final_chain.npy'.format(i))
                    lnprob  = np.load(idir + 'comp{}/final_lnprob.npy'.format(i))
                    npars   = len(Component.PARAMETER_FORMAT)
                    best_ix = np.argmax(lnprob)
                    best_pars    = chain.reshape(-1, npars)[best_ix]
                    old_comps[i] = Component(emcee_pars=best_pars)
                    logging.info('Now start with calc_med_and_spans')
                    all_med_and_spans[i] = compfitter.calc_med_and_span(
                            chain, intern_to_extern=True, Component=Component,
                    )

            all_init_pars = [old_comp.get_emcee_pars()
                             for old_comp in old_comps]
            logging.info('old_overall_lnlike')
            old_overall_lnlike, old_memb_probs = \
                    get_overall_lnlikelihood(data, old_comps,
                                             inc_posterior=False,
                                             return_memb_probs=True,)
            ref_counts = np.sum(old_memb_probs, axis=0)

            logging.info('append')
            list_prev_comps.append(old_comps)
            list_prev_memberships.append(old_memb_probs)
            list_all_init_pos.append(all_init_pos)
            list_all_med_and_spans.append(all_med_and_spans)
            list_prev_bics.append(calc_bic(data, len(old_comps),
                                           lnlike=old_overall_lnlike,
                                           memb_probs=old_memb_probs,
                                           Component=Component))

            all_bics.append(list_prev_bics[-1])

            iter_count += 1
            found_prev_iters = True

        except IOError:
            logging.info("Managed to find {} previous iterations".format(
                iter_count
            ))
            prev_iters = False

    # Until convergence is achieved (or max_iters is exceeded) iterate through
    # the Expecation and Maximisation stages

    logging.info("MZ: Start EM algorithm")

    # TODO: put convergence checking at the start of the loop so restarting doesn't repeat an iteration
    while not all_converged and stable_state and iter_count < max_em_iterations:
        ignore_stable_comps_iter = ignore_stable_comps and (iter_count % 5 != 0)

        # for iter_count in range(10):
        idir = rdir+"iter{:02}/".format(iter_count)
        mkpath(idir)

        log_message('Iteration {}'.format(iter_count),
                    symbol='-', surround=True)
        if not ignore_stable_comps_iter:
            log_message('Fitting all {} components'.format(ncomps))
            unstable_comps = np.where(np.array(ncomps * [True]))
        else:
            log_message('Fitting the following unstable comps:')
            log_message('TC: maybe fixed?')
            log_message(str(np.arange(ncomps)[unstable_comps]))
            log_message('MZ: removed this line due to index error (unstable_comps too big number)')
            log_message(str(unstable_comps))

        # EXPECTATION
        # Need to handle couple of side cases of initalising by memberships.
        if found_prev_iters:
            logging.info("Using previously found memberships")
            memb_probs_new = memb_probs_old
            found_prev_iters = False
            skip_first_e_step = False       # Unset the flag to initialise with
                                            # memb probs
        elif skip_first_e_step:
            logging.info("Using initialising memb_probs for first iteration")
            memb_probs_new = init_memb_probs
            skip_first_e_step = False
        else:
            memb_probs_new = expectation(data, old_comps, memb_probs_old,
                                         inc_posterior=inc_posterior)
        logging.info("Membership distribution:\n{}".format(
            memb_probs_new.sum(axis=0)
        ))
        np.save(idir+"membership.npy", memb_probs_new)

        # MAXIMISE
        new_comps, all_samples, _, all_init_pos, success_mask =\
            maximisation(data, ncomps=ncomps,
                         burnin_steps=burnin,
                         plot_it=True, pool=pool, convergence_tol=C_TOL,
                         memb_probs=memb_probs_new, idir=idir,
                         all_init_pars=all_init_pars,
                         all_init_pos=all_init_pos,
                         Component=Component,
                         ignore_dead_comps=ignore_dead_comps,
                         trace_orbit_func=trace_orbit_func,
                         store_burnin_chains=store_burnin_chains,
                         unstable_comps=unstable_comps,
                         ignore_stable_comps=ignore_stable_comps_iter,
                         nthreads=nthreads, 
                         optimisation_method=optimisation_method,
                         nprocess_ncomp=nprocess_ncomp
                         )

        for i in range(ncomps):
            if i in success_mask:
                j = success_mask.index(i)
                if optimisation_method=='emcee':
                    all_med_and_spans[i] = compfitter.calc_med_and_span(
                            all_samples[j], intern_to_extern=True,
                            Component=Component,
                    )
                else: # Nelder-Mead
                    all_med_and_spans[i] = None
                    
            # If component is stable, then it wasn't fit, so just duplicate
            # from last fit
            else:
                all_med_and_spans[i] = list_all_med_and_spans[-1][i]
                new_comps.insert(i,list_prev_comps[-1][i])
                all_init_pos.insert(i,list_all_init_pos[-1][i])

        Component.store_raw_components(idir + 'best_comps.npy', new_comps)
        np.save(idir + 'best_comps_bak.npy', new_comps)


        logging.info('DEBUG: new_comps length: {}'.format(len(new_comps)))

        # LOG RESULTS OF ITERATION
        overall_lnlike = get_overall_lnlikelihood(data, new_comps,
                                                 inc_posterior=False)
        overall_lnposterior = get_overall_lnlikelihood(data, new_comps,
                                                      inc_posterior=True)
        bic = calc_bic(data, ncomps, overall_lnlike,
                       memb_probs=memb_probs_new,
                       Component=Component)

        logging.info("---        Iteration results         --")
        logging.info("-- Overall likelihood so far: {} --".\
                     format(overall_lnlike))
        logging.info("-- Overall posterior so far:  {} --". \
                     format(overall_lnposterior))
        logging.info("-- BIC so far: {}                --". \
                     format(calc_bic(data, ncomps, overall_lnlike,
                                     memb_probs=memb_probs_new,
                                     Component=Component)))

        list_prev_comps.append(new_comps)
        list_prev_memberships.append(memb_probs_new)
        list_all_init_pos.append(all_init_pos)
        list_all_med_and_spans.append(all_med_and_spans)
        list_prev_bics.append(bic)

        all_bics.append(bic)

        if len(list_prev_bics) < min_em_iterations:
            all_converged = False
        else:
            all_converged = compfitter.burnin_convergence(
                    lnprob=np.expand_dims(list_prev_bics[-min_em_iterations:], axis=0),
                    tol=bic_conv_tol, slice_size=int(min_em_iterations/2)
            )
        old_overall_lnlike = overall_lnlike
        log_message('Convergence status: {}'.format(all_converged),
                    symbol='-', surround=True)
        if not all_converged:
            logging.info('BIC not converged')
            np.save(rdir + 'all_bics.npy', all_bics)

            # Plot all bics to date
            plt.clf()
            plt.plot(all_bics,
                     label='All {} BICs'.format(len(all_bics)))
            plt.vlines(np.argmin(all_bics), linestyles='--', color='red',
                       ymin=plt.ylim()[0], ymax=plt.ylim()[1],
                       label='best BIC {:.2f} | iter {}'.format(np.min(all_bics),
                                                                np.argmin(all_bics)))
            plt.legend(loc='best')
            plt.savefig(rdir + 'all_bics.pdf')

        # Check individual components stability
        if (iter_count % 5 == 0 and ignore_stable_comps):
            memb_probs_new = expectation(data, new_comps, memb_probs_new,
                                         inc_posterior=inc_posterior)
            log_message('Orig ref_counts {}'.format(ref_counts))

            unstable_comps, ref_counts = check_comps_stability(memb_probs_new,
                                                               unstable_comps,
                                                               ref_counts,
                                                               using_bg=use_background)
            log_message('New memb counts: {}'.format(memb_probs_new.sum(axis=0)))
            log_message('Unstable comps: {}'.format(unstable_comps))
            log_message('New ref_counts {}'.format(ref_counts))


        # Check stablity, but only affect run after sufficient iterations to
        # settle
        temp_stable_state = check_stability(data, new_comps, memb_probs_new)
        logging.info('Stability: {}'.format(temp_stable_state))
        if iter_count > 10:
            stable_state = temp_stable_state

        # only update if we're about to iterate again
        if not all_converged:
            old_comps = new_comps
            memb_probs_old = memb_probs_new

        iter_count += 1

    logging.info("CONVERGENCE COMPLETE")
    np.save(rdir + 'bic_list.npy', list_prev_bics)

    # Plot final few BICs
    plt.clf()
    nbics = len(list_prev_bics)
    start_ix = iter_count - nbics

    plt.plot(range(start_ix, iter_count), list_prev_bics,
             label='Final {} BICs'.format(len(list_prev_bics)))
    plt.vlines(start_ix + np.argmin(list_prev_bics), linestyle='--', color='red',
               ymin=plt.ylim()[0], ymax=plt.ylim()[1],
               label='best BIC {:.2f} | iter {}'.format(np.min(list_prev_bics),
                                                        start_ix+np.argmin(list_prev_bics)))
    plt.legend(loc='best')
    plt.savefig(rdir + 'bics.pdf')

    best_bic_ix = np.argmin(list_prev_bics)
    # Since len(list_prev_bics) is capped, need to count backwards form iter_count
    best_iter = iter_count - (len(list_prev_bics) - best_bic_ix)
    logging.info('Picked iteration: {}'.format(best_iter))
    logging.info('With BIC: {}'.format(list_prev_bics[best_bic_ix]))

    log_message('EM Algorithm finished', symbol='*')

    final_best_comps    = list_prev_comps[best_bic_ix]
    final_memb_probs    = list_prev_memberships[best_bic_ix]
    best_all_init_pos   = list_all_init_pos[best_bic_ix]
    final_med_and_spans = list_all_med_and_spans[best_bic_ix]

    log_message('Storing final result', symbol='-', surround=True)
    final_dir = rdir+'final/'
    mkpath(final_dir)

#         memb_probs_final = expectation(data, best_comps, best_memb_probs,
#                                        inc_posterior=inc_posterior)
    np.save(final_dir+'final_membership.npy', final_memb_probs)
    logging.info('Membership distribution:\n{}'.format(
        final_memb_probs.sum(axis=0)
    ))

    # Save membership fits file
    try:
        tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(self.fit_pars['data_table'], final_memb_probs, final_best_comps, os.path.join(final_dir, 'final_memberships_%d.fits'%len(final_best_comps)), get_background_overlaps=True, stellar_id_colname = self.fit_pars['stellar_id_colname'])
    except:
        logging.info("[WARNING] Couldn't print membership.fits file. Is source_id available?")

    # SAVE FINAL RESULTS IN MAIN SAVE DIRECTORY
    Component.store_raw_components(final_dir+'final_comps.npy', final_best_comps)
    np.save(final_dir+'final_comps_bak.npy', final_best_comps)
    np.save(final_dir+'final_med_and_spans.npy', final_med_and_spans)

    # Save components in fits file
    tabcomps = Component.convert_components_array_into_astropy_table(final_best_comps)
    tabcomps.write(os.path.join(final_dir, 'final_comps_%d.fits'%len(final_best_comps)), overwrite=True)

    overall_lnlike = get_overall_lnlikelihood(
            data, final_best_comps, inc_posterior=False
    )
    overall_lnposterior = get_overall_lnlikelihood(
            data, final_best_comps, inc_posterior=True
    )
    bic = calc_bic(data, ncomps, overall_lnlike,
                   memb_probs=final_memb_probs, Component=Component)
    logging.info("Final overall lnlikelihood: {}".format(overall_lnlike))
    logging.info("Final overall lnposterior:  {}".format(overall_lnposterior))
    logging.info("Final BIC: {}".format(bic))

    np.save(final_dir+'likelihood_post_and_bic.npy', (overall_lnlike,
                                                      overall_lnposterior,
                                                      bic))

    logging.info("FINISHED SAVING")
    logging.info("Best fits:\n{}".format(
        [fc.get_pars() for fc in final_best_comps]
    ))
    logging.info("Stars per component:\n{}".format(
            final_memb_probs.sum(axis=0)
    ))
    logging.info("Memberships: \n{}".format(
            (final_memb_probs*100).astype(np.int)
    ))

    # If compoents aren't super great, log a message, but return whatever we
    # get.
    if not stable_state:
        log_message('BAD RUN TERMINATED (not stable_state)', symbol='*', surround=True)

    logging.info(50*'=')

    return final_best_comps, np.array(final_med_and_spans), final_memb_probs

    # # Handle the case where the run was not stable
    # # Should this even return something....?
    # else:
    #     log_message('BAD RUN TERMINATED', symbol='*', surround=True)
    #
    #     # Store the bad results anyway, just in case.
    #     final_dir = rdir+'failed_final/'
    #     mkpath(final_dir)
    #     np.save(final_dir+'final_membership.npy', final_memb_probs)
    #     Component.store_raw_components(final_dir+'final_comps.npy', final_best_comps)
    #     np.save(final_dir+'final_comps_bak.npy', final_best_comps)
    #     np.save(final_dir+'final_med_and_spans.npy', final_med_and_spans)
    #     raise UserWarning('Was unable to reach convergence within given iterations')
    #     # return final_best_comps, np.array(final_med_and_spans), final_memb_probs

