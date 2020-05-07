"""
Performs the 'maximisation' step of the EM algorithm for 1 component
only!

all_init_pars must be given in 'internal' form, that is the standard
deviations must be provided in log form.

The code creates folders and writes output into them: Describe this!

Parameters
----------
data: dict
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
    Number of components being fitted
icomp: int
    Compute maximisation for the icomp-th component.
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
DEATH_THRESHOLD: float {2.1}
    The total expected stellar membership below which a component is
    deemed 'dead' (if `ignore_dead_comps` is True)


Returns: Rewrite this as this is different now
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

import numpy as np
import os
import sys
sys.path.insert(0, '..')

from chronostar import compfitter
from chronostar import traceorbit
from chronostar import readparam
from chronostar import tabletool
from chronostar.component import SphereComponent

import logging

print('Importing done.')

def dummy_trace_orbit_func(loc, times=None):
    """
    Purely for testing purposes

    Dummy trace orbit func to skip irrelevant computation
    A little constraint on age (since otherwise its a free floating
    parameter)
    """
    if times is not None:
        if np.all(times > 1.):
            return loc + 1000.
    return loc



# For detailed description of parameters, see the main README.md file
# in parent directory.
DEFAULT_FIT_PARS = {
    'results_dir':'',

    # Output from dataprep, XYZUVW data, plus background overlaps
    # Can be a filename to a astropy table, or an actual table
    'data_table':None,

    # File name that points to a stored list of components, typically from
    # a previous fit. Some example filenames could be:
    #  - 'some/prev/fit/final_comps.npy
    #  - 'some/prev/fit/2/A/final_comps.npy
    # Alternatively, if you already have the list of components, just
    # provide them to `init_comps`. Don't do both.
    # 'init_comps_file':None, # TODO: Is this redundant with 'init_comps'
    'init_comps':None,

    # One of these two are required if initialising a run with ncomps != 1

    # One can also initialise a Chronostar run with memberships.
    # Array is [nstars, ncomps] float array
    # Each row should sum to 1.
    # Same as in 'final_membership.npy'
    #'init_memb_probs':None,     # TODO: IMPLEMENT THIS

    # Provide a string name that corresponds to a ComponentClass
    'component':'sphere',
    'max_comp_count':20,
    'max_em_iterations':200,
    'nthreads':1,     # TODO: NOT IMPLEMENTED
    'use_background':True,

    'overwrite_prev_run':False,
    'burnin':500,
    'sampling_steps':1000,
    'store_burnin_chains':False,
    'ignore_stable_comps':True,

    # If loading parameters from text file, can provide strings:
    #  - 'epicyclic' for epicyclic
    #  - 'dummy_trace_orbit_func' for a trace orbit funciton that doens't do antyhing (for testing)
    # Alternativley, if building up parameter dictionary in a script, can
    # provide actual function.
    'trace_orbit_func':traceorbit.trace_cartesian_orbit,

    'par_log_file':'fit_pars.log',
}


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


def maximisation_for_1_component(data, ncomps, memb_probs, burnin_steps, idir,
                 all_init_pars, all_init_pos=None, plot_it=False, pool=None,
                 convergence_tol=0.25, ignore_dead_comps=False,
                 Component=SphereComponent,
                 trace_orbit_func=None,
                 store_burnin_chains=False,
                 unstable_comps=None,
                 ignore_stable_comps=False,
                 nthreads=1,
                 icomp=None,
                 DEATH_THRESHOLD = 2.1,
                 ):


    # Ensure None value inputs are still iterable
    if all_init_pos is None:
        all_init_pos = ncomps * [None]
    if all_init_pars is None:
        all_init_pars = ncomps * [None]
    if unstable_comps is None:
        unstable_comps = ncomps * [True]

    log_message('Ignoring stable comps? {}'.format(ignore_stable_comps))
    log_message('Unstable comps are {}'.format(unstable_comps))


    log_message('Fitting comp {}'.format(icomp), symbol='.', surround=True)
    gdir = idir + "comp{}/".format(icomp)
    if not os.path.exists(gdir):
        os.makedirs(gdir)

    # If component has too few stars, skip fit, and use previous best walker
    if ignore_dead_comps and (np.sum(memb_probs[:, icomp]) < DEATH_THRESHOLD):
        logging.info("Skipped component {} with nstars {}".format(
                icomp, np.sum(memb_probs[:, icomp])
        ))
    elif ignore_stable_comps and not unstable_comps[icomp]:
        logging.info("Skipped stable component {}".format(icomp))
    # Otherwise, run maximisation and sampling stage
    else:
        best_comp, chain, lnprob = compfitter.fit_comp(
                data=data, memb_probs=memb_probs[:, icomp],
                burnin_steps=burnin_steps, plot_it=plot_it,
                pool=pool, convergence_tol=convergence_tol,
                plot_dir=gdir, save_dir=gdir, init_pos=all_init_pos[icomp],
                init_pars=all_init_pars[icomp], Component=Component,
                trace_orbit_func=trace_orbit_func,
                store_burnin_chains=store_burnin_chains,
                nthreads=nthreads,
        )
        logging.info("Finished fit")
        logging.info("Best comp pars:\n{}".format(
                best_comp.get_pars()
        ))
        final_pos = chain[:, -1, :]
        logging.info("With age of: {:.3} +- {:.3} Myr".
                     format(np.median(chain[:,:,-1]),
                            np.std(chain[:,:,-1])))

        best_comp.store_raw(gdir + 'best_comp_fit.npy')
        np.save(gdir + "best_comp_fit_bak.npy", best_comp) # can remove this line when working
        np.save(gdir + 'final_chain.npy', chain)
        np.save(gdir + 'final_lnprob.npy', lnprob)

    return best_comp, chain, lnprob, final_pos, icomp # I don't really need to return icomp


if __name__=='__main__':
    # Test
    
    icomp=0
    
    # Read fit parameters from the file
    fit_pars = readparam.readParam(sys.argv[1], default_pars=DEFAULT_FIT_PARS)
    
    ncomps = int(sys.argv[2]) # Is this OK?

    # Prepare data
    data_dict = tabletool.build_data_dict_from_table(fit_pars['data_table'])
    nstars = data_dict['means'].shape[0]

    burnin_steps = fit_pars['burnin']

    # memb_probs is what we get from the expectation step
    init_memb_probs = np.ones((nstars, ncomps)) / ncomps

    # Add background
    init_memb_probs = np.hstack((init_memb_probs, np.zeros((nstars,1))))
    memb_probs = init_memb_probs
    
    rdir = os.path.join(fit_pars['results_dir'], '{}/'.format(ncomps))
    iter_count=0 # This is an external parameter
    idir = os.path.join(rdir, "iter{:02}/".format(iter_count))
        
    all_init_pars = ncomps * [None] # Maybe I need to update this
    all_init_pos = ncomps * [None]
    pool=None
 
 
     # Set up trace_orbit_func
    if fit_pars['trace_orbit_func'] == 'dummy_trace_orbit_func':
        sfit_pars['trace_orbit_func'] = dummy_trace_orbit_func
    elif fit_pars['trace_orbit_func'] == 'epicyclic':
        log_message('trace_orbit: epicyclic')
        fit_pars['trace_orbit_func'] = traceorbit.trace_epicyclic_orbit
    else:
        fit_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit   
    
    best_comp, chain, lnprob, final_pos, icomp = maximisation_for_1_component(
        data_dict, ncomps, memb_probs, burnin_steps, idir,
        all_init_pars, all_init_pos=all_init_pos, pool=pool,
        nthreads=1,
        icomp=icomp)
