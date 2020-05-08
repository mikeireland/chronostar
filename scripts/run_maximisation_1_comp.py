"""
Performs the maximisation step of the EM algorithm for 1 component
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

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)

#############################
### PREPARE DATA ############
#############################
# Example: python run_maximisation_1_comp.py testing.pars run_maximisation_1_comp.pars

# For detailed description of parameters, see the main README.md file
# in parent directory.
DEFAULT_FIT_PARS = readparam.readParam('default_fit.pars')

# Read global parameters from the file
fit_pars = readparam.readParam(sys.argv[1], default_pars=DEFAULT_FIT_PARS)

# Read local parameters from the file
local_pars = readparam.readParam(sys.argv[2])
ncomps = local_pars['ncomps']
icomp = local_pars['icomp']
burnin_steps = fit_pars['burnin']
ignore_stable_comps = local_pars['ignore_stable_comps']
ignore_dead_comps = local_pars['ignore_dead_comps']

if local_pars['Component'].lower()=='spherecomponent':
    component = SphereComponent
else:
    print('WARNING: Component not defined.')
    component=None

# Prepare data
#~ import time
#~ start = time.time()
data_dict = tabletool.build_data_dict_from_table(fit_pars['data_table'])
#~ end = time.time()
#~ print('data_dict took ', end-start)
nstars = data_dict['means'].shape[0]
print('nstars', nstars)



# memb_probs is what we get from the expectation step
init_memb_probs = np.ones((nstars, ncomps)) / ncomps

# Add background
init_memb_probs = np.hstack((init_memb_probs, np.zeros((nstars,1))))
memb_probs = init_memb_probs

rdir = os.path.join(fit_pars['results_dir'], '{}/'.format(ncomps))
idir = os.path.join(rdir, "iter{:02}/".format(local_pars['iter_count']))
    
all_init_pars = ncomps * [None] # Maybe I need to update this
all_init_pos = ncomps * [None]
pool=None


 # Set up trace_orbit_func
if fit_pars['trace_orbit_func'] == 'dummy_trace_orbit_func':
    sfit_pars['trace_orbit_func'] = traceorbit.dummy_trace_orbit_func
elif fit_pars['trace_orbit_func'] == 'epicyclic':
    log_message('trace_orbit: epicyclic')
    fit_pars['trace_orbit_func'] = traceorbit.trace_epicyclic_orbit
else:
    fit_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit   


# Ensure None value inputs are still iterable
if all_init_pos is None:
    all_init_pos = ncomps * [None]
if all_init_pars is None:
    all_init_pars = ncomps * [None]
unstable_comps = local_pars['unstable_comps']
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
            data=data_dict, memb_probs=memb_probs[:, icomp],
            burnin_steps=burnin_steps, plot_it=local_pars['plot_it'],
            pool=pool, convergence_tol=local_pars['convergence_tol'],
            plot_dir=gdir, save_dir=gdir, init_pos=all_init_pos[icomp],
            init_pars=all_init_pars[icomp], Component=component,
            trace_orbit_func=fit_pars['trace_orbit_func'],
            store_burnin_chains=local_pars['store_burnin_chains'],
            nthreads=local_pars['nthreads'],
    )
    logging.info("Finished fit")
    logging.info("Best comp pars:\n{}".format(
            best_comp.get_pars()
    ))
    final_pos = chain[:, -1, :]
    logging.info("With age of: {:.3} +- {:.3} Myr".
                 format(np.median(chain[:,:,-1]),
                        np.std(chain[:,:,-1])))

    #~ best_comp.store_raw(gdir + 'best_comp_fit.npy')
    #~ np.save(gdir + "best_comp_fit_bak.npy", best_comp) # can remove this line when working
    #~ np.save(gdir + 'final_chain.npy', chain)
    #~ np.save(gdir + 'final_lnprob.npy', lnprob)

    best_comp.store_raw(os.path.join(gdir, 'best_comp_fit.npy'))
    np.save(os.path.join(gdir, 'best_comp_fit_bak.npy'), best_comp) # can remove this line when working
    np.save(os.path.join(gdir, 'final_chain.npy'), chain)
    np.save(os.path.join(gdir, 'final_lnprob.npy'), lnprob)


#~ Don't return but print into a file
#~ print(best_comp, chain, lnprob, final_pos, icomp # I don't really need to return icomp
print(best_comp, chain, lnprob, final_pos)

