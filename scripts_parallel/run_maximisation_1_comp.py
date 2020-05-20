"""
Performs the maximisation step of the EM algorithm for 1 component only!


Run with
python run_maximisation_1_comp.py testing.pars run_maximisation_1_comp.pars


Required input data (global_pars):
- Stellar data

###################################################
##### Required fields in the local_pars file: #####

ncomps: Total number of components in the model
icomp: Fit icomp-th component here
idir: path to the iteration directory, e.g. results/1/iter00 or results/2/A/iter00

Output filenames:
filename_comp: filename of a npy file where component is stored
filename_samples: Store MCMC chain in this filename
filename_lnprob: Store lnprob in a file
filename_init_pos: Store final_pos in this file. This is used as a starting point in the next iteration.
###################################################

# OUTPUT OF THIS SCRIPT
- component fit results
- chain
- lnprob
- final_pos (this is used as an input in the next iteration)
- plots


Describe where this output goes
# Output destination
idir = local_pars['idir'] # os.path.join(rdir, "iter{:02}/".format(local_pars['iter_count']))
gdir = os.path.join(idir, "comp{}/".format(icomp))


all_init_pars must be given in 'internal' form, that is the standard
deviations must be provided in log form.


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

import warnings
warnings.filterwarnings("ignore")
print('run_maximisation_1_comp: all warnings suppressed.')

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

#~ from schwimmbad import MPIPool

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)

#~ pool=MPIPool()
#~ if not pool.is_master():
    #~ pool.wait()
    #~ sys.exit(0)

##################################
### SET PARAMETERS ###############
##################################
default_fit_pars = readparam.readParam('default_fit.pars')

# Read global parameters from the file
global_pars = readparam.readParam(sys.argv[1], default_pars=default_fit_pars)

# Read local parameters from the file
local_pars = readparam.readParam(sys.argv[2])

print('ocmp', sys.argv[2], sys.argv[1])

ncomps = local_pars['ncomps']
icomp = local_pars['icomp']

# TODO ###############
pool=None#pool#None

if global_pars['component'].lower()=='sphere':
    component = SphereComponent
else:
    print('WARNING: Component not defined.')
    component=None

# Set up trace_orbit_func. Maybe move this into compfitter.
if global_pars['trace_orbit_func'] == 'dummy_trace_orbit_func':
    global_pars['trace_orbit_func'] = traceorbit.dummy_trace_orbit_func
elif global_pars['trace_orbit_func'] == 'epicyclic':
    log_message('trace_orbit: epicyclic')
    global_pars['trace_orbit_func'] = traceorbit.trace_epicyclic_orbit
else:
    global_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit   


##################################
### READ DATA ####################
##################################
# Stellar data
#~ data_dict = tabletool.build_data_dict_from_table(global_pars['data_table'], mask_good=mask_good)
data_dict = tabletool.build_data_dict_from_table(global_pars['data_table'], get_background_overlaps=global_pars['use_background'])
#~ print('ONECOME', len(data_dict['means']), global_pars['data_table'])

# Membership: memb_probs is what we get from the expectation step
if os.path.exists(local_pars['filename_membership']):
    memb_probs = np.load(local_pars['filename_membership'])
else:
    # This is first run and we have to start somewhere
    nstars = data_dict['means'].shape[0]
    init_memb_probs = np.ones((nstars, ncomps)) / ncomps
    print('MEMB PROBS INIT EQUAL')

    # Add background
    if global_pars['use_background']:
        memb_probs = np.hstack((init_memb_probs, np.zeros((nstars,1))))

# Init_pos
if os.path.exists(local_pars['filename_init_pos']):
    all_init_pos = np.load(local_pars['filename_init_pos'])
else:
    #~ all_init_pos = ncomps * [None]
    all_init_pos = None

# Init_pars
if os.path.exists(local_pars['filename_init_pars']):
    all_init_pars = np.load(local_pars['filename_init_pars'])
else:
    #~ all_init_pos = ncomps * [None]
    all_init_pars = None

#~ print('ONECOMP', len(memb_probs), len(data_dict['means']))

##################################
### COMPUTATIONS #################
##################################

#~ print('SHAPES', memb_probs, all_init_pos, all_init_pars)


log_message('Fitting comp {}'.format(icomp), symbol='.', surround=True)
best_comp, chain, lnprob = compfitter.fit_comp(
        data=data_dict, memb_probs=memb_probs,
        init_pos=all_init_pos,
        init_pars=all_init_pars,
        burnin_steps=global_pars['burnin'],
        plot_it=global_pars['plot_it'], pool=pool,
        convergence_tol=global_pars['convergence_tol'],
        plot_dir=local_pars['gdir'], save_dir=local_pars['gdir'], Component=component,
        trace_orbit_func=global_pars['trace_orbit_func'],
        store_burnin_chains=global_pars['store_burnin_chains'],
        nthreads=global_pars['nthreads'],
)
logging.info("Finished fit")
logging.info("Best comp pars:\n{}".format(
        best_comp.get_pars()
))
final_pos = chain[:, -1, :]
logging.info("With age of: {:.3} +- {:.3} Myr".
             format(np.median(chain[:,:,-1]),
                    np.std(chain[:,:,-1])))

##################################
### SAVE RESULTS #################
##################################
best_comp.store_raw(local_pars['filename_comp'])
np.save(local_pars['filename_samples'], chain)
np.save(local_pars['filename_lnprob'], lnprob)

#~ pool.close()
