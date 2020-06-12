import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from chronostar import expectmax
from chronostar import readparam
from chronostar import tabletool
from chronostar import component
from chronostar import traceorbit
from chronostar import likelihood

if len(sys.argv)<3:
    #~ filename_comp = 'scripts_scipy/betaPicParallel_1_from_emcee_continue_with_scipy/2/A/init_comps.npy'    filename_comp = 'scripts_scipy/betaPicParallel_1_from_emcee_continue_with_scipy/2/A/init_comps.npy'
    filename_comp = 'scripts_scipy/betaPicParallel_1_from_emcee_continue_with_scipy/2/A/final/final_comps.npy'
    filename_membership = 'scripts_scipy/betaPicParallel_1_from_emcee_continue_with_scipy/2/A/final/final_membership.npy'
    filename_params = '/Users/marusa/chronostar/scripts_scipy/bpic_start_1_from_emcee.pars'
else:
    filename_params = sys.argv[1]
    filename_comp = sys.argv[2]
    
    try:
        filename_membership = sys.argv[3]
    except:
        filename_membership = None
    
print(filename_params)
print(filename_comp)
print(filename_membership)

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

fit_pars = readparam.readParam(filename_params, default_pars=DEFAULT_FIT_PARS)
data_dict = tabletool.build_data_dict_from_table(fit_pars['data_table'])
fit_pars['trace_orbit_func'] = traceorbit.trace_epicyclic_orbit
#~ fit_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit
#~ print('CARTESIAN ORBIT!!')
print(fit_pars['data_table'])

nstars=len(data_dict['means'])
print('Number of stars', nstars)
ncomps=2
use_background=True

Component = component.SphereComponent
init_comps = Component.load_raw_components(filename_comp)


if filename_membership is None:
    memb_probs=np.ones((nstars, ncomps+use_background))\
                             / (ncomps+use_background)
    memb_probs_new = expectmax.expectation(data_dict, init_comps, memb_probs)
else:
    memb_probs_new = np.load(filename_membership)

#~ print(data_dict['means'])
#~ print(init_comps)
#~ print(memb_probs_new)

#~ print('memb_probs_new', memb_probs_new[:,0])
pos = np.load(filename_comp)

lnprob = likelihood.lnprob_func(init_comps[0].get_emcee_pars(), [data_dict, memb_probs_new[:,0], fit_pars['trace_orbit_func']])
print(lnprob)
