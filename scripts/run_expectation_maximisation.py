"""
A Chronostar script that runs Expectation-Maximisation algorithm for
multiple components (only one?). It should really fit only one!

Run fit_many_comps here and call the rest of the functions directly
from expectmax.py

Run with
python run_expectation_maximisation.py testing.pars run_expectmax.pars

rdir: results dir for this number of components, e.g. testresults/2/ Where do ABC come from?
idir: iteration directory, e.g. testresults/2/iter00/
gdir: component directory, e.g. testresults/2/iter00/comp0/

Remove component stability check here as only one component is fitted!


Input
-----------------
data=self.data_dict,
ncomps: number of components to fit to the data (MZ: Does this include a new component that is to be added to the set as well? I think so.)
rdir: results folder (output destination)

Output: This should be printed out in a file.
-----------------
final_best_comps
final_med_and_spans
final_memb_probs


"""
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys
sys.path.insert(0, '..')

#~ import chronostar.expectmax as expectmax
#~ import chronostar.tabletool as tabletool
from chronostar import expectmax
from chronostar import tabletool
from chronostar import readparam
from chronostar import compfitter
from chronostar import component
from chronostar import likelihood

import subprocess # to call external scripts

import logging

import time

#~ def fit_many_comps(data, ncomps, rdir='', pool=None, init_memb_probs=None,
                   #~ init_comps=None, inc_posterior=False, burnin=1000,
                   #~ sampling_steps=5000, ignore_dead_comps=False,
                   #~ Component=SphereComponent, trace_orbit_func=None,
                   #~ use_background=False, store_burnin_chains=False,
                   #~ ignore_stable_comps=False, max_em_iterations=100,
                   #~ record_len=30, bic_conv_tol=0.1, min_em_iterations=30,
                   #~ nthreads=1,
                   #~ **kwargs):
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

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)

##################################
### SET PARAMETERS ###############
##################################
default_global_pars = readparam.readParam('default_fit.pars')

# Read global parameters from the file
global_pars = readparam.readParam(sys.argv[1], default_pars=default_global_pars)

# Read local parameters from the file
local_pars = readparam.readParam(sys.argv[2])

ncomps = local_pars['ncomps']
use_background = global_pars['use_background']

#~ component = 'sphere'#SphereComponent # TODO: from a par file
if global_pars['component'].lower() == 'sphere':
    Component = component.SphereComponent
elif global_pars['component'].lower() == 'ellip':
    Component = component.EllipComponent
else:
    raise UserWarning('Unknown (or missing) component parametrisation')

# EMCEE params
burnin = global_pars['burnin']
pool = None # TODO
nthreads = 1 # TODO

# EM PARAMS
max_em_iterations = global_pars['max_em_iterations']
min_em_iterations = global_pars['min_em_iterations']
#~ ignore_stable_comps = local_pars['ignore_stable_comps']
ignore_dead_comps = global_pars['ignore_dead_comps']


##################################
### OUTPUT DESTINATIONS ##########
##################################
rdir = os.path.join(global_pars['results_dir'], '{}/'.format(ncomps))
import shutil # TODO!!!!!!!!!!!!!!!!!!!!!!
#~ shutil.rmtree(rdir)
#~ print('DELETING PREVIOUS RESULTS...!!!!!')
if not os.path.exists(rdir):
    os.makedirs(rdir)

logging.basicConfig(filename=os.path.join(rdir, 'log.log'), level=logging.INFO)

# BIC plot output
filename_final_bic = os.path.join(rdir, 'bics.pdf')

##################################
### READ DATA ####################
##################################
# Stellar data
data = tabletool.build_data_dict_from_table(
        global_pars['data_table'], get_background_overlaps=use_background)

if use_background:
    assert 'bg_lnols' in data.keys()

nstars = data['means'].shape[0]

# Component data
init_comp_filename = os.path.join('init_comps.npy') # TODO: ADD DIR
if os.path.exists(init_comp_filename):
    init_comps = np.load(init_comp_filename) # TODO: Add step that selects only one component from this list
else:
    #~ init_comps = ncomps * [None]
    init_comps = [None] # Only one component here

#~ if os.path.exists(init_memb_probs_filename):
    #~ todo=True
#~ else:
    #~ init_memb_probs = None # TODO
init_memb_probs = None # TODO

##################################
### INITIALIZE ###################
##################################
logging.info("Fitting {} groups with {} burnin steps with cap "
             "of {} iterations".format(ncomps, burnin, max_em_iterations))
             
# If initialising with components then need to convert to emcee parameter lists
if init_comps[0] is not None:
    logging.info('Initialised by components')
    all_init_pars = [ic.get_emcee_pars() for ic in init_comps]
    skip_first_e_step = False
    memb_probs_old = np.ones((nstars, ncomps+use_background))\
                     / (ncomps+use_background)

# If initialising with membership probabilities, we need to skip first
# expectation step, but make sure other values are iterable
elif init_memb_probs is not None:
    logging.info('Initialised by memberships')
    skip_first_e_step = True
    all_init_pars = [None] # ncomps * 
    init_comps = [None] # ncomps * 
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
    all_init_pars     = [None] # ncomps * 
    init_comps        = [None] # ncomps * 

# Store the initial components if available
if init_comps[0] is not None:
    Component.store_raw_components(os.path.join(rdir, init_comp_filename), init_comps)
# np.save(rdir + init_comp_filename, init_comps)

# Initialise values for upcoming iterations
old_comps          = init_comps
lnols              = None # TODO Check if this is true. Do we have anything from previous iterations?
# old_overall_lnlike = -np.inf
all_init_pos       = [None] # ncomps * 
all_med_and_spans  = [None] # ncomps * 
converged          = False

# Keep track of all fits for convergence checking
list_prev_comps        = []
list_prev_memberships  = []
list_all_init_pos      = []
list_all_med_and_spans = []
list_prev_bics         = []

# Keep track of ALL BICs, so that progress can be observed
all_bics = []

##################################
### LOOK FOR PREVIOUS ITERATIONS #
##################################
#~ # Look for previous iterations and update values as appropriate
# (Not included here)
#~ prev_iters       = True
iter_count       = 0
found_prev_iters = False
#~ lnols           = 

##################################
### START EM ITERATIONS ##########
##################################

# Iterate through the Expecation and Maximisation stages until
# convergence is achieved (or max_iters is exceeded)

inc_posterior=False
filename_comp_prev_iter = ''
filename_init_pos_prev_iter = ''

# TODO: put convergence checking at the start of the loop so restarting doesn't repeat an iteration
while not converged and iter_count < max_em_iterations:
    time_start = time.time()
    
    print('\n\n\n\n')
    print('iter', iter_count)
    
    idir = os.path.join(rdir, "iter{:02}".format(iter_count))
    if not os.path.exists(idir):
        os.makedirs(idir)


    gdir = os.path.join(idir, "comp{}".format(local_pars['icomp'])) # Do I need idir or is gdir enough?
    if not os.path.exists(gdir):
        os.makedirs(gdir)
        print('MKDIR', gdir, 'iter', iter_count)

    filename_comp = os.path.join(gdir, 'best_comp_fit.npy') # TODO: Add this comp ID and define gdir
    filename_samples = os.path.join(gdir, 'final_chain.npy') # TODO
    filename_init_pos = os.path.join(gdir, 'init_pos.npy') # TODO
    filename_lnprob = os.path.join(gdir, 'final_lnprob.npy') # TODO

    #~ if filename_comp_prev_iter is None:
        #~ filename_comp_prev_iter = filename_comp

    log_message('Iteration {}'.format(iter_count),
                symbol='-', surround=True)

    ##################################
    ### EXPECTATION ##################
    ##################################
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
        memb_probs_new = expectmax.expectation(data, old_comps, memb_probs_old,
                                     inc_posterior=inc_posterior, lnols_precomputed=lnols)
    logging.info("Membership distribution:\n{}".format(
        memb_probs_new.sum(axis=0)
    ))
    
    filename_membership = os.path.join(idir, "membership.npy")
    np.save(filename_membership, memb_probs_new)

    ##################################
    ### MAXIMISATION #################
    ##################################
    
    print('main filename_comp', filename_comp)
    print('main filename_comp_prev_iter', filename_comp_prev_iter)
    
    # Write params file. TODO: Make defaults file. Review what is really
    # needed here and what could go into defaults.
    pars = {'ncomps':ncomps,
        'icomp': local_pars['icomp'],
        'iter_count': iter_count,
        'pool': pool,
        'nthreads': nthreads,
        'idir': idir,
        'all_init_pars': all_init_pars,
        'filename_membership': filename_membership,
        'filename_comp_prev_iter': filename_comp_prev_iter,
        'filename_comp': filename_comp,
        'filename_samples': filename_samples,
        'filename_init_pos': filename_init_pos,
        'filename_init_pos_prev_iter': filename_init_pos_prev_iter,
        'filename_lnprob': filename_lnprob,
        }
    filename_params = os.path.join(gdir, 'run_maximisation_1_comp.pars') # TODO: folder
    readparam.writeParam(filename_params, pars)
    
    time_end = time.time()
    dur = time_end-time_start
    print('DURATION part 1', dur)
    
    time_start = time.time()
    # Run external maximisation code - the results are written into files.
    bashCommand = 'python run_maximisation_1_comp.py testing.pars %s'%filename_params
    #~ bashCommand = 'mpirun -np 8 python run_maximisation_1_comp.py testing.pars %s'%filename_params
    #~ bashCommand = 'python run_maximisation_1_comp_gradient_descent.py testing.pars %s'%filename_params
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    #~ output, error = process.communicate()
    #~ _, _ = process.communicate()
    process_output, _ = process.communicate()
    print('process_output', process_output)

    time_end = time.time()
    dur = time_end-time_start
    print('DURATION maximisation with emcee/scipy optimizer', dur)

    # READ MAXIMISATION RESULTS
    new_comps = Component.load_raw_components(filename_comp)
    print('new_comps', new_comps)
    try:
        all_samples = np.load(filename_samples)
    except:
        all_samples = [None]
    all_init_pos = np.load(filename_init_pos) # not saved. Make it save it!
    

    #~ # DETERMINE MEDS AND SPANS
    #~ all_med_and_spans[0] = compfitter.calc_med_and_span(
            #~ all_samples[0], intern_to_extern=True,
            #~ Component=Component)
    # DETERMINE MEDS AND SPANS
    try:
        all_med_and_spans[0] = compfitter.calc_med_and_span(
                all_samples[0], intern_to_extern=True,
                Component=Component)
    except:
        all_med_and_spans = [None]
            #~ all_med_and_spans[i] = compfitter.calc_med_and_span(
                    #~ all_samples[j], intern_to_extern=True,
                    #~ Component=Component,
            #~ )


    # STORE COMPONENTS INTO A FILE: change it so it stores only one component.
    Component.store_raw_components(os.path.join(idir, 'best_comps.npy'), new_comps) # Isn't this saved in the maximisation algorithm? YES


    ##################################
    ### COMPUTE LNLIKE AND BIC #######
    ################################## 
    
    # Computationally expensive part. Do this only once as it only depends
    # on data and component. Save it into a file (maybe I can propagate it
    # in the loop but it should be saved when converged - will be needed
    # later for overall BIC etc.). # TODO
    time_start = time.time()
    lnols = likelihood.get_lnoverlaps(new_comps[0], data) # TODO: MAKE THIS PARALLEL

    time_end = time.time()
    dur = time_end-time_start
    print('DURATION lnols', dur)

    # LOG RESULTS OF ITERATION: OVERLAPS ARE COMPUTED AGAIN HERE - BUT THIS TIME WITH NEW COMPONENTS. Can this be used in the Expectation step in the next step of the iteration? I think so. Write there results in a file.
    # This things here need all the components anyway. So maybe this is not 
    
    overall_lnlike = expectmax.get_overall_lnlikelihood(data, new_comps, # This thing computes overlaps twice!
                            inc_posterior=False, lnols_precomputed=lnols) # SAVE this result of expectmax.expectation for the next step
                            
    overall_lnposterior = expectmax.get_overall_lnlikelihood(data, new_comps, # This thing computes overlaps twice!
                            inc_posterior=True, lnols_precomputed=lnols)
    
    # Deep down overlaps only depend on comp and data. Nothing else! So this step could be done once, and then
    # the rest can just read this data.
    
    # TODO: save one of these (check which one) overlaps for the expectation part in the next iteration!
    
    # BIC is determined from the number of members in a component (np.sum(memb_probs))
    # and lnlike. So it needs to be computed in every iteration.
    print('number of members', np.sum(memb_probs_new[:,local_pars['icomp']]))
    bic = expectmax.calc_bic(data, ncomps, overall_lnlike,
                   memb_probs=memb_probs_new,
                   Component=Component)

    logging.info("---        Iteration results         --")
    logging.info("-- Overall likelihood so far: {} --".\
                 format(overall_lnlike))
    logging.info("-- Overall posterior so far:  {} --". \
                 format(overall_lnposterior))
    logging.info("-- BIC so far: {}                --". \
                 format(expectmax.calc_bic(data, ncomps, overall_lnlike,
                                 memb_probs=memb_probs_new,
                                 Component=Component)))

    list_prev_comps.append(new_comps)
    list_prev_memberships.append(memb_probs_new)
    list_all_init_pos.append(all_init_pos)
    list_all_med_and_spans.append(all_med_and_spans)
    list_prev_bics.append(bic)

    all_bics.append(bic)
    
    print('BIC', bic)


    ##################################
    ### CHECK CONVERGENCE ############
    ################################## 
    
    if len(list_prev_bics) < min_em_iterations: # TODO: why min_em_iterations? Why lower limit?
        converged = False
    else:
        # Check early lnprob vals with final lnprob vals for convergence
        converged = compfitter.burnin_convergence(
                lnprob=np.expand_dims(list_prev_bics[-min_em_iterations:], axis=0),
                tol=global_pars['bic_conv_tol'], slice_size=int(min_em_iterations/2)
        )
    
    # MZ: I think overall_lnlike is for all components altogether. So not relevant here.
    old_overall_lnlike = overall_lnlike
    log_message('Convergence status: {}'.format(converged),
                symbol='-', surround=True)
    if not converged:
        logging.info('BIC not converged')
        np.save(rdir + 'all_bics.npy', all_bics)

        #~ # Plot all bics to date
        #~ plt.clf()
        #~ plt.plot(all_bics,
                 #~ label='All {} BICs'.format(len(all_bics)))
        #~ plt.vlines(np.argmin(all_bics), linestyles='--', color='red',
                   #~ ymin=plt.ylim()[0], ymax=plt.ylim()[1],
                   #~ label='best BIC {:.2f} | iter {}'.format(np.min(all_bics),
                                                            #~ np.argmin(all_bics)))
        #~ plt.legend(loc='best')
        #~ plt.savefig(rdir + 'all_bics.pdf')


    # only update if we're about to iterate again
    if not converged:
        old_comps = new_comps
        memb_probs_old = memb_probs_new
        filename_comp_prev_iter = filename_comp
        filename_init_pos_prev_iter = filename_init_pos

    iter_count += 1

logging.info("CONVERGENCE COMPLETE")
print("CONVERGENCE COMPLETE")

##################################
### SAVE RESULTS #################
##################################

np.save(os.path.join(rdir, 'bic_list.npy'), list_prev_bics)


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
final_dir = os.path.join(rdir, 'final')
os.makedirs(final_dir)

np.save(os.path.join(final_dir, 'final_membership.npy'), final_memb_probs)
logging.info('Membership distribution:\n{}'.format(
    final_memb_probs.sum(axis=0)
))

# SAVE FINAL RESULTS IN MAIN SAVE DIRECTORY
Component.store_raw_components(os.path.join(final_dir, 'final_comps.npy'), final_best_comps)
np.save(os.path.join(final_dir, 'final_comps_bak.npy'), final_best_comps) # TODO: delete this?
np.save(os.path.join(final_dir, 'final_med_and_spans.npy'), final_med_and_spans) # should it be np.array(final_med_and_spans)?

overall_lnlike = expectmax.get_overall_lnlikelihood(
        data, final_best_comps, inc_posterior=False
)
overall_lnposterior = expectmax.get_overall_lnlikelihood(
        data, final_best_comps, inc_posterior=True
)
bic = expectmax.calc_bic(data, ncomps, overall_lnlike,
               memb_probs=final_memb_probs, Component=Component)
logging.info("Final overall lnlikelihood: {}".format(overall_lnlike))
logging.info("Final overall lnposterior:  {}".format(overall_lnposterior))
logging.info("Final BIC: {}".format(bic))

np.save(os.path.join(final_dir, 'likelihood_post_and_bic.npy'), (overall_lnlike,
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
#~ if not stable_state:
    #~ log_message('BAD RUN TERMINATED', symbol='*', surround=True)

logging.info(50*'=')


##################################
### PLOTTING FINAL BICS ##########
##################################
logging.info("PLOTTING")
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
plt.savefig(filename_final_bic)


logging.info("EXPECTMAX FINISHED FOR...")
