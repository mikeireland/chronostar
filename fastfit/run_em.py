"""
MZ: a standalone EM algorithm
python3 run_em.py example.pars


NOTE: comments below are not up to date. REVIEW!!!!

A Chronostar script that runs Expectation-Maximisation algorithm for
multiple components (only one?). It should really fit only one!
Run fit_many_comps here and call the rest of the functions directly
from expectmax.py
Run with
python run_expectation_maximisation.py testing.pars run_expectmax.pars
# Wrong
rdir: results dir for this number of components, e.g. testresults/2/ Where do 
ABC come from?
idir: iteration directory, e.g. testresults/2/iter00/
gdir: component directory, e.g. testresults/2/iter00/comp0/
local_pars['run_dir']: testresults/2/A/
Remove component stability check here as only one component is fitted!
Input
-----------------
data=self.data_dict,
ncomps: number of components to fit to the data (MZ: Does this include a new 
component that is to be added to the set as well? I think so.)
rdir: results folder (output destination)
Output: This should be printed out in a file.
-----------------
final_best_comps
final_med_and_spans
final_memb_probs
"""
import warnings
warnings.filterwarnings("ignore")
print('run_expectation_maximisation: all warnings suppressed.')

import numpy as np
#import matplotlib.pyplot as plt
import os.path
import sys
sys.path.insert(0, '..')


#~ from chronostar import expectmax2 as expectmax
from chronostar import expectmax
from chronostar import tabletool
from chronostar import readparam
#from chronostar import compfitter
from chronostar import component
#from chronostar import likelihood
from chronostar import traceorbit
from chronostar import default_pars
from chronostar import utils

# New libraries
from chronostar import expectation_marusa as expectation
#~ from chronostar import maximisation
from chronostar import maximisationC

try:
    from chronostar._expectation import expectation as expectationC
except ImportError:
    print("C IMPLEMENTATION OF expectation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?

try:
    from chronostar._temporal_propagation import trace_epicyclic_orbit, trace_epicyclic_covmatrix
except ImportError:
    print("C IMPLEMENTATION OF temporal_propagation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?


#~ import subprocess # to call external scripts

import logging

#~ import time


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

def lnprob_convergence(lnprob, slice_size=10):
    """
    Check if lnprob is not changing anymore: Determine median values
    for chunks of slice_size. If median worsens, declare convergence.
    """
    lnprob = np.array(lnprob)
    
    chunk_size = int(float(len(lnprob))/float(slice_size))
    indices_chunks = np.array_split(range(len(lnprob)), chunk_size)

    # Medians of lnprob for chunks of chunk_size
    medians = [np.nanmedian(lnprob[i]) for i in indices_chunks]

    # Did the median worsen? Then we claim convergence!
    #~ convergence = medians[-2]>medians[-1]
    
    # Convergence when the median is not significantly improved anymore
    f = 0.01
    r1 = np.abs(1.0 - medians[-1]/medians[-2])
    r2 = np.abs(1.0 - medians[-1]/medians[-3])
    convergence = (r1<f) & (r2<f)

    # Did the median worsen for the last two chunks? Then we claim convergence!
    #~ convergence = (medians[-2]>medians[-3]) & (medians[-1]>medians[-3])
    
    print('CONVERGENCE', convergence, len(lnprob), chunk_size, r1, r2, medians)

    return convergence


def run_expectmax_simple(pars, data_dict=None, init_comps=None, 
    init_memb_probs=None):
    """
    Run expectation-maximisation algorithm...
    """

    ####################################################################
    #### PARAMETERS ####################################################
    ####################################################################
    # Component type
    if pars['component'].lower() == 'sphere':
        Component = component.SphereComponent
    elif pars['component'].lower() == 'ellip':
        Component = component.EllipComponent
    else:
        raise UserWarning('Unknown (or missing) component parametrisation')


    # Set up trace_orbit_func
    if pars['trace_orbit_func'] == 'epicyclic':
        utils.log_message('trace_orbit: epicyclic')
        trace_orbit_func = traceorbit.trace_epicyclic_orbit
    else:
        utils.log_message('trace_orbit: cartesian (galpy)')
        trace_orbit_func = traceorbit.trace_cartesian_orbit




    # Do we really need these?
    use_box_background = True # TODO: MZ: I made this up because this parameter is needed.... Revise this!!!
    inc_posterior=False


    ####################################################################
    ### OUTPUT DESTINATION #############################################
    ####################################################################
    folder_destination = pars['folder_destination']
    if not os.path.exists(folder_destination):
        os.makedirs(folder_destination)


    ####################################################################
    #### READ DATA if not provided as an argument ######################
    ####################################################################
    # Stellar data
    if data_dict is None:
        data_dict = tabletool.build_data_dict_from_table(
            pars['data_table'], get_background_overlaps=pars['use_background']) # TODO: background???

    # Read initial membership probabilities
    if init_memb_probs is None:
        filename_init_memb_probs = pars['filename_init_memb_probs']
        if filename_init_memb_probs is not None and os.path.exists(filename_init_memb_probs):
            init_memb_probs = np.load(filename_init_memb_probs)
            print('Managed to load in %d init_memb_probs from file'%\
                len(init_memb_probs))
    
    # Read initial components
    if init_comps is None:
        filename_init_comps = pars['filename_init_comps']
        if filename_init_comps is not None and os.path.exists(filename_init_comps):
            init_comps = Component.load_raw_components(filename_init_comps)
            print('Managed to load in %d init_comps from file'%\
                len(init_comps))
        else:
            init_comps = [None]








    # TODO: review this
    # Rething this. Background should always be used...
    use_background = pars['use_background']
    if use_background:
        assert 'bg_lnols' in data_dict.keys()

    #~ nstars = data_dict['means'].shape[0]
    #~ print('EMnstars', nstars, pars['data_table'])





    ####################################################################
    #### INITIAL COMPONENTS, MEMBERSHIPS, NCOMPS AND INIT_PARS #########
    ####################################################################
    # Update missing info
    
    # No info known. Set ncomps=1
    if init_memb_probs is None and init_comps[0] is None:
        logging.info('No specificed initialisation... assuming all stars are members.')
        ncomps = 1
        init_comps = [None]
        all_init_pars = [None]

        # Assume all stars are members of the component. Background 0
        init_memb_probs = np.zeros((len(data_dict['means']),
            ncomps + pars['use_background']))
        init_memb_probs[:, 0] = 1. - 1.e-10
        init_memb_probs[:, 1] = 1.e-10        
    
    # init_memb_probs available, but not comps
    elif init_memb_probs is not None and init_comps[0] is None:
        logging.info('Initialised by memberships')
        ncomps = init_memb_probs.shape[1]-1
        init_comps = [None] * ncomps
        all_init_pars = [None] * ncomps
    
    # Comps available, but not init_memb_probs
    elif init_memb_probs is None and init_comps[0] is not None:
        logging.info('Initialised by components')
        ncomps = len(init_comps)
        all_init_pars = [ic.get_emcee_pars() for ic in init_comps]

        # Assume equal memberships to start with. +1 for background
        nstars=len(data_dict['means'])
        memb_probs_tmp = np.ones((nstars, ncomps+1)) / (ncomps+1)
                
        #~ init_memb_probs = expectmax.expectationC(data_dict, init_comps, 
        init_memb_probs = expectmax.expectation(data_dict, init_comps, 
            memb_probs_tmp, inc_posterior=inc_posterior,
            use_box_background=use_box_background)

    # Everything available
    else:
        logging.info('Initialised by components and memberships')
        ncomps = len(comps)
        assert ncomps==init_memb_probs.shape[1]-1
        all_init_pars = [ic.get_emcee_pars() for ic in init_comps]


    # Check if ncomps matches the number from the pars
    try:
        if pars['ncomps']!=ncomps:
            print('WARNING: ncomps (%d) determined from the data does NOT match the number specified in the pars file (%d)!!!'%(ncomps, pars['ncomps']))
    except:
        pass

    print('EM ncomps: %d'%ncomps)


    ####################################################################
    #### INITIALIZE ####################################################
    ####################################################################
    #~ logging.info("Fitting {} groups with {} burnin steps with cap "
        #~ "of {} iterations".format(ncomps, burnin, max_em_iterations))

    # Initialise values for upcoming iterations
    memb_probs_old = init_memb_probs
    comps_old = init_comps
    #lnols = None
    all_init_pos = [None] * ncomps

    # Keep track of all fits for convergence checking
    list_prev_comps        = []
    list_prev_memberships  = []
    list_all_init_pos      = []
    list_prev_lnlikes      = []


    ####################################################################
    #### START EM ITERATIONS ###########################################
    ####################################################################
    # Iterate through the Expecation and Maximisation stages until
    # convergence is achieved (or max_iters is exceeded)

    converged = False
    iter_count = 0
    while not converged and iter_count < pars['max_em_iterations']:
        ################################################################
        #### Folders and filenames #####################################
        ################################################################
        # Folder for iteration
        folder_iter = os.path.join(pars['folder_destination'], 
            str(ncomps), pars['split_label'], 
            "iter{:03}".format(iter_count))
        
        if not os.path.exists(folder_iter):
            try:
                os.makedirs(folder_iter)
            except:
                # When doing this in parallel, more than one process might 
                # try to create this dir at the same time.
                pass


        #~ utils.log_message('Iteration {}'.format(iter_count),
                    #~ symbol='-', surround=True)

        filename_memberships_iter = os.path.join(folder_iter, 
            pars['filename_iter_memberships'])
        filename_components_iter = os.path.join(folder_iter, 
            pars['filename_iter_comps'])
        filename_lnprob_and_bic_iter= os.path.join(folder_iter, 
            pars['filename_iter_lnprob_and_bic'])


        ################################################################
        #### MAXIMISATION ##############################################
        ################################################################   
        
        print('################# START MAXIMISATION')
        # maximisation.maximisation_gradient_descent_serial(
        # maximisation.maximisation_gradient_descent_multiprocessing(
        #~ comps_new, _, all_init_pos =\
            #~ maximisation.maximisation_gradient_descent_serial(
                #~ data_dict, ncomps=ncomps, 
                #~ convergence_tol=pars['convergence_tol'],
                #~ memb_probs=memb_probs_old, all_init_pars=all_init_pars,
                #~ all_init_pos=all_init_pos, 
                #~ trace_orbit_func=trace_orbit_func, Component=Component,
                #~ optimisation_method=pars['optimisation_method'],
                #~ idir=folder_iter,
            #~ )
            
        comps_new, _, all_init_pos =\
            maximisationC.maximisation_gradient_descent_serial(
            data_dict, ncomps=ncomps, memb_probs=memb_probs_old, 
            all_init_pars=all_init_pars, all_init_pos=all_init_pos,
            Component=Component, trace_orbit_func=trace_orbit_func, 
            optimisation_method=pars['optimisation_method'], 
            idir=folder_iter)
            
        print('################# END MAXIMISATION')
        # Save new components
        Component.store_raw_components(filename_components_iter, 
            comps_new)


        ################################################################
        #### EXPECTATION ###############################################
        ################################################################

        #~ import pickle
        #~ with open('input_data_to_expectation.pkl', 'wb') as f:
            #~ pickle.dump([data_dict, comps_new, memb_probs_old, 
                #~ inc_posterior, use_box_background], f)
        

        comps_new_list = [[comp.get_mean(), comp.get_covmatrix()] for comp in comps_new]

        # Python version
        #~ memb_probs_new = expectation.expectation(data_dict, 
            #~ comps_new_list, memb_probs_old, inc_posterior=inc_posterior, 
            #~ use_box_background=use_box_background) # TODO background
        
        # C version
        st_mns = data_dict['means']
        st_covs = data_dict['covs']
        #~ gr_mns = [c.get_mean_now() for c in comps_new]
        #~ gr_covs = [c.get_covmatrix_now() for c in comps_new]
        
        # Means
        dim = len(comps_new[0].get_mean())
        gr_mns = [trace_epicyclic_orbit(comp.get_mean(), comp.get_age(), 
            dim) for comp in comps_new]

        # Covmatrices
        c = comps_new[0].get_covmatrix()
        dim1 = c.shape[0]
        dim2 = c.shape[1]
        h=1e-3 # HARDCODED...
        gr_covs = [trace_epicyclic_covmatrix(
            c.get_covmatrix(), c.get_mean(), c.get_age(), h, 
            dim1*dim2).reshape(dim1, dim2) for c in comps_new]
              
        #~ comps = [[m, co] for m, co in zip(gr_mns, gr_covs)]
        bg_lnols = data_dict['bg_lnols']
        
        print("start expectationC")
        memb_probs_new = expectationC(st_mns, st_covs, gr_mns, gr_covs, 
            bg_lnols, memb_probs_old, nstars*ncomps)
        memb_probs_new = memb_probs_new.reshape(nstars, ncomps)
        print("end expectationC")
        
        print('MEMB_PROBS_NEW DONE.')
        
        # WORKS
        #~ memb_probs_new = expectation.expectation(data_dict, comps_new, 
            #~ memb_probs_old, inc_posterior=inc_posterior, 
            #~ use_box_background=use_box_background) # TODO background

        #~ with open('output_data_from_expectation.pkl', 'wb') as f:
            #~ pickle.dump(memb_probs_new, f)

        #~ import sys
        #~ sys.exit()
            
        #~ logging.info("Membership distribution:\n{}".format(
            #~ memb_probs_new.sum(axis=0)
        #~ ))
        
        # Save new memberships
        np.save(filename_memberships_iter, memb_probs_new)


        ################################################################
        #### STORE RESULTS OF ITERATION ################################
        ################################################################
        #~ print("About to log without and with posterior lnlikelihoods") #!!!MJI
        
        # This is likelihood for all comps combined
        # get_overall_lnlikelihood computes memb_probs again, but
        # this was already computed a few lines earlier...
        print('start python expectation.get_overall_lnlikelihood')
        overall_lnlike = expectation.get_overall_lnlikelihood(data_dict, 
            comps_new_list, old_memb_probs=memb_probs_new, 
            inc_posterior=False,
            use_box_background=use_box_background) # TODO background
        print('end python expectation.get_overall_lnlikelihood')
        

        # MZ added
        np.save(filename_lnprob_and_bic_iter, [overall_lnlike])

        list_prev_comps.append(comps_new)
        list_prev_memberships.append(memb_probs_new)
        list_all_init_pos.append(all_init_pos)
        list_prev_lnlikes.append(overall_lnlike)

        comps_old = comps_new
        memb_probs_old = memb_probs_new


        ################################################################
        #### CHECK CONVERGENCE #########################################
        ################################################################
        if len(list_prev_lnlikes) < pars['min_em_iterations']:
            converged = False
        else:
            converged = lnprob_convergence(list_prev_lnlikes, 
                slice_size=pars['lnlike_convergence_slice_size'])
        
        #~ utils.log_message('Convergence status: {}'.format(converged),
            #~ symbol='-', surround=True)
        
        # NOT CONVERGED
        #~ if not converged:
            #~ logging.info('lnlike not converged')


        iter_count += 1


    logging.info("CONVERGENCE COMPLETE")
    utils.log_message('EM Algorithm finished', symbol='*')


    ####################################################################
    #### RESULTS #######################################################
    ####################################################################
    # FIND BEST ITERATION
    best_index = np.argmax(list_prev_lnlikes)
    logging.info('Best index : {} with lnlike: {}'.format(best_index, 
        list_prev_lnlikes[best_index]))


    # RESULTS
    final_best_comps = list_prev_comps[best_index]
    final_memb_probs = list_prev_memberships[best_index]


    # Likelihood
    overall_lnlike = expectation.get_overall_lnlikelihood(
            data_dict, final_best_comps, inc_posterior=False,
            use_box_background=use_box_background,
    )
                   
    logging.info("Final overall lnlikelihood: {}".format(overall_lnlike))


    ####################################################################
    #### SAVE RESULTS ##################################################
    ####################################################################
    # Create folder with final results
    utils.log_message('Storing final result', symbol='-', surround=True)
    if ncomps==1:
        folder_final = os.path.join(pars['folder_destination'], 
            str(ncomps), 'final')
    else:
        folder_final = os.path.join(pars['folder_destination'], 
            str(ncomps), pars['split_label'], 'final')
            
    if not os.path.exists(folder_final):
        try:
            os.makedirs(folder_final)
        except:
            # When doing this in parallel, more than one process might 
            # try to create this dir at the same time.
            pass


    #### SAVE MEMBERSHIPS ##############################################
    #         memb_probs_final = expectation(data_dict, best_comps, best_memb_probs,
    #                                        inc_posterior=inc_posterior)
    filename_memberships = os.path.join(folder_final, 
        pars['filename_final_memberships'])
    np.save(filename_memberships, final_memb_probs)
    logging.info('Membership final distribution:\n{}'.format(
        final_memb_probs.sum(axis=0)
    ))

    # Save membership fits file
    filename_memberships_fits = filename_memberships.replace('npy', 'fits') # TODO
    try:
        tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(
            pars['data_table'], final_memb_probs, final_best_comps,
            filename_memberships_fits, get_background_overlaps=True, 
            stellar_id_colname = pars['stellar_id_colname']
            )
    except:
        logging.info("[WARNING] Couldn't print membership.fits file. Is source_id available?")


    #### SAVE COMPONENTS ###############################################
    filename_components = os.path.join(folder_final, 
        pars['filename_final_components'])
    Component.store_raw_components(filename_components, final_best_comps)

    # Save components in fits file
    filename_components_fits = filename_components.replace('npy', 'fits')
    tabcomps = Component.convert_components_array_into_astropy_table(final_best_comps)
    tabcomps.write(filename_components_fits, overwrite=True)


    #### SAVE LIKELIHOOD AND BIC #######################################
    #~ filename_bic_list = os.path.join(pars['folder_destination'], 
        #~ pars['filename_bics_list'])
    #~ np.save(filename_bic_list, list_prev_bics)


    filename_lihelihood_bic = os.path.join(folder_final, 
        pars['filename_lihelihood_and_bic'])
    np.save(filename_lihelihood_bic, 
        (overall_lnlike))


    #### LOGGING #######################################################
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
    logging.info(50*'=')



    result = {'comps': final_best_comps, 'memb_probs': final_memb_probs}      
    return result


if __name__ == "__main__":
    filename_user_pars = sys.argv[1]
    user_pars = readparam.readParam(filename_user_pars)
    pars = default_pars.pars
    pars.update(user_pars)


    final_best_comps, final_memb_probs = run_expectmax_simple(pars)


