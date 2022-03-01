"""
MZ
Component manager: run run_em.py from this script. This script manages
components and decides what fits need to be done with run_em.
It reads in the data, splits comps and prepares input data to the
run_em.py.
It also decides what model is the best.
"""

import warnings
warnings.filterwarnings("ignore")
print('fastfit: all warnings suppressed.')

import numpy as np
import os
import sys
import logging
from distutils.dir_util import mkpath
import random
#~ import uuid
#~ import subprocess

from multiprocessing import cpu_count

sys.path.insert(0, os.path.abspath('..'))
#~ from chronostar import expectmax2 as expectmax
#~ from chronostar import expectmax # replaced by C modules
from chronostar import readparam
from chronostar import tabletool
from chronostar import component
from chronostar.component import SphereComponent
from chronostar import utils
from chronostar import default_pars

from chronostar import traceorbitC
import run_em

try:
    from chronostar._overall_likelihood import get_overall_lnlikelihood_for_fixed_memb_probs
except ImportError:
    print("C IMPLEMENTATION OF overall_likelihood NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?
    
# SPLIT COMPONENTS HERE:
def build_init_comps(prev_comps=None, split_comp_ix=0, memb_probs=None,
    Component=None, data_dict=None, split_group_method='age'):
    """
    Given a list of converged components from a N component fit, generate
    a list of N+1 components with which to initialise an EM run.

    This is done by taking the target component, `prev_comps[comp_ix]`,
    replacing it in the list of comps, by splitting it into two components
    with a lower and higher age,

    Parameters
    ----------
    prev_comps : [N] list of Component objects
        List of components from the N component fit
    split_comp_ix : int
        The index of component which is to be split into two
    prev_med_and_spans : [ncomps,npars,3] np.array
        The median and spans of

    Return
    ------
    init_comps: [N+1] list of Component objects

    Side effects
    ------------
    Updates self.fit_pars['init_comps'] with a [N+1] list of Component
    objects

    Edit history
    ------------
    2020-11-14 TC: replaced explicit check for emcee vs Nelder-mead when
    trying to use prev_med_and_spans. This enables emcee runs to continue
    on from Nelder-mead runs, and hopefully generalises this section to
    be agnostic of optimisation method
    """
    target_comp = prev_comps[split_comp_ix]

    # Decompose and replace the ith component with two new components
    # by using the 16th and 84th percentile ages from previous run

    if split_group_method=='age':
        age = target_comp.get_age()
        lo_age = 0.8*age
        hi_age = 1.2*age
        split_comps = target_comp.split_group_age(lo_age=lo_age, 
            hi_age=hi_age)
    elif split_group_method=='spatial':
        split_comps = target_comp.split_group_spatial(data_dict,
            memb_probs[:,split_comp_ix])

    init_comps = list(prev_comps)
    init_comps.pop(split_comp_ix)
    init_comps.insert(split_comp_ix, split_comps[1])
    init_comps.insert(split_comp_ix, split_comps[0])

    return init_comps


def calc_bic(data, ncomps, lnlike, memb_probs=None, 
    Component=SphereComponent):
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
    # 2020/11/15 TC: removed this...
#     if memb_probs is not None:
#         nstars = np.sum(memb_probs[:, :ncomps])
#     else:
    nstars = len(data['means'])
    ncomp_pars = len(Component.PARAMETER_FORMAT)
    n = nstars * 6                      # 6 for phase space origin
    k = ncomps * (ncomp_pars)           # parameters for each component model
                                        #  -1 for age, +1 for amplitude
    return np.log(n)*k - 2 * lnlike


def calc_score(data_dict, comps, memb_probs, Component,
    use_box_background=False):
    """
    Calculate global score of fit for comparison with future fits with different
    component counts

    Parameters
    ----------
    :param comps:
    :param memb_probs:
    :return:

    TODO: Establish relevance of bg_ln_ols
    """
    
    ncomps = len(comps)

    # python
    #~ lnlike = expectmax.get_overall_lnlikelihood(data_dict, comps,
        #~ old_memb_probs=memb_probs, use_box_background=use_box_background)

    #~ print('lnlikeP', lnlike)

    #~ lnpost = expectmax.get_overall_lnlikelihood(data_dict, comps,
        #~ old_memb_probs=memb_probs, use_box_background=use_box_background,
        #~ inc_posterior=True)

    #~ bic = expectmax.calc_bic(data_dict, ncomps, lnlike,
        #~ memb_probs=memb_probs, Component=Component)


    ####################################################################
    #### DATA FOR C MODULES ############################################
    ####################################################################
    gr_mns, gr_covs = traceorbitC.get_gr_mns_covs_now(comps)
    st_mns = data_dict['means']
    st_covs = data_dict['covs']
    bg_lnols = data_dict['bg_lnols']
    
    # For some reason, bg_ols in C only work this way now. They worked before from data_dict... A mystery! data_dict now produces values +/-1e+240 or similar.
    filename_tmp = 'bgols_tmp.dat'
    np.savetxt(filename_tmp, bg_lnols)
    bg_lnols = np.loadtxt(filename_tmp)
    print('run_em: bg_lnols read from a txt file!')
    
    lnlike = get_overall_lnlikelihood_for_fixed_memb_probs(
        st_mns, st_covs, gr_mns, gr_covs, bg_lnols, memb_probs) # TODO background
    
    #~ print('lnlikeC', lnlike)

    # use lnlikeC
    bic = calc_bic(data_dict, ncomps, lnlike, memb_probs=memb_probs, 
        Component=Component)
    
    #~ lnpost = expectmax.get_overall_lnlikelihood(data_dict, comps,
        #~ old_memb_probs=memb_probs, use_box_background=use_box_background,
        #~ inc_posterior=True)  
          
    lnpost=np.nan # don't need it
        
    
    # 2020/11/16 TC: handling the case for a bad bic.
    # This comes up for the initial 1 component fit with box background
    # because I haven't thought of a general way to initialise memberships
    # that doesn't yield 0 background members.
    if np.isnan(bic):
        logging.info('Warning, bic was NaN')
        bic = np.inf

    return {'bic': bic, 'lnlike': lnlike, 'lnpost': lnpost}


def log_score_comparison(prev, new):
    """
    Purely a logging helper function.
    Log BIC comparisons.

    Parameters
    ----------
    prev: dict
        A dictinoary of scores from the previous run with the following entries
        - bic: the Bayesian Information Criterion
        - lnlike : the log likelihood
        - lnpost : the log posterior
    new: dict
        A dictinoary of scores from the new run, with identical entries as
        `prev`

    Result
    ------
    None
    """
    if new['bic'] < prev['bic']:
        logging.info("Extra component has improved BIC...")
        logging.info(
                "New BIC: {} < Old BIC: {}".format(new['bic'], prev['bic']))
    else:
        logging.info("Extra component has worsened BIC...")
        logging.info(
                "New BIC: {} > Old BIC: {}".format(new['bic'], prev['bic']))

    logging.info("lnlike: {} | {}".format(new['lnlike'], prev['lnlike']))
    logging.info("lnpost: {} | {}".format(new['lnpost'], prev['lnpost']))


def write_results_to_file(prev_result, prev_score, pars):
    """
    Write final results of the fit to the files

    TODO: write fits file with id and memberships
    TODO: ascii file with components today
    """
    logging.info("... saving previous fit as best fit to data")
    
    #### Components ####################################################
    filename_final_comps = os.path.join(pars['folder_destination'], 
        pars['filename_final_components'])
    
    # npy
    Component.store_raw_components(filename_final_comps, 
        prev_result['comps'])
    
    # Ascii
    filename_final_comps_ascii = filename_final_comps.replace('.npy', 
        '.txt')
    Component.store_components_ascii(filename_final_comps_ascii,
        prev_result['comps'], overwrite=pars['overwrite_prev_run'])

    # Fits
    tabcomps = Component.convert_components_array_into_astropy_table(prev_result['comps'])

    filename_final_comps_fits = filename_final_comps.replace('.npy', 
        '.fits')

    tabcomps.write(filename_final_comps_fits, 
        overwrite=pars['overwrite_prev_run'])



    #### Memberships ###################################################
    filename_final_memberships = os.path.join(pars['folder_destination'],
        pars['filename_final_memberships'])
    
    # npy
    np.save(filename_final_memberships, prev_result['memb_probs'])

    # Fits
    filename_final_memberships_fits = filename_final_memberships.replace(
        '.npy', '.fits')
    try:
        tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(
            pars['data_table'], 
            prev_result['memb_probs'], 
            prev_result['comps'], 
            filename_final_memberships_fits, 
            get_background_overlaps=True, 
            stellar_id_colname=pars['stellar_id_colname'], 
            overwrite_fits=pars['overwrite_prev_run'])
    except:
        logging.info("[WARNING] Couldn't print membership.fits file. Check column id.")

    
    
    #### Likelihood and BIC ############################################
    filename_final_likelihood_and_bic =\
        os.path.join(pars['folder_destination'], 
            pars['filename_final_lnprob_and_bic'])
    np.save(filename_final_likelihood_and_bic, prev_score)


    #### Final log #####################################################
    log_final_log(prev_result, prev_score, pars)


def iter_end_log(best_split_ix, prev_result, new_result): # TODO
    """
    This is not working. chr(ord(A))...
    """
    logging.info("Selected {} as best decomposition".format(
        chr(ord('A') + best_split_ix)))
    logging.info(
        "Turned\n{}".format(prev_result['comps'][best_split_ix].get_pars()))
    logging.info('with {} members'.format(
        prev_result['memb_probs'].sum(axis=0)[best_split_ix]))
    logging.info("into\n{}\n&\n{}".format(
        new_result['comps'][best_split_ix].get_pars(),
        new_result['comps'][best_split_ix + 1].get_pars(),
    ))
    logging.info('with {} and {} members'.format(
        new_result['memb_probs'].sum(axis=0)[best_split_ix],
        new_result['memb_probs'].sum(axis=0)[best_split_ix + 1],
    ))
    logging.info("for an overall membership breakdown\n{}".format(
        new_result['memb_probs'].sum(axis=0)
    ))


def log_final_log(prev_result, prev_score, pars):
    logging.info('Final best fits:')
    [logging.info(c.get_pars()) for c in prev_result['comps']]
    logging.info('Final age med and span:')
    logging.info('PRINT TODO')
    #~ if pars['optimisation_method']=='emcee':
        #~ [logging.info(row[-1]) for row in prev_result['med_and_spans']]
    logging.info('Membership distribution: {}'.format(
            prev_result['memb_probs'].sum(axis=0)))
    logging.info('Final membership:')
    logging.info('\n{}'.format(np.round(prev_result['memb_probs'] * 100)))
    logging.info('Final lnlikelihood: {}'.format(prev_score['lnlike']))
    logging.info('Final lnposterior:  {}'.format(prev_score['lnpost']))
    logging.info('Final BIC: {}'.format(prev_score['bic']))
    logging.info('#########################')
    logging.info('### END #################')
    logging.info('#########################')


########################################################################
#### PARAMETERS ########################################################
########################################################################
filename_user_pars = sys.argv[1]
user_pars = readparam.readParam(filename_user_pars)
pars = default_pars.pars
pars.update(user_pars)

pars['filename_pars_log'] = os.path.join(pars['folder_destination'], 
    pars['par_log_file'])

# Log fit parameters
readparam.log_used_pars(pars, default_pars=default_pars.pars)


if pars['component'].lower() == 'sphere':
    Component = component.SphereComponent
elif pars['component'].lower() == 'ellip':
    Component = component.EllipComponent


########################################################################
#### INPUT DATA ########################################################
########################################################################
# Data prep should already have been completed, so we simply build
# the dictionary of arrays from the astropy table
data_dict = tabletool.build_data_dict_from_table(pars['data_table'],
    historical=pars['historical_colnames'])


########################################################################
#### OUTPUT DESTINATION ################################################
########################################################################
# Folder: destination with results
# If path exists, make a new results_directory with a random int
if os.path.exists(pars['folder_destination']):
    folder_destination = os.path.join(pars['folder_destination'],
        str(random.randint(0, 1000)))
    pars['folder_destination'] = folder_destination

mkpath(pars['folder_destination'])

# tmp folder to store tmp files
folder_tmp = os.path.join(pars['folder_destination'], pars['folder_tmp'])
mkpath(folder_tmp)


########################################################################
#### LOGGING ###########################################################
########################################################################
# Logging filename
filename_log = os.path.join(pars['folder_destination'], 
    pars['filename_log'])
logging.basicConfig(filename=filename_log, level=logging.INFO)

# Make some logs
utils.log_message(msg='Component count cap set to {}'.format(
    pars['max_comp_count']), symbol='+', surround=True)
utils.log_message(msg='Iteration count cap set to {}'.format(
    pars['max_em_iterations']), symbol='+', surround=True)


########################################################################
#### INITIAL COMPONENTS, MEMBERSHIPS AND NCOMPS ########################
########################################################################
ncomps = 1

# Initial components
filename_init_comps = pars['filename_init_comps']
if filename_init_comps is not None and os.path.exists(filename_init_comps):
    init_comps = Component.load_raw_components(filename_init_comps)
    ncomps = len(init_comps)
    print('Managed to load in %d init_comps from file'%ncomps)
else:
    init_comps = None

# Initial membership probabilities
filename_init_memb_probs = pars['filename_init_memb_probs']
if filename_init_memb_probs is not None and os.path.exists(filename_init_memb_probs):
    init_memb_probs = np.load(filename_init_memb_probs)
    print('Managed to load in %d init_memb_probs from file'%len(init_memb_probs))
else:
    init_memb_probs = None


# Check if ncomps and init_memb_probs.shape[1] match!!!!
if init_comps is not None and init_memb_probs is not None:
    assert len(init_comps)==init_memb_probs.shape[1]-1
    # What happens if they are not? [background component...]
    
if init_comps is None and init_memb_probs is not None:
    ncomps = init_memb_probs.shape[1]-1 # remove background
    init_comps = None

print('ncomps: %d'%ncomps)

########################################################################
#### ESTABLISHING INITIAL FIT ##########################################
########################################################################
"""
Handle special case of very first run: This is here either for 
ncomps=1 or if initialised by more comps/membership probs, Chronostar
needs to build and fit these in order to continue with further splits.

Either by fitting one component (default) or by using `init_comps`
to initialise the EM fit.
"""
utils.log_message('Beginning Chronostar run', symbol='_', surround=True)
utils.log_message(msg='FITTING {} COMPONENT'.format(ncomps),
    symbol='*', surround=True)

pars_tmp = pars
pars_tmp['ncomps'] = ncomps
#~ pars_tmp['split_label'] = 'initial'
pars_tmp['split_label'] = ''

prev_result = run_em.run_expectmax_simple(pars_tmp, data_dict=data_dict,
    init_comps=init_comps, init_memb_probs=init_memb_probs)

prev_score = calc_score(data_dict, prev_result['comps'], 
    prev_result['memb_probs'], Component,
    use_box_background=pars['use_box_background'])

print('prev_score')
print(prev_score)

ncomps += 1

########################################################################
#### EXPLORE EXTRA COMPONENT BY DECOMPOSITION ##########################
########################################################################
"""
Calculate global score of fit for comparison with future fits with 
different component counts

Begin iterative loop, each time trialing the incorporation of a new 
component

`prev_result` track the previous fit, which is taken to be the best fit 
so far

As new fits are acquired, we call them `new_result`.
The new fits are compared against the previous fit, and if determined to
be an improvement, they are taken as the best fit, and are renamed to
`prev_result`
"""

global_bics = []
while ncomps <= pars['max_comp_count']:
    utils.log_message(msg='FITTING {} COMPONENT'.format(ncomps),
        symbol='*', surround=True)


    ####################################################################
    #### COMPUTE ALL SPLITS FOR A MODEL WITH NCOMPS ####################
    ####################################################################
    all_results = []
    all_scores = []

    # Iteratively try subdividing each previous component
    # target_comp is the component we will split into two.
    # This will make a total of ncomps (the target comp split into 2,
    # plus the remaining components from prev_result['comps']
    
    for i, target_comp in enumerate(prev_result['comps']):
        ################################################################
        #### INITIALISE ################################################
        ################################################################
        #~ print(pars['init_comps'])
        #~ ncomps = len(pars['init_comps'])
        
        # Folders for splits are named S1... rather than letters alone
        split_label = 'S%d'%(i+1)
        
        # OUTPUT FOLDER
        folder_split = os.path.join(pars['folder_destination'], 
            str(ncomps), split_label)
        
        utils.log_message(msg='Subdividing stage {}'.format(split_label),
            symbol='+', surround=True)
        mkpath(folder_split)

        #### PREPARE INITIAL COMPONENTS BY SPLITTING THEM ##############            
        init_comps = build_init_comps(prev_comps=prev_result['comps'], 
            split_comp_ix=i, memb_probs=prev_result['memb_probs'], 
            Component=Component, data_dict=data_dict, 
            split_group_method=pars['split_group_method'])
        
        # Save components to the file so EM algorithm can read them
        filename_comps_split = os.path.join(folder_tmp, 
            'init_comps_%d_%s.npy'%(ncomps, split_label))
        Component.store_raw_components(filename_comps_split, init_comps)

        #### PREPARE EM PARS FILE FOR THIS SPLIT #######################
        pars_tmp = pars
        pars_tmp['ncomps'] = ncomps
        pars_tmp['split_label'] = split_label
        pars_tmp['filename_init_comps'] = filename_comps_split
        
        
        ################################################################
        #### FIT: EM ALGORITHM #########################################
        ################################################################        
        result = run_em.run_expectmax_simple(pars_tmp, 
            data_dict=data_dict, init_comps=init_comps)    


        ################################################################
        #### STORE RESULTS #############################################
        ################################################################  
        all_results.append(result)

        score = calc_score(data_dict, result['comps'], 
            result['memb_probs'], Component,
            use_box_background=pars['use_box_background'])
        all_scores.append(score)
        print('score')
        print(score)

        logging.info(
            'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
            'lnpost: {}'.format(split_label, all_scores[-1]['bic'],
            all_scores[-1]['lnlike'], all_scores[-1]['lnpost']))


    ####################################################################
    #### ALL SPLITS DONE. FIND THE BEST ONE ############################
    ####################################################################
    # Identify all the improving splits
    all_bics = np.array([score['bic'] for score in all_scores])
    best_split_ix = np.nanargmin(all_bics)

    new_result = all_results[best_split_ix]
    new_score = all_scores[best_split_ix]

    #~ self.iter_end_log(best_split_ix, prev_result=prev_result, 
        #~ new_result=new_result)

    ####################################################################
    #### CONVERGENCE: DO WE HAVE THE MODEL WITH OPTIMAL NUMBER OF ######
    #### COMPONENTS OR DO WE NEED TO INTRODUCE ANOTHER COMPONENT? ######
    ####################################################################
    # Check if the fit has improved
    log_score_comparison(new=new_score, prev=prev_score)
    
    
    print('scores in all possible splits')
    for s in all_scores:
        print(s)
    print('')
    
    
    print('all BICs so far')
    print(all_bics)
           
    print('SCORE COMPARISON FOR CONVERGENCE', new_score['bic'], 
        prev_score['bic'], 'Does new BIC improve the model?', new_score['bic'] < prev_score['bic'])
    #### NOT CONVERGED YET, CONTINUE WITH SPLITTING ####################
    if new_score['bic'] < prev_score['bic']:
        print('Not converged. Continue')
        prev_score = new_score
        prev_result = new_result
        ncomps += 1
        
        global_bics.append(new_score['bic'])
        
        utils.log_message(
            msg="Commencing {} component fit on {}{}".format(
            ncomps, ncomps - 1,
            chr(ord('A') + best_split_ix)), symbol='+'
        )
    
    #### CONVERGED. WRITE RESULTS AND EXIT #############################
    else:
        print('CONVERGED. EXIT THE LOOP')
        print('global bics')
        print(global_bics)
        print('last BIC', new_score['bic'])
        # WRITING THE FINAL RESULTS INTO FILES
        # SAVE prev_result rather than new_result because prev_result
        # is optimal while new_result has worsened the score.
        write_results_to_file(prev_result, prev_score, pars)
        
        #~ fig=plt.figure()
        #~ ax=fig.add_subplot(111)
        #~ ax.plot(range(len(all_bics)), all_bics)
        #~ plt.savefig('all_bics.png')
        
        # Terminate the loop
        break

    logging.info("Best fit:\n{}".format(
            [group.get_pars() for group in prev_result['comps']]))




# FINAL LOGGING
if ncomps >= pars['max_comp_count']:
    utils.log_message(msg='REACHED MAX COMP LIMIT', symbol='+', 
    surround=True)

utils.log_message(msg='END', symbol='+', surround=True)
utils.log_message(msg='####################', symbol='+', surround=True)
print('END')

########################################################################
# END
