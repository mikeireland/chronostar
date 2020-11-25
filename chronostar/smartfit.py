"""
smartfit.py

2020-11-12
Timothy Crundall

A SmartSplitFit (hereafter SSF) follows a new approach which is an
extension on that described in Crundall et al. (2019).

The initial stage (stage 1) is as follows.
SSF begins with an initial guess provided by user of an N component fit.
If no guess is provided, all provided stars are assumed to be members of
one component.

SSF will perform an Expectation Maximisation on this N component fit until
converged.

The following stage (stage 2) is iterative, and will repeat until converged.
Then SSF will test increasing the component count to N+1. This is done by
for each component out of the N existing, substituting it for 2 similar
components with slight age offsets, and running an EM fit. The result
is N separate "N+1 component" fits. The best one will be compared to the
"N component" fit using the Bayesian Information Criterion (BIC).

At this point the behavior from SSF deviates from that of NaiveFit.

SSF gathers all the "N+1 component" fits (hereafter called "splitfits")
that have a better BIC than the N component fit.

If there are no improving splitfits, then the fit has converged, and execution halts.

If there is one, then this is taken as the next best fit and we repeat stage 2
with this new fit.

If there are multiple improving splitfits, then to avoid throwing away all
splitifts but one (as would happen with NaiveFit), we generate a new potential fit.
This is Stage 2a.

This is done by inspecting each improving splitfit. We gather all the components
that came about from a component split. We include also the components that
weren't split. This gives us a set of N+k components (where k is the number of
improving splitfits). We perform an EM fit with these components to ensure
consistency.

If the BIC is still an improvement from each splitfit, then we accept this fit
as our N+k component fit and repeat from Stage 2.

If the BIC is not an improvment, then we reject the splitfit with the worst BIC
and repeat from Stage 2a.
"""
import numpy as np
import os
import sys
import logging
from distutils.dir_util import mkpath
import random
import uuid

#~ from emcee.utils import MPIPool
from multiprocessing import Pool

from multiprocessing import cpu_count

sys.path.insert(0, os.path.abspath('..'))
from . import expectmax
from . import readparam
from . import tabletool
from . import component
from . import traceorbit
from chronostar.parentfit import ParentFit

# python3 throws FileNotFoundError that is essentially the same as IOError
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

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


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5 * symbol, msg, 5 * symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50 * symbol, res, 50 * symbol)
    logging.info(res)


class SmartFit(ParentFit):
    """
        Many arguments can be taken straight from the fit_pars dictionary,
        so no point explicitly looking for them.

        Description of parameters can be found in README.md along with their
        default values and whether they are required.
    """
    def __init__(self, fit_pars):
        """
        Pass the fit parameters onto Parent Calss
        """
        super(SmartFit, self).__init__(fit_pars)

    def run_fit(self):
        """
        Perform a fit (as described in Paper I) to a set of prepared data.

        Results are outputted as two dictionaries
        results = {'comps':best_fit, (list of components)
                   'med_and_spans':median and spans of model parameters,
                   'memb_probs': membership probability array (the standard one)}
        scores  = {'bic': the bic,
                   'lnlike': log likelihood of that run,
                   'lnpost': log posterior of that run}
        """

        log_message('Beginning Chronostar run',
                    symbol='_', surround=True)

        # ------------------------------------------------------------
        # -----  EXECUTE RUN  ----------------------------------------
        # ------------------------------------------------------------

        if self.fit_pars['store_burnin_chains']:
            log_message(msg='Storing burnin chains', symbol='-')

        # ------------------------------------------------------------
        # -----  STAGE 1: ESTABLISHING INITIAL FIT         -----------
        # ------------------------------------------------------------

        # Handle special case of very first run
        # Either by fitting one component (default) or by using `init_comps`
        # to initialise the EM fit.

        # Check if not provided with init comps or membs
        if (self.fit_pars['init_comps'] is None) and (self.fit_pars['init_memb_probs'] is None):
            # NaiveFit doesn't know how to blindly intiialise runs with ncomps > 1
            assert self.ncomps == 1, 'If no initialisation set, can only accept ncomp==1'
            # If no init conditions provided, assume all stars are members and begine
            # fit with 1 component.
            init_memb_probs = np.zeros((len(self.data_dict['means']),
                                        self.ncomps + self.fit_pars[
                                            'use_background']))
            init_memb_probs[:, 0] = 1. - 1.e-10
            init_memb_probs[:, 1] = 1.e-10
            self.fit_pars['init_memb_probs'] = init_memb_probs

            log_message(msg='No initial information provided', symbol='-')
            log_message(msg='Assuming all stars are members', symbol='-')
        # Otherwise, we must have been given an init_comps, or an init_memb_probs
        #  to start things with
        else:
            log_message(msg='Initialising with init_comps or init_memb_probs with'
                            '%i components'%self.ncomps, symbol='*', surround=True)
            pass


        log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                    symbol='*', surround=True)
        run_dir = self.rdir + '{}/'.format(self.ncomps)

        prev_result = self.run_em_unless_loadable(run_dir)
        prev_score = self.calc_score(
                prev_result['comps'], prev_result['memb_probs'],
                use_box_background=self.fit_pars['use_box_background']
        )

        self.ncomps += 1

        # ------------------------------------------------------------
        # -----  STAGE 2: EXPLORE EXTRA COMPONENT BY DECOMPOSITION  --
        # ------------------------------------------------------------

        # Calculate global score of fit for comparison with future fits with different
        # component counts

        # Begin iterative loop, each time trialing the incorporation of a new component
        #
        # `prev_result` track the previous fit, which is taken to be
        # the best fit so far
        #
        # As new fits are acquired, we call them `new_result`.
        # The new fits are compared against the previous fit, and if determined to
        # be an improvement, they are taken as the best fit, and are renamed to
        # `prev_result`


        stage_2_ncomps = 2
        while stage_2_ncomps <= self.fit_pars['max_comp_count']:
            log_message(msg='FITTING {} COMPONENT'.format(stage_2_ncomps),
                        symbol='*', surround=True)

            all_results = []
            all_scores = []

            # Iteratively try subdividing each previous component
            # target_comp is the component we will split into two.
            # This will make a total of ncomps (the target comp split into 2,
            # plus the remaining components from prev_result['comps']
            for i, target_comp in enumerate(prev_result['comps']):
                div_label = chr(ord('A') + i)
                run_dir = self.rdir + '{}/{}/'.format(stage_2_ncomps, div_label)
                log_message(msg='Subdividing stage {}'.format(div_label),
                            symbol='+', surround=True)
                mkpath(run_dir)

                self.fit_pars['init_comps'] = self.build_init_comps(
                        prev_result['comps'], split_comp_ix=i,
                        prev_med_and_spans=prev_result['med_and_spans'],
                        memb_probs = prev_result['memb_probs'])
                self.ncomps = len(self.fit_pars['init_comps'])

                result = self.run_em_unless_loadable(run_dir)
                all_results.append(result)

                score = self.calc_score(
                        result['comps'], result['memb_probs'],
                        use_box_background=self.fit_pars['use_box_background']
                )
                all_scores.append(score)

                logging.info(
                        'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
                        'lnpost: {}'.format(
                                div_label, all_scores[-1]['bic'],
                                all_scores[-1]['lnlike'], all_scores[-1]['lnpost'],
                        ))

            # ------------------------------------------------------------
            # -----  STAGE 2a: COMBINE RESULTS OF EACH GOOD SPLIT  -------
            # ------------------------------------------------------------


            # identify all the improving splits
            all_bics = np.array([score['bic'] for score in all_scores])
            improvsplit_mask = all_bics < prev_score['bic']

            better_split_labels =' '.join([chr(ord('A') + i)
                                           for i, flag in enumerate(improvsplit_mask) if flag])
            print("Found better splits: %s"%better_split_labels)
            log_message(msg='Found better splits: %s'%better_split_labels,
                        symbol='-', surround=True)
            log_message(msg='With bics: %s'%[s['bic'] for s in all_scores])
            log_message(msg='Compared with prev_bic: %s'%prev_score['bic'])

            converged_2a = False

            # If no better splits found, this loop is skipped, and `prev_result`
            # is simply left alone, to be taken as best fit at the end
            while not converged_2a and sum(improvsplit_mask) > 0:
                # if len(improvsplit_mask[0]) == 1:
                if sum(improvsplit_mask) == 1:
                    # if only one improve split, then that is our new fit.
                    best_split_ix = np.where(improvsplit_mask)[0][0]
                    new_result = all_results[best_split_ix]
                    new_score = all_scores[best_split_ix]
                    converged_2a = True
                    self.iter_end_log(best_split_ix, prev_result=prev_result, new_result=new_result)
                    prev_result = new_result
                    prev_score = new_score
                    stage_2_ncomps += 1

                elif sum(improvsplit_mask) > 1:
                    # Construct set of components including each split pair from an
                    # improving split, and also all components that weren't split
                    # (arbitrarily taken from prev_fit)

                    new_comps = []
                    for i, orig_comp in enumerate(prev_result['comps']):
                        if improvsplit_mask[i]:
                            # then we managed to split it, so we take those comps
                            # So we get the components from the i fit,
                            # and the succesfully splitted ones will the the ith, and the ith+1
                            new_comps.append(all_results[i]['comps'][i])
                            new_comps.append(all_results[i]['comps'][i+1])
                        else:
                            # If we didn't manage to split it, take it's original form
                            new_comps.append(prev_result['comps'][i])

                    # We now have a set of N+k components, where k is the number of
                    # improving splits

                    # We run a consolidating fit, and see if BIC is improved
                    # label includes all split comps?
                    run_label = ''.join([chr(ord('a') + i) for i, flag in enumerate(improvsplit_mask) if flag])
                    run_dir = self.rdir + '{}/{}/'.format(stage_2_ncomps,
                                                          run_label)
                    log_message(msg='Consolidating %s/%s'%(stage_2_ncomps, run_label),
                                symbol='+', surround=True)
                    mkpath(run_dir)

                    self.fit_pars['init_comps'] = new_comps
                    # TODO: now that ncomps is fluctuating here, it seems unwise
                    #       to have it as its own parameter.... need to refactor this
                    self.ncomps = len(new_comps)

                    new_result = self.run_em_unless_loadable(run_dir)
                    # all_results.append(result)

                    new_score = self.calc_score(
                            result['comps'], result['memb_probs'],
                            use_box_background=self.fit_pars['use_box_background']
                    )
                    # all_scores.append(score)

                    logging.info(
                            'Consolidation {} finished with \nBIC: {}\n'
                            'lnlike: {}\n'
                            'lnpost: {}'.format(
                                    run_label, new_score['bic'],
                                    new_score['lnlike'],
                                    new_score['lnpost'],
                            ))


                    # Check if BIC is better than all previous
                    if new_score['bic'] < np.min(all_bics):
                        # If improved, then accept as new fit
                        # TODO: rename new_comps to something more intutitive, as `new_comps` here, are not the comps just ac
                        self.iter_end_log(best_split_ix, prev_result=prev_result,
                                          new_result=new_result)
                        prev_score = new_score
                        prev_result = new_result
                        converged_2a = True
                        stage_2_ncomps = len(prev_result['comps'])
                    else:
                        # Kick out worst split, and redo
                        worst_bic = -np.inf
                        worst_bic_ix = -1
                        for i in range(len(all_scores)):
                            if improvsplit_mask[i] and all_scores[i]['bic'] > worst_bic:
                                worst_bic_ix = i
                                worst_bic = all_scores[i]['bic']
                        improvsplit_mask[worst_bic_ix] = False
                        log_message(
                                msg='Consolidation run %s/%s failed, kicking out %s and retrying.'%(
                                    stage_2_ncomps, run_label, chr(ord('A') + worst_bic_ix)),
                                symbol='-'
                        )

                else:
                    raise UserWarning('Should not be here')



#             new_result = all_results[best_split_ix]
#             new_score = all_scores[best_split_ix]

            # self.iter_end_log(best_split_ix, prev_result=prev_result, new_result=new_result)

#             # Check if the fit has improved
#             self.log_score_comparison(new=new_score,
#                                       prev=prev_score)
#             if new_score['bic'] < prev_score['bic']:
#                 prev_score = new_score
#                 prev_result = new_result
#
#                 self.ncomps += 1
#                 log_message(msg="Commencing {} component fit on {}{}".format(
#                         self.ncomps, self.ncomps - 1,
#                         chr(ord('A') + best_split_ix)), symbol='+'
#                 )
#             else:
#                 self.write_results_to_file(prev_result, prev_score)


            logging.info("Best fit:\n{}".format(
                    [group.get_pars() for group in prev_result['comps']]))

        if self.ncomps >= self.fit_pars['max_comp_count']:
            log_message(msg='REACHED MAX COMP LIMIT', symbol='+',
                        surround=True)

        return prev_result, prev_score
