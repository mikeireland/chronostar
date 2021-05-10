"""
naivefit.py


A NaiveFit follows the approach described in Crundall et al. (2019).

NaiveFit begins with an initial guess provided by user of an N component fit.
If no guess is provided, all provided stars are assumed to be members of one
component.

NaiveFit will perform an Expectation Maximisation on this N component fit until
converged.

Then NaiveFit will test increasing the compoennt count to N+1. This is done by
for each component out of the N existing, substituting it for 2 similar
components with slight age offsets, and running an EM fit. The result
is N separate "N+1 component" fits. The best one will be compared to the
"N component" fit using the Bayesian Information Criterion (BIC). If the
BIC has improved, this "N+1 component fit" will be taken as the best fit so far.

This process iterates until adding a component fails to yield a better fit.
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

#ACW: put these into a helper module /start
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
#ACW: /end


class NaiveFit(ParentFit):
    def __init__(self, fit_pars):
        """
        Parameters
        ----------
        fit_pars : str -or- dictionary
            If a string, `fit_pars` should be a path to a parameter file which
            can be parsed by readparam.readParam, to construct a dictionary.
            Alternatively, an actual dictionary can be passed in. See README.md
            for a description of parameters.
        """
        super(NaiveFit, self).__init__(fit_pars)

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

        # ACW: Make this a function (~50 lines)
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

        # MZ: just testing. Delete after if works
        print("self.fit_pars['init_memb_probs']", self.fit_pars['init_memb_probs'])
        print("self.fit_pars['init_comps']", self.fit_pars['init_comps'])




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
            best_split_ix = np.nanargmin(all_bics)

            new_result = all_results[best_split_ix]
            new_score = all_scores[best_split_ix]

            self.iter_end_log(best_split_ix, prev_result=prev_result, new_result=new_result)

            # Check if the fit has improved
            self.log_score_comparison(new=new_score,
                                      prev=prev_score)
            if new_score['bic'] < prev_score['bic']:
                prev_score = new_score
                prev_result = new_result

                stage_2_ncomps += 1
                log_message(msg="Commencing {} component fit on {}{}".format(
                        self.ncomps, self.ncomps - 1,
                        chr(ord('A') + best_split_ix)), symbol='+'
                )
            else:
                # WRITING THE FINAL RESULTS INTO FILES
                self.write_results_to_file(prev_result, prev_score)
                break

            logging.info("Best fit:\n{}".format(
                    [group.get_pars() for group in prev_result['comps']]))

        if stage_2_ncomps >= self.fit_pars['max_comp_count']:
            log_message(msg='REACHED MAX COMP LIMIT', symbol='+',
                        surround=True)

        return prev_result, prev_score

