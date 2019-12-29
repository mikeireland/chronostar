import numpy as np
import os
import sys
import logging
from distutils.dir_util import mkpath
import random

from emcee.utils import MPIPool

sys.path.insert(0, os.path.abspath('..'))
from . import expectmax
from . import epicyclic
from . import readparam
from . import tabletool
from . import component
from . import traceorbit

def dummy_trace_orbit_func(loc, times=None):
    """
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


class NaiveFit(object):
    """
    TODO: Build a argument dictionary for em.fit_many_comps
        Many arguments can be taken straight from the fit_pars dictionary,
        so no point explicitly looking for them.
    """

    # Some default filenames
    final_comps_file = 'final_comps.npy'
    final_med_and_spans_file = 'final_med_and_spans.npy'
    final_memb_probs_file = 'final_membership.npy'

    DEFAULT_FIT_PARS = {
        'results_dir':'',

        'data_table':None,
        'init_comps_file':None, # TODO: Merge these two ?
        'skip_init_fit':False,

        # One of these two are required if initialising a run with ncomps != 1
        'init_comps':None,
        'init_memb_probs':None,

        'component':'sphere',
        'max_comp_count':20,
        'max_em_iterations':200,
        'mpi_threads':None,     # TODO: NOT IMPLEMENTED
        'use_background':True,

        'overwrite_prev_run':False,
        'burnin':500,
        'sampling_steps':1000,
        'store_burnin_chains':False,
        'ignore_stable_comps':True,
        'trace_orbit_func':traceorbit.trace_cartesian_orbit,

        'par_log_file':'fit_pars.log',
        'return_results':False,

        # REDUNDANT PARAMTERS
        'epicyclic':False,
    }

    def __init__(self, fit_pars):

        # Parse parameter file if required
        if type(fit_pars) is str:
            fit_pars = readparam.readParam(fit_pars, default_pars=self.DEFAULT_FIT_PARS)

        # Make a new dictionary, with priority given to contents of fit_pars
        self.fit_pars = {**self.DEFAULT_FIT_PARS, **fit_pars}
        assert type(self.fit_pars) is dict

        # Data prep should already have been completed, so we simply build
        # the dictionary of arrays from the astropy table
        self.data_dict = tabletool.build_data_dict_from_table(self.fit_pars['data_table'])


        # The NaiveFit approach is to assume staring with 1 component
        self.ncomps = 1

        # Import suitable component class
        if self.fit_pars['component'] == 'sphere':
            self.Component = component.SphereComponent
        elif self.fit_pars['component'] == 'ellip':
            self.Component = component.EllipComponent
        else:
            raise UserWarning('Unknown (or missing) component parametrisation')

        # Check results directory is valid
        # If path exists, make a new results_directory with a random int
        if os.path.exists(self.fit_pars['results_dir']) and \
                not self.fit_pars['overwrite_prev_run']:
            rdir = '{}_{}'.format(self.fit_pars['results_dir'].rstrip('/'),
                                  random.randint(0, 1000))
        else:
            rdir = self.fit_pars['results_dir']
        self.rdir = rdir.rstrip('/') + '/'
        mkpath(self.rdir)
        assert os.access(self.rdir, os.W_OK)

        # Log fit parameters,
        readparam.log_used_pars(self.fit_pars, default_pars=self.DEFAULT_FIT_PARS)

        # Now that results directory is set up, can set up log file
        logging.basicConfig(filename=self.rdir + 'log.log', level=logging.INFO)


    def setup(self):

        # ------------------------------------------------------------
        # -----  SETTING UP DEFAULT RUN VARS  ------------------------
        # ------------------------------------------------------------
        self.ncomps = 1

        # Set a ceiling on how long code can run for
        log_message(msg='Component count cap set to {}'.format(
                self.fit_pars['max_comp_count']),
                symbol='+', surround=True)
        log_message(msg='Iteration count cap set to {}'.format(
                self.fit_pars['max_em_iterations']),
                symbol='+', surround=True)

        # ------------------------------------------------------------
        # -----  SETTING UP RUN CUSTOMISATIONS  ----------------------
        # ------------------------------------------------------------

        # Set up trace_orbit_func
        if self.fit_pars['trace_orbit_func'] == 'dummy_trace_orbit_func':
            self.fit_pars['trace_orbit_func'] = dummy_trace_orbit_func
        elif self.fit_pars['trace_orbit_func'] == 'epicyclic':
            self.fit_pars['trace_orbit_func'] = epicyclic.trace_cartesian_orbit_epicyclic
        else:
            self.fit_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit

        # TODO: ensure this is redundant
        if self.fit_pars['epicyclic']:
            self.fit_pars['trace_orbit_func'] =  epicyclic.trace_cartesian_orbit_epicyclic
            log_message('trace_orbit: epicyclic')


        # TODO: replace init_comps_file with just init_comps and check if file
        if self.fit_pars['init_comps_file'] is not None:
            self.fit_pars['init_comps'] = self.Component.load_raw_components(
                    self.fit_pars['init_comps_file'])
            self.ncomps = len(self.fit_pars['init_comps'])
            # prev_comps = init_comps  # MZ: I think that's correct
            print('Managed to load in init_comps from file')
        else:
            self.fit_pars['init_comps'] = None
            print("'Init comps' is initialised as none")

    def build_comps_from_chains(self, run_dir):
        logging.info('Component class has been modified, reconstructing '
                     'from chain')

        comps = self.ncomps * [None]
        for i in range(self.ncomps):
            final_cdir = run_dir + 'final/comp{}/'.format(i)
            chain = np.load(final_cdir + 'final_chain.npy')
            lnprob = np.load(final_cdir + 'final_lnprob.npy')
            npars = len(self.Component.PARAMETER_FORMAT)
            best_ix = np.argmax(lnprob)
            best_pars = chain.reshape(-1, npars)[best_ix]
            comps[i] = self.Component(emcee_pars=best_pars)
        self.Component.store_raw_components(
                str(run_dir + 'final/' + self.final_comps_file),
                comps)

        return comps


    def log_score_comparison(self, prev, new):
        if new['bic'] > prev['bic']:
            logging.info("Extra component has improved BIC...")
            logging.info(
                    "New BIC: {} < Old BIC: {}".format(new['bic'], prev['bic']))
        else:
            logging.info("Extra component has worsened BIC...")
            logging.info(
                    "New BIC: {} > Old BIC: {}".format(new['bic'], prev['bic']))

        logging.info("lnlike: {} | {}".format(new['lnlike'], prev['lnlike']))
        logging.info("lnpost: {} | {}".format(new['lnpost'], prev['lnpost']))

    def build_init_comps(self, prev_comps, comp_ix, prev_med_and_spans):
        target_comp = prev_comps[comp_ix]

        assert isinstance(target_comp, self.Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run
        split_comps = target_comp.splitGroup(
            lo_age=prev_med_and_spans[comp_ix, -1, 1],
            hi_age=prev_med_and_spans[comp_ix, -1, 2])
        init_comps = list(prev_comps)
        init_comps.pop(comp_ix)
        init_comps.insert(comp_ix, split_comps[1])
        init_comps.insert(comp_ix, split_comps[0])

        # Insert init_comps into parameter dicitonary
        self.fit_pars['init_comps'] = init_comps

    def run_em_unless_loadable(self, run_dir):
        """
        Run and EM fit, but only if not loadable from a previous run
        """
        try:
            med_and_spans = np.load(run_dir + 'final/'
                                         + self.final_med_and_spans_file)
            memb_probs = np.load(
                run_dir + 'final/' + self.final_memb_probs_file)
            comps = self.Component.load_raw_components(
                    str(run_dir + 'final/' + self.final_comps_file))
            logging.info('Loaded from previous run')

            # Handle case where Component class has been modified and can't
            # load the raw components
        except AttributeError:
            comps = self.build_comps_from_chains(run_dir)

            # Handle the case where files are missing, which means we must
            # perform the fit.
        except IOError:
            comps, med_and_spans, memb_probs = \
                expectmax.fit_many_comps(data=self.data_dict,
                                         ncomps=self.ncomps, rdir=run_dir,
                                         **self.fit_pars)

        # Since init_comps and init_memb_probs are only meant for one time uses
        # we clear them to avoid any future usage
        self.fit_pars['init_comps'] = None
        self.fit_pars['init_memb_probs'] = None

        return {'comps':comps, 'med_and_spans':med_and_spans, 'memb_probs':memb_probs}


    def iter_end_log(self, best_split_ix, prev_result, new_result):
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


    def log_final_log(self, prev_result, prev_score):
        logging.info('Final best fits:')
        [logging.info(c.get_pars()) for c in prev_result['comps']]
        logging.info('Final age med and span:')
        [logging.info(row[-1]) for row in prev_result['med_and_spans']]
        logging.info('Membership distribution: {}'.format(
                prev_result['memb_probs'].sum(axis=0)))
        logging.info('Final membership:')
        logging.info('\n{}'.format(np.round(prev_result['memb_probs'] * 100)))
        logging.info('Final lnlikelihood: {}'.format(prev_score['lnlike']))
        logging.info('Final lnposterior:  {}'.format(prev_score['lnpost']))
        logging.info('Final BIC: {}'.format(prev_score['bic']))


    def calc_score(self, comps, memb_probs):
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
        lnlike = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                         comps,
                                                         # bg_ln_ols=bg_ln_ols,
                                                         )
        lnpost = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                         comps,
                                                         # bg_ln_ols=bg_ln_ols,
                                                         inc_posterior=True)

        bic = expectmax.calc_bic(self.data_dict, self.ncomps, lnlike,
                                      memb_probs=memb_probs,
                                      Component=self.Component)

        return {'bic':bic, 'lnlike':lnlike, 'lnpost':lnpost}


    def run_fit(self):
        # # ------------------------------------------------------------
        # # -----  BEGIN MPIRUN THING  ---------------------------------
        # # ------------------------------------------------------------
        #
        # # Only even try to use MPI if config file says so
        # using_mpi = self.fit_pars['run_with_mpi']
        # if using_mpi:
        #     try:
        #         pool = MPIPool()
        #         logging.info("Successfully initialised mpi pool")
        #     except:
        #         # print("MPI doesn't seem to be installed... maybe install it?")
        #         logging.info(
        #             "MPI doesn't seem to be installed... maybe install it?")
        #         using_mpi = False
        #         pool = None
        #
        # if using_mpi:
        #     if not pool.is_master():
        #         print("One thread is going to sleep")
        #         # Wait for instructions from the master process.
        #         pool.wait()
        #         sys.exit(0)
        #
        # time.sleep(5)

        print("Only one thread is master! (if not, maybe config "
              "file is missing run_with_mpi=True)")

        log_message('Beginning Chronostar run',
                    symbol='_', surround=True)

        # ------------------------------------------------------------
        # -----  RXECUTE RUN  ----------------------------------------
        # ------------------------------------------------------------

        if self.fit_pars['store_burnin_chains']:
            log_message(msg='Storing burnin chains', symbol='-')

        # Handle special case of very first run
        # Either by fitting one component (default) or by using `init_comps`
        # to initialise the EM fit.

        # If beginning with 1 component, assume all stars are members
        if self.ncomps == 1:
            init_memb_probs = np.zeros((len(self.data_dict['means']),
                                        self.ncomps + self.fit_pars['use_background']))
            init_memb_probs[:, 0] = 1.
        # Otherwise, we must have been given an init_comps, or an init_memb_probs
        #  to start things with
        else:
            assert self.fit_pars['init_comps'] is not None or self.fit_pars['init_memb_probs'] is not None
        log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                    symbol='*', surround=True)
        run_dir = self.rdir + '{}/'.format(self.ncomps)

        prev_result = self.run_em_unless_loadable(run_dir)
        prev_score = self.calc_score(prev_result['comps'], prev_result['memb_probs'])
        self.ncomps += 1

        # ------------------------------------------------------------
        # -----  EXPLORE EXTRA COMPONENT BY DECOMPOSITION  -----------
        # ------------------------------------------------------------

        # Calculate global score of fit for comparison with future fits with different
        # component counts

        # Begin iterative loop, each time trialing the incorporation of a new component
        while self.ncomps <= self.fit_pars['max_comp_count']:
            log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                        symbol='*', surround=True)

            all_results = []
            all_scores = []

            # Iteratively try subdividing each previous component
            for i, target_comp in enumerate(prev_result['comps']):
                div_label = chr(ord('A') + i)
                run_dir = self.rdir + '{}/{}/'.format(self.ncomps, div_label)
                log_message(msg='Subdividing stage {}'.format(div_label),
                            symbol='+', surround=True)
                mkpath(run_dir)

                self.build_init_comps(prev_result['comps'], comp_ix=i,
                                      prev_med_and_spans=prev_result['med_and_spans'])

                result = self.run_em_unless_loadable(run_dir)
                all_results.append(result)

                ### APPEND TO MASTER RESULTS TRACKER
                # TODO: merge into one data structure
                # TODO: two lists of dicts: results, and scores

                score = self.calc_score(result['comps'], result['memb_probs'])
                all_scores.append(score)

                logging.info(
                        'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
                        'lnpost: {}'.format(
                                div_label, all_scores[-1]['bic'],
                                all_scores[-1]['lnlike'], all_scores[-1]['lnpost'],
                        ))

            # identify the best performing decomposition
            all_bics = [score['bic'] for score in all_scores]
            best_split_ix = np.argmin(all_bics)

            new_result = all_results[best_split_ix]
            new_score = all_scores[best_split_ix]

            self.iter_end_log(best_split_ix, prev_result=prev_result, new_result=new_result)

            # Check if the fit has improved
            self.log_score_comparison(new=new_score,
                                      prev=prev_score)
            if new_score['bic'] < prev_score['bic']:
                prev_score = new_score
                prev_result = new_result

                self.ncomps += 1
                log_message(msg="Commencing {} component fit on {}{}".format(
                        self.ncomps, self.ncomps - 1,
                        chr(ord('A') + best_split_ix)), symbol='+'
                )
            else:
                logging.info("... saving previous fit as best fit to data")
                self.Component.store_raw_components(self.rdir + self.final_comps_file,
                                                    prev_result['comps'])
                np.save(self.rdir + self.final_med_and_spans_file, prev_result['med_and_spans'])
                np.save(self.rdir + self.final_memb_probs_file, prev_result['memb_probs'])
                np.save(self.rdir + 'final_likelihood_post_and_bic',
                        prev_score)

                self.log_final_log(prev_result, prev_score)
                break

            logging.info("Best fit:\n{}".format(
                    [group.get_pars() for group in prev_result['comps']]))

        if self.ncomps >= self.fit_pars['max_comp_count']:
            log_message(msg='REACHED MAX COMP LIMIT', symbol='+',
                        surround=True)

        # # TODO: using_mpi is not defined if you don't use MPI.
        # #  Try-except is not the best thing here but will do for now.
        # try:
        #     if using_mpi:
        #         pool.close()
        # except:
        #     pass

        if self.fit_pars['return_results']:
            return prev_result, prev_score
