"""
parentfit.py

ParentFit is the parent class for various Fit implementations.
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


class ParentFit(object):
    """
        Many arguments can be taken straight from the fit_pars dictionary,
        so no point explicitly looking for them.

        Description of parameters can be found in README.md along with their
        default values and whether they are required.
    """
    OPTIMISATION_METHODS = ['emcee', 'Nelder-Mead']

    # Internal filestems that Chronostar uses to store results throughout a fit
    # Should not be changed, otherwise Chronostar may struggle to retreive progress
    # from previous fits.
    final_comps_file = 'final_comps.npy'
    final_med_and_spans_file = 'final_med_and_spans.npy'
    final_memb_probs_file = 'final_membership.npy'


    # For detailed description of parameters, see the main README.md file
    # in parent directory.
    DEFAULT_FIT_PARS = {
        'results_dir':'',

        # Output from dataprep, XYZUVW data, plus background overlaps
        # Can be a filename to a astropy table, or an actual table
        'data_table':None,

        # Whether to look for dX, .. c_XY or X_error, .. corr_X_Y in
        # the column names
        'historical_colnames':False,

        # Column name for stellar IDs. This is used at the end when generating
        # final fits table with IDs and membership probabilities.
        # This is optional.
        'stellar_id_colname': None,

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
        # TODO: implement this in a way that info can be passed in from text file
        #       e.g. a path to a file name
        #       for now, can only be used from within a script, i.e. given a numpy
        #       array object
        'init_memb_probs':None,

        # Provide a string name that corresponds to a ComponentClass
        # An actual Component Class will be inserted into the paramter
        # dictionary to be passed into expectmax
        'component':'sphere',

        'max_comp_count':20,
        'max_em_iterations':200,

        # Convergence criteria for when a fit_many_comps run has converged
        'bic_conv_tol':0.1, # TODO: NOT TESTED!
        'nthreads':1,     # TODO: NOT IMPLEMENTED
        'use_background':True,
        'use_box_background':False,

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

        # MZ
        # Specify what optimisation method in the maximisation step of
        # the EM algorithm to use. Default: emcee. Also available:
        # In principle any method from scipy.optimise.minimise, but
        # here we recommend Nelder-Mead (because the initialisation
        # with any additional arguments, e.g. Jacobian etc. is not
        # implemented in Chronostar).
        # 'emcee' | 'Nelder-Mead'
        'optimisation_method': 'emcee',

        # Optimise components in parallel in expectmax.maximise.
        'nprocess_ncomp': False,

        # Overwrite final results in a fits file
        'overwrite_fits': False,

        # How to split group: in age or in space?
        'split_group': 'age',

        'par_log_file':'fit_pars.log',
    }

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
        # Parse parameter file if required
        if type(fit_pars) is str:
            fit_pars = readparam.readParam(fit_pars, default_pars=self.DEFAULT_FIT_PARS)

        # Make a new dictionary, with priority given to contents of fit_pars
        self.fit_pars = dict(self.DEFAULT_FIT_PARS)
        self.fit_pars.update(fit_pars)
        assert type(self.fit_pars) is dict

        try:
            assert np.isin(self.fit_pars['optimisation_method'], self.OPTIMISATION_METHODS)
        except AssertionError:
            raise UserWarning('%s is not in %s\nMake sure no quotation marks in par file'%(
                self.fit_pars['optimisation_method'], self.OPTIMISATION_METHODS
            ))

        # MZ: Make sure 'par_log_file' is written into the results folder
        self.fit_pars['par_log_file'] = os.path.join(self.fit_pars['results_dir'], self.fit_pars['par_log_file'])

        # Data prep should already have been completed, so we simply build
        # the dictionary of arrays from the astropy table
        self.data_dict = tabletool.build_data_dict_from_table(self.fit_pars['data_table'],
                                                              historical=self.fit_pars['historical_colnames'])

        # The NaiveFit approach is to assume starting with 1 component
        self.ncomps = 1

        # Import suitable component class
        if self.fit_pars['component'] == 'sphere':
            self.Component = component.SphereComponent
            self.fit_pars['Component'] = component.SphereComponent
        elif self.fit_pars['component'] == 'ellip':
            self.Component = component.EllipComponent
            self.fit_pars['Component'] = component.EllipComponent
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
 
        # Make some logs about how many iterations (+ other stuff) code can run for
        log_message(msg='Component count cap set to {}'.format(
                self.fit_pars['max_comp_count']),
                symbol='+', surround=True)
        log_message(msg='Iteration count cap set to {}'.format(
                self.fit_pars['max_em_iterations']),
                symbol='+', surround=True)
        print('printed')

        # Check nthreads does not exceed hardware
        if self.fit_pars['nthreads'] > cpu_count() - 1:
            raise UserWarning('Provided nthreads exceeds cpu count on this machine. '
                              'Rememeber to leave one cpu free for master thread!')

        # MZ: If nthreads>1: create an MPIPool
        if self.fit_pars['nthreads']>1:
            #self.pool = MPIPool()
            log_message('pool = Pool(nthreads) = pool(%d)'%self.fit_pars['nthreads'])
            self.fit_pars['pool']=Pool(self.fit_pars['nthreads'])
        else:
            self.pool = None

        # ------------------------------------------------------------
        # -----  SETTING UP RUN CUSTOMISATIONS  ----------------------
        # ------------------------------------------------------------

        # Set up trace_orbit_func
        if self.fit_pars['trace_orbit_func'] == 'dummy_trace_orbit_func':
            self.fit_pars['trace_orbit_func'] = dummy_trace_orbit_func
        elif self.fit_pars['trace_orbit_func'] == 'epicyclic':
            log_message('trace_orbit: epicyclic')
            self.fit_pars['trace_orbit_func'] = traceorbit.trace_epicyclic_orbit
        else:
            self.fit_pars['trace_orbit_func'] = traceorbit.trace_cartesian_orbit

        if type(self.fit_pars['init_comps']) is str:
            self.fit_pars['init_comps'] = self.Component.load_raw_components(
                    self.fit_pars['init_comps'])
            self.ncomps = len(self.fit_pars['init_comps'])
            print('Managed to load in init_comps from file')
        else:
            self.fit_pars['init_comps'] = None
            print("'Init comps' is initialised as none")
            print('test')

        # TODO: If initialising with membership probabilities, adjust self.ncomps


    def build_comps_from_chains(self, run_dir):
        """
        Build compoennt objects from stored emcee chains and cooresponding
        lnprobs.

        Parameters
        ----------
        run_dir: str
            Directory of an EM fit, which in the context of NaiveFit will be
            e.g. 'myfit/1', or 'myfit/2/A'

        Returns
        -------
        comps: [Component]
            A list of components that correspond to the best fit from the
            run in question.
        """
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


    def build_init_comps(self, prev_comps, split_comp_ix, prev_med_and_spans,
                         memb_probs):
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

        assert isinstance(target_comp, self.Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run

        if self.fit_pars['split_group']=='age':
            try:
                lo_age = prev_med_and_spans[split_comp_ix, -1, 1]
                hi_age = prev_med_and_spans[split_comp_ix, -1, 2]
            except TypeError:
                age = target_comp.get_age()
                lo_age = 0.8*age
                hi_age = 1.2*age
            except IndexError: # Added my MZ due to IndexError: too many indices for array (when using Nelder-Mead, 'final_med_and_spans.npy' is an empty file
                # Maybe previous iteration was done with Nelder-Mead
                age = target_comp.get_age()
                lo_age = 0.8*age
                hi_age = 1.2*age
            split_comps = target_comp.split_group_age(lo_age=lo_age, hi_age=hi_age)
        elif self.fit_pars['split_group']=='spatial':
            split_comps = target_comp.split_group_spatial(self.data_dict,
                                                          memb_probs[:,split_comp_ix])

        init_comps = list(prev_comps)
        init_comps.pop(split_comp_ix)
        init_comps.insert(split_comp_ix, split_comps[1])
        init_comps.insert(split_comp_ix, split_comps[0])

        return init_comps


    def run_em_unless_loadable(self, run_dir):
        """
        Run and EM fit, but only if not loadable from a previous run

        """
        try:
            # This fails when gradient descent is used and med_and_spans are not meaningful.
            try:
                med_and_spans = np.load(os.path.join(run_dir, 'final/', self.final_med_and_spans_file))
                print('run_em_unless_loadable.try successful')
            except ValueError:
                logging.info('med_and_spans not read. Presumably you are using gradient descent optimisation procedure?')
                print('run_em_unless_loadable.except ValueError')
                med_and_spans = [None]
            memb_probs = np.load(os.path.join(
                    run_dir, 'final/', self.final_memb_probs_file))
            comps = self.Component.load_raw_components(
                    str(os.path.join(run_dir, 'final/', self.final_comps_file)))
            logging.info('Loaded from previous run')

            # Handle case where Component class has been modified and can't
            # load the raw components
        except AttributeError:
            print('run_em_unless_loadable.except AttributeError')
            # TODO: check that the final chains looked for are guaranteed to be saved
            comps = self.build_comps_from_chains(run_dir)

            # Handle the case where files are missing, which means we must
            # perform the fit.
        #~ except (IOError, FileNotFoundError) as e:
        except IOError:
            print('run_em_unless_loadable.except IOError')
            print('run_em_unless_loadable: fitting comps', self.fit_pars['init_comps'])
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
        if self.fit_pars['optimisation_method']=='emcee':
            [logging.info(row[-1]) for row in prev_result['med_and_spans']]
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


    def calc_score(self, comps, memb_probs, use_box_background=False):
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
        
        print('calc_score memb_probs', memb_probs)
        
        lnlike = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                    comps,
                                                    old_memb_probs=memb_probs,
                                                    use_box_background=use_box_background,
                                                    # bg_ln_ols=bg_ln_ols,
                                                    )
        lnpost = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                    comps,
                                                    # bg_ln_ols=bg_ln_ols,
                                                    old_memb_probs=memb_probs,
                                                    use_box_background=use_box_background,
                                                    inc_posterior=True)

        bic = expectmax.calc_bic(self.data_dict, self.ncomps, lnlike,
                                 memb_probs=memb_probs,
                                 Component=self.Component)
        # 2020/11/16 TC: handling the case for a bad bic.
        # This comes up for the initial 1 component fit with box background
        # because I haven't thought of a general way to initialise memberships
        # that doesn't yield 0 background members.
        if np.isnan(bic):
            logging.info('Warning, bic was NaN')
            bic = np.inf

        return {'bic':bic, 'lnlike':lnlike, 'lnpost':lnpost}

    def write_results_to_file(self, prev_result, prev_score):
        """
        Various means of storing result to file

        Edit history
        -------------
        2020-11-12 Tim Crundall
        code originally by Marusa, Tim just moved it to avoid cluttering main
        execution method


        TODO: write fits file with id and memberships
        TODO: ascii file with components today
        """
        # WRITING THE FINAL RESULTS INTO FILES
        logging.info("... saving previous fit as best fit to data")
        self.Component.store_raw_components(self.rdir + self.final_comps_file,
                                            prev_result['comps'])
        self.Component.store_components_ascii(self.rdir + 'final_comps_ascii.txt',
                                              prev_result['comps'], overwrite=self.fit_pars['overwrite_prev_run'])
        np.save(self.rdir + self.final_med_and_spans_file, prev_result['med_and_spans'])
        np.save(self.rdir + self.final_memb_probs_file, prev_result['memb_probs'])
        np.save(self.rdir + 'final_likelihood_post_and_bic',
                prev_score)


        # Save components in fits file
        tabcomps = self.Component.convert_components_array_into_astropy_table(prev_result['comps'])

        if self.fit_pars['overwrite_fits']:
            tabcomps.write(os.path.join(self.rdir, 'final_comps_%d.fits'%len(prev_result['comps'])), overwrite=self.fit_pars['overwrite_fits'])
        else:
            filename_comps_fits_random = os.path.join(self.rdir, 'final_comps_%d_%s.fits'%(len(prev_result['comps']), str(uuid.uuid4().hex)))
            tabcomps.write(filename_comps_fits_random, overwrite=self.fit_pars['overwrite_fits'])

        # Save membership fits file
        try:
            if self.fit_pars['overwrite_fits']:
                tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(self.fit_pars['data_table'], prev_result['memb_probs'], prev_result['comps'], os.path.join(self.rdir, 'final_memberships_%d.fits'%len(prev_result['comps'])), get_background_overlaps=True, stellar_id_colname = self.fit_pars['stellar_id_colname'], overwrite_fits = self.fit_pars['overwrite_fits'])
            else:
                filename_memb_probs_fits_random = os.path.join(self.rdir, 'final_memberships_%d_%s.fits'%(len(prev_result['comps']), str(uuid.uuid4().hex)))
                tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(self.fit_pars['data_table'], prev_result['memb_probs'], prev_result['comps'], filename_memb_probs_fits_random, get_background_overlaps=True, stellar_id_colname = self.fit_pars['stellar_id_colname'], overwrite_fits = self.fit_pars['overwrite_fits'])
        except:
            logging.info("[WARNING] Couldn't print membership.fits file. Check column id.")

        self.log_final_log(prev_result, prev_score)

