"""
smartsplitterfit.py

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


class SmartSplitFit(object):
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
            except ValueError:
                logging.info('med_and_spans not read. Presumably you are using gradient descent optimisation procedure?')
                med_and_spans = [None]
            memb_probs = np.load(os.path.join(
                run_dir, 'final/', self.final_memb_probs_file))
            comps = self.Component.load_raw_components(
                str(os.path.join(run_dir, 'final/', self.final_comps_file)))
            logging.info('Loaded from previous run')

            # Handle case where Component class has been modified and can't
            # load the raw components
        except AttributeError:
            # TODO: check that the final chains looked for are guaranteed to be saved
            comps = self.build_comps_from_chains(run_dir)

            # Handle the case where files are missing, which means we must
            # perform the fit.
        #~ except (IOError, FileNotFoundError) as e:
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
