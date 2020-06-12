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
from numpy.polynomial import polynomial
import os
import sys
import logging
from distutils.dir_util import mkpath
import random

#~ from emcee.utils import MPIPool
#~ from multiprocessing import Pool

from multiprocessing import cpu_count

sys.path.insert(0, os.path.abspath('..'))
from . import expectmax
from . import readparam
from . import tabletool
from . import component
from . import traceorbit

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


class NaiveFit(object):
    """
        Many arguments can be taken straight from the fit_pars dictionary,
        so no point explicitly looking for them.

        Description of parameters can be found in README.md along with their
        default values and whether they are required.
    """

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

        # MZ: Make sure 'par_log_file' is written into the results folder
        self.fit_pars['par_log_file'] = os.path.join(self.fit_pars['results_dir'], self.fit_pars['par_log_file'])

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

        # Make some logs about how many iterations (+ other stuff) code can run for
        log_message(msg='Component count cap set to {}'.format(
                self.fit_pars['max_comp_count']),
                symbol='+', surround=True)
        log_message(msg='Iteration count cap set to {}'.format(
                self.fit_pars['max_em_iterations']),
                symbol='+', surround=True)

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


    def build_init_comps_original(self, prev_comps, split_comp_ix, prev_med_and_spans):
        """
        Given a list of converged components from a N compoennt fit, generate
        a list of N+1 components with which to intiailise an EM run.

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
        """
        target_comp = prev_comps[split_comp_ix]

        assert isinstance(target_comp, self.Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run
        split_comps = target_comp.splitGroup(
            lo_age=prev_med_and_spans[split_comp_ix, -1, 1],
            hi_age=prev_med_and_spans[split_comp_ix, -1, 2])
        init_comps = list(prev_comps)
        init_comps.pop(split_comp_ix)
        init_comps.insert(split_comp_ix, split_comps[1])
        init_comps.insert(split_comp_ix, split_comps[0])

        return init_comps


    def get_age_from_slope_sigma_clipping(self, memb_probs):
        mask = memb_probs>0.5 # All True
        
        means = self.data_dict['means']
        
        x=means[mask,0]
        y=means[mask,3]
        
        len_init = len(x)
        len_prev=len_init
        
        niter=0
        while len(x)>0.5*len_init and niter<20:
            coeff = np.polyfit(x, y, 1)
            f = np.poly1d(coeff)
            
            # O-C
            omc = y - f(x)
            sigma = np.std(omc) # Not really sigma...
            
            mask = np.abs(omc)<1.0*sigma
            
            #~ print(omc)
            print(sigma, len(x))
            
            x=x[mask]
            y=y[mask]
            
            niter+=1
            
            if len(x)==len_prev:
                break
            else:
                len_prev=len(x)


        coeff = np.polyfit(x, y, 1)
        f = np.poly1d(coeff)
        
        slope = coeff[0]
        age = 1.0 / (slope * 1.0227121650537077)
        
        # Age cannot be negative
        if age<0:
            age=1
        
        if age>30:
            age=30
            print('WARNING: Age estimate from slope set to the max limit of 30 Myr!')
        
        
        print('slopesigma, age', slope, age)
        
        return age

    def get_age_from_slope(self, memb_probs):
        """
        Estimate component age from the slope in the XU space.
        This is used as a reference age where a component is split
        into two. Such estimation is needed due to degeneracy in...
        """
        
        means = self.data_dict['means']
        #~ coeff = polynomial.polyfit(means[:,0], means[:,3], 1, w=memb_probs) # TODO: Maybe really take only members if number of stars gets large!!
        
        # Take onlt 30% of the more likely members
        prob_value = np.percentile(memb_probs, 70)
        mask_members = memb_probs>prob_value
        
        #~ coeff = polynomial.polyfit(means[mask_members,0], means[mask_members,3], 1, w=memb_probs[mask_members])
        #~ coeff = polynomial.polyfit(means[mask_members,0], means[mask_members,3], 1)
        
        coeff = np.polyfit(means[mask_members,0], means[mask_members,3], 1)
        f = np.poly1d(coeff)
        
        slope = coeff[0]
        age = 1.0 / (slope * 1.0227121650537077) # Myr; 1.0227121650537077 is to convert from km/s/pc to 1/Myr
        
        #~ import matplotlib.pyplot as plt
        #~ fig=plt.figure()
        #~ ax=fig.add_subplot(111)
        #~ ax.scatter(means[mask_members,0], means[mask_members,3])
        #f=np.polyval(coeff)
        #~ x=[np.min(means[mask_members,0]), np.max(means[mask_members,0])]
        #~ ax.plot(x, f(x), c='r')
        #~ plt.show()
        
        print('slope, age', slope, age)
        return age


    def build_init_comps(self, prev_comps, split_comp_ix, prev_med_and_spans, memb_probs): # Scipy modification
        """
        Given a list of converged components from a N compoennt fit, generate
        a list of N+1 components with which to intiailise an EM run.
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
        """
        target_comp = prev_comps[split_comp_ix]

        assert isinstance(target_comp, self.Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run
        
        # This age estimate was used in emcee fit
        age = target_comp.get_age()
        print('age target_comp.get_age()', age)
        
        #~ split_comps = target_comp.splitGroup(
            #~ lo_age=prev_med_and_spans[split_comp_ix, -1, 1],
            #~ hi_age=prev_med_and_spans[split_comp_ix, -1, 2])
        
        ##~ age_slope_estimate = self.get_age_from_slope(memb_probs)
        #~ age_slope_estimate = self.get_age_from_slope_sigma_clipping(memb_probs)
        #~ import pickle
        #~ with open('data_for_plot_%d'%self.ncomps, 'wb') as handle:
            #~ pickle.dump([self.data_dict, memb_probs, age, age_slope_estimate], handle)
        
        #~ split_comps = target_comp.splitGroup( # TODO: Maybe even smaller change
            #~ lo_age=0.9*age_slope_estimate,
            #~ hi_age=1.1*age_slope_estimate)
        split_comps = target_comp.splitGroup( # TODO: Maybe even smaller change
            lo_age=0.8*age,
            hi_age=1.2*age)
        init_comps = list(prev_comps)
        init_comps.pop(split_comp_ix)
        init_comps.insert(split_comp_ix, split_comps[1])
        init_comps.insert(split_comp_ix, split_comps[0])

        print('INIT_COMPS from build_init_comps', np.array([c.get_emcee_pars() for c in init_comps]))
        
        return init_comps


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
            print('Loaded from previous run', str(run_dir + 'final/' + self.final_comps_file))

            # Handle case where Component class has been modified and can't
            # load the raw components
        except AttributeError:
            # TODO: check that the final chains looked for are guaranteed to be saved
            comps = self.build_comps_from_chains(run_dir)

            # Handle the case where files are missing, which means we must
            # perform the fit.
        except IOError:
            # Fit many compy should be scipy modified!
            comps, med_and_spans, memb_probs = \
                expectmax.fit_many_comps(data=self.data_dict,
                                         ncomps=self.ncomps, rdir=run_dir,
                                         #~ init_comps = self.fit_pars['init_comps'],
                                         **self.fit_pars)

        # Since init_comps and init_memb_probs are only meant for one time uses
        # we clear them to avoid any future usage
        self.fit_pars['init_comps'] = None
        self.fit_pars['init_memb_probs'] = None

        print('COMPS', comps)

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
            # target_comp is the component we will split into two.
            # This will make a total of ncomps (the target comp split into 2,
            # plus the remaining components from prev_result['comps']
            for i, target_comp in enumerate(prev_result['comps']):
                div_label = chr(ord('A') + i)
                run_dir = self.rdir + '{}/{}/'.format(self.ncomps, div_label)
                log_message(msg='Subdividing stage {}'.format(div_label),
                            symbol='+', surround=True)
                mkpath(run_dir)

                # SPLIT THE i-TH COMPONENT: KEEP XYZUVWdXdV but split in the age
                self.fit_pars['init_comps'] = self.build_init_comps(
                        prev_result['comps'], split_comp_ix=i,
                        prev_med_and_spans=prev_result['med_and_spans'], memb_probs=prev_result['memb_probs'][:,i])

                
                #~ print('BEFORE SPLIT:', i)
                #~ print(prev_result['comps'])
                
                #~ print('AFTER SPLIT:', i)
                #~ print(self.fit_pars['init_comps'])

                result = self.run_em_unless_loadable(run_dir)
                all_results.append(result)

                score = self.calc_score(result['comps'], result['memb_probs'])
                all_scores.append(score)

                logging.info(
                        'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
                        'lnpost: {}'.format(
                                div_label, all_scores[-1]['bic'],
                                all_scores[-1]['lnlike'], all_scores[-1]['lnpost'],
                        ))

            # identify the best performing decomposition
            all_bics = [score['bic'] for score in all_scores if not np.isnan(score['bic'])] # MZ: ADDED np.isnan()
            best_split_ix = np.argmin(all_bics)
            print('ALL_BICS', all_bics)
            print('BEST_SPLIT_IX', best_split_ix)

            new_result = all_results[best_split_ix]
            new_score = all_scores[best_split_ix]

            self.iter_end_log(best_split_ix, prev_result=prev_result, new_result=new_result)

            # Check if the fit has improved
            self.log_score_comparison(new=new_score,
                                      prev=prev_score)
            print('NCOMPS', self.ncomps, new_score['bic'], prev_score['bic'], new_score['bic'] < prev_score['bic'])
            if new_score['bic'] < prev_score['bic']:
                prev_score = new_score
                prev_result = new_result

                self.ncomps += 1
                log_message(msg="Commencing {} component fit on {}{}".format(
                        self.ncomps, self.ncomps - 1,
                        chr(ord('A') + best_split_ix)), symbol='+'
                )
            else:
                # WRITING THE FINAL RESULTS INTO FILES
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

        return prev_result, prev_score

    def first_run_fit(self):
        """
        MZ: 2020 - 04 - 23
        This is the first part of run_fit. run_fit is split into two parts
        for the multiprocessing purposes.
        
        
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
        
        self.prev_result = prev_result
        self.prev_score = prev_score

    def run_split_for_one_comp_multiproc(self, i=0, pool=None):
        """
        MZ: 2020 - 04 - 23
        
        Compute split of one component.
        This is taken from the run_fit and is made for multiprocessing
        purposes.
        
        Params:
        -----------
        i: int
            Compute split for the i-th component
            
        Returns:
        -----------
        result:
        score:
        
        """
        # ------------------------------------------------------------
        # -----  EXPLORE EXTRA COMPONENT BY DECOMPOSITION  -----------
        # ------------------------------------------------------------

        # Calculate global score of fit for comparison with future fits with different
        # component counts

        log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                    symbol='*', surround=True)


        # Iteratively try subdividing each previous component
        # target_comp is the component we will split into two.
        # This will make a total of ncomps (the target comp split into 2,
        # plus the remaining components from prev_result['comps']
        #~ print('$$$', i, self.prev_result['comps'])
        target_comp = self.prev_result['comps'][i]
        div_label = chr(ord('A') + i)
        run_dir = self.rdir + '{}/{}/'.format(self.ncomps, div_label)
        log_message(msg='Subdividing stage {}'.format(div_label),
                    symbol='+', surround=True)
        mkpath(run_dir)

        #~ self.fit_pars['init_comps'] = self.build_init_comps(
                #~ self.prev_result['comps'], split_comp_ix=i,
                #~ prev_med_and_spans=self.prev_result['med_and_spans'])
                        # SPLIT THE i-TH COMPONENT: KEEP XYZUVWdXdV but split in the age
        self.fit_pars['init_comps'] = self.build_init_comps(
                self.prev_result['comps'], split_comp_ix=i,
                prev_med_and_spans=self.prev_result['med_and_spans'], memb_probs=self.prev_result['memb_probs'][:,i])

        result = self.run_em_unless_loadable(run_dir)
        score = self.calc_score(result['comps'], result['memb_probs'])
        all_scores=[score]

        logging.info(
                'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
                'lnpost: {}'.format(
                        div_label, all_scores[-1]['bic'],
                        all_scores[-1]['lnlike'], all_scores[-1]['lnpost'],
                ))

        return result, score
            
                        
    def run_fit_gather_results_multiproc(self, all_results, all_scores):
        # identify the best performing decomposition
        all_bics = [score['bic'] for score in all_scores]
        best_split_ix = np.argmin(all_bics)

        new_result = all_results[best_split_ix]
        new_score = all_scores[best_split_ix]

        self.iter_end_log(best_split_ix, prev_result=self.prev_result, new_result=new_result)

        # Check if the fit has improved
        self.log_score_comparison(new=new_score,
                                  prev=self.prev_score)
        if new_score['bic'] < self.prev_score['bic']:
            prev_score = new_score
            prev_result = new_result
            self.prev_result = prev_result
            self.prev_score = prev_score

            self.ncomps += 1
            log_message(msg="Commencing {} component fit on {}{}".format(
                    self.ncomps, self.ncomps - 1,
                    chr(ord('A') + best_split_ix)), symbol='+'
            )
            terminate=False
            print('decide', terminate)
        else:
            # WRITING THE FINAL RESULTS INTO FILES
            logging.info("... saving previous fit as best fit to data")
            self.Component.store_raw_components(self.rdir + self.final_comps_file,
                                                self.prev_result['comps'])
            np.save(self.rdir + self.final_med_and_spans_file, self.prev_result['med_and_spans'])
            np.save(self.rdir + self.final_memb_probs_file, self.prev_result['memb_probs'])
            np.save(self.rdir + 'final_likelihood_post_and_bic',
                    self.prev_score)

            # Save fits files as well
            tabcomps = self.Component.convert_components_array_into_astropy_table(self.prev_result['comps'])
            tabcomps.write(os.path.join(self.rdir, 'final_comps_%d.fits'%(self.ncomps-1)), overwrite=True)
            try:
                tabletool.construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(self.fit_pars['data_table'], self.prev_result['memb_probs'], self.prev_result['comps'], os.path.join(self.rdir, 'final_memberships_%d.fits'%(self.ncomps-1)), get_background_overlaps=True)
            except:
                logging.info("[WARNING] Couldn't print membership.fits file. Is source_id available?")

            self.log_final_log(self.prev_result, self.prev_score)
            terminate=True
            print('decide', terminate)
            #~ break

        logging.info("Best fit:\n{}".format(
                [group.get_pars() for group in self.prev_result['comps']]))


        return terminate
