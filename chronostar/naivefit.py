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
        'init_comps':None,
        'init_comps_file':None, # TODO: Merge these two
        'skip_init_fit':False,

        'component':'sphere',
        'max_comp_count':20,
        'max_em_iterations':200,
        'mpi_threads':None,     # TODO: NOT IMPLEMENTED
        'using_bg':True,
        'include_background_distribution':True, # TODO: Redundant with 'using_bg'?

        'overwrite_prev_run':False,
        'burnin_steps':500,
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
        # -----  EXECUTE RUN  ----------------------------------------
        # ------------------------------------------------------------

        if self.fit_pars['store_burnin_chains']:
            log_message(msg='Storing burnin chains', symbol='-')

        if self.ncomps == 1:
            # Fit the first component
            log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                        symbol='*', surround=True)
            run_dir = self.rdir + '{}/'.format(self.ncomps)

            # Initialise all stars in dataset to be full members of first component
            using_bg = self.fit_pars['include_background_distribution']
            init_memb_probs = np.zeros((len(self.data_dict['means']), 1 + using_bg))
            init_memb_probs[:, 0] = 1.

            # Try and recover any results from previous run
            try:
                prev_med_and_spans = np.load(run_dir + 'final/'
                                             + self.final_med_and_spans_file)
                prev_memb_probs = np.load(
                    run_dir + 'final/' + self.final_memb_probs_file)
                try:
                    prev_comps = self.Component.load_raw_components(
                            str(run_dir + 'final/' + self.final_comps_file))

                # Final comps are there, they just can't be read by current module
                # so quickly fit them based on fixed prev membership probabilities
                except AttributeError:
                    logging.info(
                        'Component class has been modified, reconstructing '
                        'from chain')
                    prev_comps = self.ncomps * [None]
                    for i in range(self.ncomps):
                        final_cdir = run_dir + 'final/comp{}/'.format(i)
                        chain = np.load(final_cdir + 'final_chain.npy')
                        lnprob = np.load(final_cdir + 'final_lnprob.npy')
                        npars = len(self.Component.PARAMETER_FORMAT)
                        best_ix = np.argmax(lnprob)
                        best_pars = chain.reshape(-1, npars)[best_ix]
                        prev_comps[i] = self.Component(emcee_pars=best_pars)
                    self.Component.store_raw_components(
                        str(run_dir + 'final/' + self.final_comps_file),
                        prev_comps)
                    # np.save(str(run_dir+'final/'+final_comps_file), prev_comps)

                logging.info('Loaded from previous run')
            except IOError:
                prev_comps, prev_med_and_spans, prev_memb_probs = \
                    expectmax.fit_many_comps(data=self.data_dict, ncomps=self.ncomps,
                                             rdir=run_dir,
                                             trace_orbit_func=self.fit_pars['trace_orbit_func'],
                                             burnin=self.fit_pars[
                                                 'burnin_steps'],
                                             sampling_steps=self.fit_pars[
                                                 'sampling_steps'],
                                             use_background=self.fit_pars[
                                                 'include_background_distribution'],
                                             init_memb_probs=init_memb_probs,
                                             init_comps=self.fit_pars['init_comps'],
                                             Component=self.Component,
                                             store_burnin_chains=self.fit_pars['store_burnin_chains'],
                                             max_iters=self.fit_pars['max_em_iterations'],
                                             ignore_stable_comps=
                                             self.fit_pars[
                                                 'ignore_stable_comps'],
                                             )


            self.ncomps += 1

        if self.fit_pars['init_comps'] is not None and len(self.fit_pars['init_comps']) > 1:
            # fit with ncomps until convergence
            # Note: might
            run_dir = self.rdir + '{}/'.format(self.ncomps)
            # TODO: VVV This is bugged, results need to be stored into 'prev_comps' etc. TC 2019.12.28
            prev_comps, prev_med_and_spans, prev_memb_probs = \
                expectmax.fit_many_comps(
                        data=self.data_dict, ncomps=self.ncomps, rdir=run_dir,
                        init_comps=self.fit_pars['init_comps'],
                        trace_orbit_func=self.fit_pars['trace_orbit_func'],
                        use_background=self.fit_pars[
                            'include_background_distribution'],
                        burnin=self.fit_pars['burnin_steps'],
                        sampling_steps=self.fit_pars['sampling_steps'],
                        Component=self.Component,
                        store_burnin_chains=self.fit_pars['store_burnin_chains'],
                        max_iters=self.fit_pars['max_em_iterations'],
                        ignore_stable_comps=self.fit_pars[
                            'ignore_stable_comps'],
                )
            self.ncomps += 1

        # Calculate global score of fit for comparison with future fits with different
        # component counts
        prev_lnlike = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                         prev_comps,
                                                         # bg_ln_ols=bg_ln_ols,
                                                         )
        prev_lnpost = expectmax.get_overall_lnlikelihood(self.data_dict,
                                                         prev_comps,
                                                         # bg_ln_ols=bg_ln_ols,
                                                         inc_posterior=True)
        prev_bic = expectmax.calc_bic(self.data_dict, self.ncomps, prev_lnlike,
                                      memb_probs=prev_memb_probs,
                                      Component=self.Component)

        # Begin iterative loop, each time trialing the incorporation of a new component
        while self.ncomps <= self.fit_pars['max_component_count']:
            log_message(msg='FITTING {} COMPONENT'.format(self.ncomps),
                        symbol='*', surround=True)

            best_fits = []
            lnlikes = []
            lnposts = []
            bics = []
            all_med_and_spans = []
            all_memb_probs = []

            # Iteratively try subdividing each previous component
            for i, target_comp in enumerate(prev_comps):
                div_label = chr(ord('A') + i)
                run_dir = self.rdir + '{}/{}/'.format(self.ncomps, div_label)
                log_message(msg='Subdividing stage {}'.format(div_label),
                            symbol='+', surround=True)
                mkpath(run_dir)

                assert isinstance(target_comp, self.Component)
                # Decompose and replace the ith component with two new components
                # by using the 16th and 84th percentile ages from previous run
                split_comps = target_comp.splitGroup(
                    lo_age=prev_med_and_spans[i, -1, 1],
                    hi_age=prev_med_and_spans[i, -1, 2])
                init_comps = list(prev_comps)
                init_comps.pop(i)
                init_comps.insert(i, split_comps[1])
                init_comps.insert(i, split_comps[0])

                # Run em fit
                # First try and find any previous runs
                try:
                    med_and_spans = np.load(run_dir + 'final/'
                                            + self.final_med_and_spans_file)
                    memb_probs = np.load(
                        run_dir + 'final/' + self.final_memb_probs_file)
                    try:
                        comps = self.Component.load_raw_components(run_dir + 'final/'
                                                              + self.final_comps_file)
                    # Final comps are there, they just can't be read by current module
                    # so quickly retrieve them from the sample chain
                    except AttributeError:
                        logging.info(
                                'Component class has been modified, reconstructing from'
                                'chain.')

                        raise UserWarning('This has not been tested, probs broken, '
                                          'safer to just cancel for now')
                        # TODO port this bug fix to run_chronostar on master
                        comps = self.ncomps * [None]
                        for j in range(self.ncomps):
                            final_cdir = run_dir + 'final/comp{}/'.format(j)
                            chain = np.load(final_cdir + 'final_chain.npy')
                            lnprob = np.load(final_cdir + 'final_lnprob.npy')
                            npars = len(self.Component.PARAMETER_FORMAT)
                            best_ix = np.argmax(lnprob)
                            best_pars = chain.reshape(-1, npars)[best_ix]
                            prev_comps[j] = self.Component(emcee_pars=best_pars)
                        self.Component.store_raw_components(
                            str(run_dir + 'final/' + self.final_comps_file),
                            prev_comps)
                        # np.save(str(run_dir + 'final/' + final_comps_file), prev_comps)

                    logging.info('Fit loaded from previous run')
                except IOError:
                    comps, med_and_spans, memb_probs = \
                        expectmax.fit_many_comps(
                                data=self.data_dict, ncomps=self.ncomps, rdir=run_dir,
                                init_comps=init_comps,
                                trace_orbit_func=self.fit_pars['trace_orbit_func'],
                                use_background=self.fit_pars[
                                    'include_background_distribution'],
                                burnin=self.fit_pars['burnin_steps'],
                                sampling_steps=self.fit_pars[
                                    'sampling_steps'],
                                Component=self.Component,
                                store_burnin_chains=self.fit_pars['store_burnin_chains'],
                                max_iters=self.fit_pars['max_em_iterations'],
                                ignore_stable_comps=self.fit_pars[
                                    'ignore_stable_comps'],
                        )

                best_fits.append(comps)
                all_med_and_spans.append(med_and_spans)
                all_memb_probs.append(memb_probs)
                lnlikes.append(
                    expectmax.get_overall_lnlikelihood(self.data_dict, comps))
                lnposts.append(
                        expectmax.get_overall_lnlikelihood(self.data_dict, comps,
                                                           inc_posterior=True)
                )
                bics.append(expectmax.calc_bic(self.data_dict, self.ncomps, lnlikes[-1],
                                               memb_probs=memb_probs,
                                               Component=self.Component))
                logging.info(
                        'Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
                        'lnpost: {}'.format(
                                div_label, bics[-1], lnlikes[-1], lnposts[-1],
                        ))

            # identify the best performing decomposition
            # best_split_ix = np.argmax(lnposts)
            best_split_ix = np.argmin(bics)
            new_comps, new_meds, new_z, new_lnlike, new_lnpost, new_bic = \
                list(zip(best_fits, all_med_and_spans, all_memb_probs,
                         lnlikes, lnposts, bics))[best_split_ix]
            logging.info("Selected {} as best decomposition".format(
                    chr(ord('A') + best_split_ix)))
            logging.info(
                "Turned\n{}".format(prev_comps[best_split_ix].get_pars()))
            logging.info('with {} members'.format(
                    prev_memb_probs.sum(axis=0)[best_split_ix]))
            logging.info("into\n{}\n&\n{}".format(
                    new_comps[best_split_ix].get_pars(),
                    new_comps[best_split_ix + 1].get_pars(),
            ))
            logging.info('with {} and {} members'.format(
                    new_z.sum(axis=0)[best_split_ix],
                    new_z.sum(axis=0)[best_split_ix + 1],
            ))
            logging.info("for an overall membership breakdown\n{}".format(
                    new_z.sum(axis=0)
            ))

            # Check if the fit has improved
            if new_bic < prev_bic:
                logging.info("Extra component has improved BIC...")
                logging.info(
                    "New BIC: {} < Old BIC: {}".format(new_bic, prev_bic))
                logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
                logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
                prev_comps, prev_med_and_spans, prev_memb_probs, prev_lnlike, prev_lnpost, \
                prev_bic = \
                    (
                    new_comps, new_meds, new_z, new_lnlike, new_lnpost, new_bic)
                self.ncomps += 1
                log_message(msg="Commencing {} component fit on {}{}".format(
                        self.ncomps, self.ncomps - 1,
                        chr(ord('A') + best_split_ix)), symbol='+'
                )
            else:
                logging.info("Extra component has worsened BIC...")
                logging.info(
                    "New BIC: {} > Old BIC: {}".format(new_bic, prev_bic))
                logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
                logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
                logging.info("... saving previous fit as best fit to data")
                self.Component.store_raw_components(self.rdir + self.final_comps_file,
                                                    prev_comps)
                np.save(self.rdir + self.final_med_and_spans_file, prev_med_and_spans)
                np.save(self.rdir + self.final_memb_probs_file, prev_memb_probs)
                np.save(self.rdir + 'final_likelihood_post_and_bic',
                        [prev_lnlike, prev_lnpost,
                         prev_bic])
                logging.info('Final best fits:')
                [logging.info(c.get_pars()) for c in prev_comps]
                logging.info('Final age med and span:')
                [logging.info(row[-1]) for row in prev_med_and_spans]
                logging.info('Membership distribution: {}'.format(
                    prev_memb_probs.sum(axis=0)))
                logging.info('Final membership:')
                logging.info('\n{}'.format(np.round(prev_memb_probs * 100)))
                logging.info('Final lnlikelihood: {}'.format(prev_lnlike))
                logging.info('Final lnposterior:  {}'.format(prev_lnpost))
                logging.info('Final BIC: {}'.format(prev_bic))
                break

            logging.info("Best fit:\n{}".format(
                    [group.get_pars() for group in prev_comps]))


        if self.ncomps >= self.fit_pars['max_component_count']:
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
            return prev_comps, prev_med_and_spans, prev_memb_probs, \
                    prev_lnlike, prev_lnpost, prev_bic
