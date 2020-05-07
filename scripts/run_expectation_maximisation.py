"""
A Chronostar script that runs Expectation-Maximisation algorithm for
multiple components (only one?). It should really fit only one!

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

import os.path
import sys
sys.path.insert(0, '..')

from chronostar.expectmax import expectmax
import chronostar.tabletool as tabletool

# Read fit parameters from the file
fit_pars = 

# Prepare data
data_dict = tabletool.build_data_dict_from_table(fit_pars['data_table'])

ncomps = sys.argv[1] # Is this OK?

output_folder = os.path.join(fit_pars['results_dir'], '{}/'.format(ncomps))

expectmax.fit_many_comps(data=data_dict, ncomps=ncomps, rdir=output_folder,
                         **fit_pars)

def fit_many_comps(data, ncomps, rdir='', pool=None, init_memb_probs=None,
                   init_comps=None, inc_posterior=False, burnin=1000,
                   sampling_steps=5000, ignore_dead_comps=False,
                   Component=SphereComponent, trace_orbit_func=None,
                   use_background=False, store_burnin_chains=False,
                   ignore_stable_comps=False, max_em_iterations=100,
                   record_len=30, bic_conv_tol=0.1, min_em_iterations=30,
                   nthreads=1,
                   **kwargs):
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
    # Tidying up input
    if not isinstance(data, dict):
        data = tabletool.build_data_dict_from_table(
                data, get_background_overlaps=use_background
        )
    if rdir == '':                      # Ensure results directory has a
        rdir = '.'                      # trailing '/'
    rdir = rdir.rstrip('/') + '/'
    if not os.path.exists(rdir):
        mkpath(rdir)

    if use_background:
        assert 'bg_lnols' in data.keys()

    # filenames
    init_comp_filename = 'init_comps.npy'

    # setting up some constants
    nstars = data['means'].shape[0]
    C_TOL = 0.5

    logging.info("Fitting {} groups with {} burnin steps with cap "
                 "of {} iterations".format(ncomps, burnin, max_em_iterations))

    # INITIALISE RUN PARAMETERS

    # If initialising with components then need to convert to emcee parameter lists
    if init_comps is not None:
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
        all_init_pars = ncomps * [None]
        init_comps = ncomps * [None]
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
        all_init_pars     = ncomps * [None]
        init_comps        = ncomps * [None]

    # Store the initial components if available
    if init_comps[0] is not None:
        Component.store_raw_components(rdir + init_comp_filename, init_comps)
    # np.save(rdir + init_comp_filename, init_comps)

    # Initialise values for upcoming iterations
    old_comps          = init_comps
    # old_overall_lnlike = -np.inf
    all_init_pos       = ncomps * [None]
    all_med_and_spans  = ncomps * [None]
    all_converged      = False
    stable_state       = True         # used to track issues

    # Keep track of all fits for convergence checking
    list_prev_comps        = []
    list_prev_memberships  = []
    list_all_init_pos      = []
    list_all_med_and_spans = []
    list_prev_bics         = []

    # Keep track of ALL BICs, so that progress can be observed
    all_bics = []

    # Keep track of unstable components, which will require
    # extra iterations
    ref_counts = None
    if ignore_stable_comps:
        unstable_comps = np.array(ncomps * [True])
    else:
        unstable_comps = None

    # Look for previous iterations and update values as appropriate
    prev_iters       = True
    iter_count       = 0
    found_prev_iters = False
    while prev_iters:
        try:
            idir = rdir+"iter{:02}/".format(iter_count)
            memb_probs_old = np.load(idir + 'membership.npy')
            try:
                old_comps = Component.load_raw_components(idir + 'best_comps.npy')
            # End up here if components aren't loadable due to change in module
            # So we rebuild from chains
            except AttributeError:
                old_comps = ncomps * [None]
                for i in range(ncomps):
                    chain   = np.load(idir + 'comp{}/final_chain.npy'.format(i))
                    lnprob  = np.load(idir + 'comp{}/final_lnprob.npy'.format(i))
                    npars   = len(Component.PARAMETER_FORMAT)
                    best_ix = np.argmax(lnprob)
                    best_pars    = chain.reshape(-1, npars)[best_ix]
                    old_comps[i] = Component(emcee_pars=best_pars)
                    all_med_and_spans[i] = compfitter.calc_med_and_span(
                            chain, intern_to_extern=True, Component=Component,
                    )

            all_init_pars = [old_comp.get_emcee_pars()
                             for old_comp in old_comps]
            old_overall_lnlike, old_memb_probs = \
                    get_overall_lnlikelihood(data, old_comps,
                                             inc_posterior=False,
                                             return_memb_probs=True,)
            ref_counts = np.sum(old_memb_probs, axis=0)

            list_prev_comps.append(old_comps)
            list_prev_memberships.append(old_memb_probs)
            list_all_init_pos.append(all_init_pos)
            list_all_med_and_spans.append(all_med_and_spans)
            list_prev_bics.append(calc_bic(data, len(old_comps),
                                           lnlike=old_overall_lnlike,
                                           memb_probs=old_memb_probs))

            all_bics.append(list_prev_bics[-1])

            iter_count += 1
            found_prev_iters = True

        except IOError:
            logging.info("Managed to find {} previous iterations".format(
                iter_count
            ))
            prev_iters = False

    # Until convergence is achieved (or max_iters is exceeded) iterate through
    # the Expecation and Maximisation stages

    # TODO: put convergence checking at the start of the loop so restarting doesn't repeat an iteration
    while not all_converged and stable_state and iter_count < max_em_iterations:
        ignore_stable_comps_iter = ignore_stable_comps and (iter_count % 5 != 0)

        # for iter_count in range(10):
        idir = rdir+"iter{:02}/".format(iter_count)
        mkpath(idir)

        log_message('Iteration {}'.format(iter_count),
                    symbol='-', surround=True)
        if not ignore_stable_comps_iter:
            log_message('Fitting all {} components'.format(ncomps))
            unstable_comps = np.where(np.array(ncomps * [True]))
        else:
            log_message('Fitting the following unstable comps:')
            log_message('TC: maybe fixed?')
            log_message(str(np.arange(ncomps)[unstable_comps]))
            log_message('MZ: removed this line due to index error (unstable_comps too big number)')
            log_message(str(unstable_comps))

        # EXPECTATION
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
            memb_probs_new = expectation(data, old_comps, memb_probs_old,
                                         inc_posterior=inc_posterior)
        logging.info("Membership distribution:\n{}".format(
            memb_probs_new.sum(axis=0)
        ))
        np.save(idir+"membership.npy", memb_probs_new)

        # MAXIMISE
        new_comps, all_samples, _, all_init_pos, success_mask =\
            maximisation(data, ncomps=ncomps,
                         burnin_steps=burnin,
                         plot_it=True, pool=pool, convergence_tol=C_TOL,
                         memb_probs=memb_probs_new, idir=idir,
                         all_init_pars=all_init_pars,
                         all_init_pos=all_init_pos,
                         ignore_dead_comps=ignore_dead_comps,
                         trace_orbit_func=trace_orbit_func,
                         store_burnin_chains=store_burnin_chains,
                         unstable_comps=unstable_comps,
                         ignore_stable_comps=ignore_stable_comps_iter,
                         nthreads=nthreads,
                         )

        for i in range(ncomps):
            if i in success_mask:
                j = success_mask.index(i)
                all_med_and_spans[i] = compfitter.calc_med_and_span(
                        all_samples[j], intern_to_extern=True,
                        Component=Component,
                )
            # If component is stable, then it wasn't fit, so just duplicate
            # from last fit
            else:
                all_med_and_spans[i] = list_all_med_and_spans[-1][i]
                new_comps.insert(i,list_prev_comps[-1][i])
                all_init_pos.insert(i,list_all_init_pos[-1][i])

        Component.store_raw_components(idir + 'best_comps.npy', new_comps)
        np.save(idir + 'best_comps_bak.npy', new_comps)


        logging.info('DEBUG: new_comps length: {}'.format(len(new_comps)))

        # LOG RESULTS OF ITERATION
        overall_lnlike = get_overall_lnlikelihood(data, new_comps,
                                                 inc_posterior=False)
        overall_lnposterior = get_overall_lnlikelihood(data, new_comps,
                                                      inc_posterior=True)
        bic = calc_bic(data, ncomps, overall_lnlike,
                       memb_probs=memb_probs_new,
                       Component=Component)

        logging.info("---        Iteration results         --")
        logging.info("-- Overall likelihood so far: {} --".\
                     format(overall_lnlike))
        logging.info("-- Overall posterior so far:  {} --". \
                     format(overall_lnposterior))
        logging.info("-- BIC so far: {}                --". \
                     format(calc_bic(data, ncomps, overall_lnlike,
                                     memb_probs=memb_probs_new,
                                     Component=Component)))

        list_prev_comps.append(new_comps)
        list_prev_memberships.append(memb_probs_new)
        list_all_init_pos.append(all_init_pos)
        list_all_med_and_spans.append(all_med_and_spans)
        list_prev_bics.append(bic)

        all_bics.append(bic)

        if len(list_prev_bics) < min_em_iterations:
            all_converged = False
        else:
            all_converged = compfitter.burnin_convergence(
                    lnprob=np.expand_dims(list_prev_bics[-min_em_iterations:], axis=0),
                    tol=bic_conv_tol, slice_size=int(min_em_iterations/2)
            )
        old_overall_lnlike = overall_lnlike
        log_message('Convergence status: {}'.format(all_converged),
                    symbol='-', surround=True)
        if not all_converged:
            logging.info('BIC not converged')
            np.save(rdir + 'all_bics.npy', all_bics)

            # Plot all bics to date
            plt.clf()
            plt.plot(all_bics,
                     label='All {} BICs'.format(len(all_bics)))
            plt.vlines(np.argmin(all_bics), linestyles='--', color='red',
                       ymin=plt.ylim()[0], ymax=plt.ylim()[1],
                       label='best BIC {:.2f} | iter {}'.format(np.min(all_bics),
                                                                np.argmin(all_bics)))
            plt.legend(loc='best')
            plt.savefig(rdir + 'all_bics.pdf')

        # Check individual components stability
        if (iter_count % 5 == 0 and ignore_stable_comps):
            memb_probs_new = expectation(data, new_comps, memb_probs_new,
                                         inc_posterior=inc_posterior)
            log_message('Orig ref_counts {}'.format(ref_counts))

            unstable_comps, ref_counts = check_comps_stability(memb_probs_new,
                                                               unstable_comps,
                                                               ref_counts,
                                                               using_bg=use_background)
            log_message('New memb counts: {}'.format(memb_probs_new.sum(axis=0)))
            log_message('Unstable comps: {}'.format(unstable_comps))
            log_message('New ref_counts {}'.format(ref_counts))


        # Check stablity, but only affect run after sufficient iterations to
        # settle
        temp_stable_state = check_stability(data, new_comps, memb_probs_new)
        logging.info('Stability: {}'.format(temp_stable_state))
        if iter_count > 10:
            stable_state = temp_stable_state

        # only update if we're about to iterate again
        if not all_converged:
            old_comps = new_comps
            memb_probs_old = memb_probs_new

        iter_count += 1

    logging.info("CONVERGENCE COMPLETE")
    np.save(rdir + 'bic_list.npy', list_prev_bics)

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
    plt.savefig(rdir + 'bics.pdf')

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
    final_dir = rdir+'final/'
    mkpath(final_dir)

#         memb_probs_final = expectation(data, best_comps, best_memb_probs,
#                                        inc_posterior=inc_posterior)
    np.save(final_dir+'final_membership.npy', final_memb_probs)
    logging.info('Membership distribution:\n{}'.format(
        final_memb_probs.sum(axis=0)
    ))

    # SAVE FINAL RESULTS IN MAIN SAVE DIRECTORY
    Component.store_raw_components(final_dir+'final_comps.npy', final_best_comps)
    np.save(final_dir+'final_comps_bak.npy', final_best_comps)
    np.save(final_dir+'final_med_and_spans.npy', final_med_and_spans)

    overall_lnlike = get_overall_lnlikelihood(
            data, final_best_comps, inc_posterior=False
    )
    overall_lnposterior = get_overall_lnlikelihood(
            data, final_best_comps, inc_posterior=True
    )
    bic = calc_bic(data, ncomps, overall_lnlike,
                   memb_probs=final_memb_probs, Component=Component)
    logging.info("Final overall lnlikelihood: {}".format(overall_lnlike))
    logging.info("Final overall lnposterior:  {}".format(overall_lnposterior))
    logging.info("Final BIC: {}".format(bic))

    np.save(final_dir+'likelihood_post_and_bic.npy', (overall_lnlike,
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
    if not stable_state:
        log_message('BAD RUN TERMINATED', symbol='*', surround=True)

    logging.info(50*'=')

    return final_best_comps, np.array(final_med_and_spans), final_memb_probs
