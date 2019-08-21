#!/usr/bin/env python3
"""
Primary Chronostar script.

Perform a kinematic fit to data as described in Crundall et al. (2019).

Run this script with either simple argument
line inputs (call with --help to see options) or with a single input
config file. Unzip contents of config_examples.zip to see some examples.
"""

from __future__ import print_function, division, unicode_literals

# prevent plots trying to display (and breaking runs on servers)
try:
    import matplotlib as mpl
    mpl.use('Agg')
except ImportError:
    pass

import argparse
import numpy as np
import os
import sys
from emcee.utils import MPIPool
import logging
import imp      # TODO: address deprecation of imp
from distutils.dir_util import mkpath
import random
import time

from get_association_region import get_region
sys.path.insert(0, os.path.abspath('..'))
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import compfitter
from chronostar import expectmax


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
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


# Check if single input is provided, and treat as config file
# at the moment config file needs to be in same directory...?
if len(sys.argv) == 2:
    config_name = sys.argv[1]
    config = imp.load_source(config_name.replace('.py', ''), config_name)
    # config = importlib.import_module(config_name.replace('.py', ''), config_name)


# Check results directory is valid
# If path exists, make a new results_directory with a random int
if os.path.exists(config.config['results_dir']) and\
        not config.config['overwrite_prev_run']:
    rdir = '{}_{}'.format(config.config['results_dir'].rstrip('/'),
                          random.randint(0,1000))
else:
    rdir = config.config['results_dir']
rdir = rdir.rstrip('/') + '/'
mkpath(rdir)

# Now that results directory is set up, can set up log file
logging.basicConfig(filename=rdir+'log.log', level=logging.INFO)

# ------------------------------------------------------------
# -----  BEGIN MPIRUN THING  ---------------------------------
# ------------------------------------------------------------

# Only even try to use MPI if config file says so
using_mpi = config.config.get('run_with_mpi', False)
if using_mpi:
    try:
        pool = MPIPool()
        logging.info("Successfully initialised mpi pool")
    except:
        #print("MPI doesn't seem to be installed... maybe install it?")
        logging.info("MPI doesn't seem to be installed... maybe install it?")
        using_mpi = False
        pool=None

if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        # Wait for instructions from the master process.
        pool.wait()
        sys.exit(0)

time.sleep(5)
print("Only one thread is master! (if not, ensure config "
      "file has run_with_mpi=True)")

log_message('Beginning Chronostar run',
            symbol='_', surround=True)

log_message('Setting up', symbol='.', surround=True)

assert os.access(rdir, os.W_OK)

# ------------------------------------------------------------
# -----  SETTING UP ALL DATA PREP  ---------------------------
# ------------------------------------------------------------

# Set up some filename constants
final_comps_file = 'final_comps.npy'
final_med_and_spans_file = 'final_med_and_spans.npy'
final_memb_probs_file = 'final_membership.npy'

# First see if a data savefile path has been provided, and if
# so, then just assume this script has already been performed
# and the data prep has already been done
if (config.config['data_savefile'] != '' and
        os.path.isfile(config.config['data_savefile'])):
    log_message('Loading pre-prepared data')
    datafile = config.config['data_savefile']
    data_table = tabletool.load(datafile)
    historical = 'c_XU' in data_table.colnames

# Otherwise, perform entire process
else:
    # Construct synthetic data if required
    if config.synth is not None:
        log_message('Getting synthetic data')
        datafile = config.config['data_savefile']
        if not os.path.exists(datafile) and config.config['pickup_prev_run']:
            synth_data = SynthData(pars=config.synth['pars'],
                                   starcounts=config.synth['starcounts'],
                                   Components=Component)
            synth_data.synthesise_everything(filename=datafile,
                                             overwrite=True)
            np.save(rdir+'true_synth_pars.npy', config.synth['pars'])
            np.save(rdir+'true_synth_starcounts.npy', config.synth['starcounts'])
        else:
            log_message('Synthetic data already exists')
    else:
        log_message('Setting datafile name '+'{}'.format(config.config['data_loadfile']))
        log_message('Current dir '+os.getcwd())
        datafile = config.config['data_loadfile']
    assert os.path.exists(datafile)

    # Read in data as table
    log_message('Read data into table')
    data_table = tabletool.read(datafile)

    historical = 'c_XU' in data_table.colnames

    # If data cuts provided, then apply them
    if config.config['banyan_assoc_name'] != '':
        bounds = get_region(
                config.config['banyan_assoc_name'],
                pos_margin=config.advanced.get('pos_margin', 30.),
                vel_margin=config.advanced.get('vel_margin', 5.),
                scale_margin=config.advanced.get('scale_margin', None),
        )
    elif config.data_bound is not None:
        bounds = (config.data_bound['lower_bound'],
                  config.data_bound['upper_bound'])
    else:
        bounds = None

    if bounds is not None:
        log_message('Applying data cuts')
        star_means = tabletool.build_data_dict_from_table(
                datafile,
                main_colnames=config.cart_colnames.get('main_colnames', None),
                only_means=True,
                historical=historical,
        )
        data_mask = np.where(
                np.all(star_means < bounds[1], axis=1)
                & np.all(star_means > bounds[0], axis=1))
        data_table = data_table[data_mask]
    log_message('Data table has {} rows'.format(len(data_table)))


    # By the end of this, data will be a astropy table
    # with cartesian data written in
    # columns in default way.
    if config.config['convert_to_cartesian']:
        log_message('Trying to convert to cartesian')
        # Performs conversion in place (in memory) on `data_table`
        if (not 'c_XU' in data_table.colnames and
            not 'X_U_corr' in data_table.colnames):
            log_message('Converting to cartesian')
            tabletool.convert_table_astro2cart(
                    table=data_table,
                    main_colnames=config.astro_colnames.get('main_colnames', None),
                    error_colnames=config.astro_colnames.get('error_colnames', None),
                    corr_colnames=config.astro_colnames.get('corr_colnames', None),
                    return_table=False
            )

    # Calculate background overlaps, storing in data
    bg_lnol_colname = 'background_log_overlap'
    if config.config['include_background_distribution']:
        # Only calculate if missing
        if bg_lnol_colname not in data_table.colnames:
            # TODO: by circumstance the means don't get masked for bad data...
            # which is what we want here. But it is ugly
            log_message('Calculating background densities')
            background_means = tabletool.build_data_dict_from_table(
                    config.config['kernel_density_input_datafile'],
                    only_means=True,
            )
            star_means = tabletool.build_data_dict_from_table(
                    data_table, only_means=True,
            )
            ln_bg_ols = expectmax.get_kernel_densities(background_means,
                                                       star_means, )

            # If allowed, save to original file path
            if config.config['overwrite_datafile']:
                tabletool.insert_column(data_table,
                                        col_data=ln_bg_ols,
                                        col_name=bg_lnol_colname,
                                        filename=datafile)
            else:
                tabletool.insert_column(data_table,
                                        col_data=ln_bg_ols,
                                        col_name=bg_lnol_colname)

    if config.config['overwrite_datafile']:
        data_table.write(datafile, overwrite=True)
    elif config.config['data_savefile'] != '':
        data_table.write(config.config['data_savefile'], overwrite=True)

if historical:
    log_message('Data set already has historical cartesian columns')


# Convert data table into numpy arrays of mean and covariance matrices
log_message('Building data dictionary')
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=config.config['include_background_distribution'],
        historical=historical,
)

# ------------------------------------------------------------
# -----  SETTING UP DEFAULT RUN VARS  ------------------------
# ------------------------------------------------------------
ncomps = 1

MAX_COMPS = config.special.get('max_component_count', 20)         # Set a ceiling on how long code can run for
MAX_ITERS = config.special.get('max_em_iterations', 100)
log_message(msg='Component count cap set to {}'.format(MAX_COMPS),
        symbol='+', surround=True)
log_message(msg='Iteration count cap set to {}'.format(MAX_ITERS),
        symbol='+', surround=True)

# ------------------------------------------------------------
# -----  SETTING UP RUN CUSTOMISATIONS  ----------------------
# ------------------------------------------------------------

# Set up trace_orbit_func
if config.config['dummy_trace_orbit_function']:
    trace_orbit_func = dummy_trace_orbit_func
else:
    trace_orbit_func = None

# Import suitable component class
if config.special['component'].lower() == 'sphere':
    from chronostar.component import SphereComponent as Component
elif config.special['component'].lower() == 'ellip':
    from chronostar.component import EllipComponent as Component
# elif config.special['component'].lower() == 'free':
#     from chronostar.component import FreeComponent as Component
else:
    raise UserWarning('Unknown (or missing) component parametrisation')

if config.config['init_comps_file'] is not None:
    init_comps = Component.load_raw_components(config.config['init_comps_file'])
    ncomps = len(init_comps)
    print('Managed to load in init_comps from file')
else:
    init_comps = None
    print("'Init comps' is initialised as none")

# ------------------------------------------------------------
# -----  EXECUTE RUN  ----------------------------------------
# ------------------------------------------------------------

# Moved here by Marusa because if was defined only in case ncomps==1
store_burnin_chains = config.advanced.get('store_burnin_chains', False)
if store_burnin_chains:
    log_message(msg='Storing burnin chains', symbol='-')

if ncomps == 1:
    # Fit the first component
    log_message(msg='FITTING {} COMPONENT'.format(ncomps),
                symbol='*', surround=True)
    run_dir = rdir + '{}/'.format(ncomps)

    # Initialise all stars in dataset to be full members of first component
    using_bg = config.config.get('include_background_distribution', False)
    init_memb_probs = np.zeros((len(data_dict['means']),1+using_bg))
    init_memb_probs[:,0] = 1.

    # Try and recover any results from previous run
    try:
        prev_med_and_spans = np.load(run_dir + 'final/'
                                     + final_med_and_spans_file)
        prev_memb_probs = np.load(run_dir + 'final/' + final_memb_probs_file)
        try:
            prev_comps = Component.load_raw_components(
                    str(run_dir+'final/'+final_comps_file))
        # Final comps are there, they just can't be read by current module
        # so quickly fit them based on fixed prev membership probabilities
        except AttributeError:
            logging.info('Component class has been modified, reconstructing '
                         'from chain')
            prev_comps = ncomps * [None]
            for i in range(ncomps):
                final_cdir = run_dir + 'final/comp{}/'.format(i)
                chain = np.load(final_cdir + 'final_chain.npy')
                lnprob = np.load(final_cdir + 'final_lnprob.npy')
                npars = len(Component.PARAMETER_FORMAT)
                best_ix = np.argmax(lnprob)
                best_pars = chain.reshape(-1,npars)[best_ix]
                prev_comps[i] = Component(emcee_pars=best_pars)
            Component.store_raw_components(str(run_dir+'final/'+final_comps_file),
                                           prev_comps)
            # np.save(str(run_dir+'final/'+final_comps_file), prev_comps)

        logging.info('Loaded from previous run')
    except IOError:
        prev_comps, prev_med_and_spans, prev_memb_probs = \
            expectmax.fit_many_comps(data=data_dict, ncomps=ncomps, rdir=run_dir,
                                     trace_orbit_func=trace_orbit_func,
                                     burnin=config.advanced['burnin_steps'],
                                     sampling_steps=config.advanced['sampling_steps'],
                                     use_background=config.config[
                                        'include_background_distribution'],
                                     init_memb_probs=init_memb_probs,
                                     init_comps=init_comps,
                                     Component=Component,
                                     store_burnin_chains=store_burnin_chains,
                                     max_iters=MAX_ITERS,
                                     )


    # Calculate global score of fit for comparison with future fits with different
    # component counts
    prev_lnlike = expectmax.get_overall_lnlikelihood(data_dict, prev_comps,
                                                     # bg_ln_ols=bg_ln_ols,
                                                     )
    prev_lnpost = expectmax.get_overall_lnlikelihood(data_dict, prev_comps,
                                                     # bg_ln_ols=bg_ln_ols,
                                                     inc_posterior=True)
    prev_bic = expectmax.calc_bic(data_dict, ncomps, prev_lnlike,
                                  memb_probs=prev_memb_probs,
                                  Component=Component)

    ncomps += 1

if init_comps is not None and len(init_comps) > 1:
    # fit with ncomps until convergence
    # Note: might
    run_dir = rdir + '{}/'.format(ncomps)
    comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(
            data=data_dict, ncomps=ncomps, rdir=run_dir,
            init_comps=init_comps, trace_orbit_func=trace_orbit_func,
            use_background=config.config[
                'include_background_distribution'],
            burnin=config.advanced['burnin_steps'],
            sampling_steps=config.advanced['sampling_steps'],
            Component=Component,
            store_burnin_chains=store_burnin_chains,
            max_iters=MAX_ITERS,
        )
    ncomps += 1

# Begin iterative loop, each time trialing the incorporation of a new component
while ncomps <= MAX_COMPS:
    if ncomps >= MAX_COMPS:
        log_message(msg='REACHED MAX COMP LIMIT', symbol='+', surround=True)
        break

    log_message(msg='FITTING {} COMPONENT'.format(ncomps),
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
        run_dir = rdir + '{}/{}/'.format(ncomps, div_label)
        log_message(msg='Subdividing stage {}'.format(div_label),
                    symbol='+', surround=True)
        mkpath(run_dir)

        assert isinstance(target_comp, Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run
        split_comps = target_comp.splitGroup(lo_age=prev_med_and_spans[i,-1,1],
                                             hi_age=prev_med_and_spans[i,-1,2])
        init_comps = list(prev_comps)
        init_comps.pop(i)
        init_comps.insert(i, split_comps[1])
        init_comps.insert(i, split_comps[0])

        # Run em fit
        # First try and find any previous runs
        try:
            med_and_spans = np.load(run_dir + 'final/'
                                    + final_med_and_spans_file)
            memb_probs = np.load(run_dir + 'final/' + final_memb_probs_file)
            try:
                comps = Component.load_raw_components(run_dir + 'final/'
                                                      + final_comps_file)
            # Final comps are there, they just can't be read by current module
            # so quickly fit them based on fixed prev membership probabilities
            except AttributeError:
                logging.info(
                    'Component class has been modified, reconstructing from'
                    'chain.')
                prev_comps = ncomps * [None]
                for i in range(ncomps):
                    final_cdir = run_dir + 'final/comp{}/'.format(i)
                    chain = np.load(final_cdir + 'final_chain.npy')
                    lnprob = np.load(final_cdir + 'final_lnprob.npy')
                    npars = len(Component.PARAMETER_FORMAT)
                    best_ix = np.argmax(lnprob)
                    best_pars = chain.reshape(-1, npars)[best_ix]
                    prev_comps[i] = Component(emcee_pars=best_pars)
                Component.store_raw_components(str(run_dir+'final/'+final_comps_file),
                                               prev_comps)
                # np.save(str(run_dir + 'final/' + final_comps_file), prev_comps)

            logging.info('Fit loaded from previous run')
        except IOError:
            comps, med_and_spans, memb_probs = \
            expectmax.fit_many_comps(
                    data=data_dict, ncomps=ncomps, rdir=run_dir,
                    init_comps=init_comps, trace_orbit_func=trace_orbit_func,
                    use_background=config.config[
                        'include_background_distribution'],
                    burnin=config.advanced['burnin_steps'],
                    sampling_steps=config.advanced['sampling_steps'],
                    Component=Component,
                    store_burnin_chains=store_burnin_chains,
                    max_iters=MAX_ITERS,
            )

        best_fits.append(comps)
        all_med_and_spans.append(med_and_spans)
        all_memb_probs.append(memb_probs)
        lnlikes.append(expectmax.get_overall_lnlikelihood(data_dict, comps))
        lnposts.append(
                expectmax.get_overall_lnlikelihood(data_dict, comps,
                                                   inc_posterior=True)
        )
        bics.append(expectmax.calc_bic(data_dict, ncomps, lnlikes[-1],
                                       memb_probs=memb_probs,
                                       Component=Component))
        logging.info('Decomposition {} finished with \nBIC: {}\nlnlike: {}\n'
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
    logging.info("Turned\n{}".format(prev_comps[best_split_ix].get_pars()))
    logging.info('with {} members'.format(prev_memb_probs.sum(axis=0)[best_split_ix]))
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
        logging.info("New BIC: {} < Old BIC: {}".format(new_bic, prev_bic))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        prev_comps, prev_med_and_spans, prev_memb_probs, prev_lnlike, prev_lnpost, \
        prev_bic = \
            (new_comps, new_meds, new_z, new_lnlike, new_lnpost, new_bic)
        ncomps += 1
        log_message(msg="Commencing {} component fit on {}{}".format(
                ncomps, ncomps-1,
                chr(ord('A') + best_split_ix)), symbol='+'
        )
    else:
        logging.info("Extra component has worsened BIC...")
        logging.info("New BIC: {} > Old BIC: {}".format(new_bic, prev_bic))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        logging.info("... saving previous fit as best fit to data")
        Component.store_raw_components(rdir + final_comps_file, prev_comps)
        # np.save(rdir + final_comps_file, prev_comps)
        np.save(rdir + final_med_and_spans_file, prev_med_and_spans)
        np.save(rdir + final_memb_probs_file, prev_memb_probs)
        np.save(rdir + 'final_likelihood_post_and_bic',
                [prev_lnlike, prev_lnpost,
                 prev_bic])
        logging.info('Final best fits:')
        [logging.info(c.get_pars()) for c in prev_comps]
        logging.info('Final age med and span:')
        [logging.info(row[-1]) for row in prev_med_and_spans]
        logging.info('Membership distribution: {}'.format(prev_memb_probs.sum(axis=0)))
        logging.info('Final membership:')
        logging.info('\n{}'.format(np.round(prev_memb_probs * 100)))
        logging.info('Final lnlikelihood: {}'.format(prev_lnlike))
        logging.info('Final lnposterior:  {}'.format(prev_lnpost))
        logging.info('Final BIC: {}'.format(prev_bic))
        break

    logging.info("Best fit:\n{}".format(
            [group.get_pars() for group in prev_comps]))

# TODO: using_mpi is not defined if you don't use MPI.
#  Try-except is not the best thing here but will do for now.
try:
    if using_mpi:
        pool.close()
except:
    pass
