"""
author: Marusa Zerjal 2020 - 04 - 23

Run naivefit in a multiprocessing mode. Fit components simultaneously,
i.e. perform splitting simultaneously.

run with
mpirun -np 4 python run_chronostar_multiprocessing.py example_runnaivefit_multiprocessing.pars

If you want to continue from the previous run, e.g. from '15', 
then find the best split of '15', e.g. '15/B' and copy '15/B/final'
to your new destination folder (new_folder/15/final/). This is where 
Chronostar reads the previous results from and loads them into Chronostar.

"""

from __future__ import print_function, division

import numpy as np
import os
import sys
import logging
from distutils.dir_util import mkpath
import random
import pickle

from multiprocessing import cpu_count

sys.path.insert(0, '..')
from chronostar.naivefit import NaiveFit
from chronostar import expectmax


import time
import itertools
import logging

from mpi4py import MPI

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5 * symbol, msg, 5 * symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50 * symbol, res, 50 * symbol)
    logging.info(res)


def build_comps_from_chains(run_dir, ncomps, fit_pars):
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

    ncomps = fit_pars['ncomps']

    comps = ncomps * [None]
    for i in range(ncomps):
        final_cdir = run_dir + 'final/comp{}/'.format(i)
        chain = np.load(final_cdir + 'final_chain.npy')
        lnprob = np.load(final_cdir + 'final_lnprob.npy')
        # TC: maybe fixed a bug?
        # npars = len(Component.PARAMETER_FORMAT)
        npars = len(fit_pars['Component'].PARAMETER_FORMAT)
        best_ix = np.argmax(lnprob)
        best_pars = chain.reshape(-1, npars)[best_ix]
        comps[i] = fit_pars['Component'](emcee_pars=best_pars)
    fit_pars['Component'].store_raw_components(
            str(run_dir + 'final/' + fit_pars['self.final_comps_file']),
            comps)

    return comps

if rank == 0:
    if len(sys.argv) != 2:
        raise UserWarning('Incorrect usage. Path to parameter file is required'
                          ' as a single command line argument. e.g.\n'
                          '   > python new_run_chronostar.py path/to/parsfile.par')

    # List of files with params/data for each decomposition
    filename_list = sys.argv[1]

    if not os.path.isfile(filename_list):
        raise UserWarning('Provided file does not exist', filename_list)

    filenames_all = np.load(filename_list)

    # SPLIT DATA into multiple processes
    filename_pars = np.array_split(filenames_all, size)

else:
    filename_pars = None

# Read params file
filename_pars = comm.scatter(filename_pars, root=0)
if len(filename_pars)==1:
    filename_pars=filename_pars[0]
else:
    print('FILENAME_PARS>1!!')
print('filename_pars', filename_pars)
with open(filename_pars, 'rb') as handle:
    fit_pars = pickle.load(handle)

run_dir = fit_pars['run_dir']

print('RUNNING PARALLEL run_em_unless_loadable')

try:
    med_and_spans = np.load(run_dir + 'final/'
                                 + fit_pars['self.final_med_and_spans_file'])
    memb_probs = np.load(
        run_dir + 'final/' + fit_pars['self.final_memb_probs_file'])
    comps = fit_pars['Component'].load_raw_components(
            str(run_dir + 'final/' + fit_pars['self.final_comps_file']))
    logging.info('Loaded from previous run')

    # Handle case where Component class has been modified and can't
    # load the raw components
except AttributeError:
    # TODO: check that the final chains looked for are guaranteed to be saved
    comps = build_comps_from_chains(run_dir, fit_pars)

    # Handle the case where files are missing, which means we must
    # perform the fit.
except IOError:
    comps, med_and_spans, memb_probs = \
        expectmax.fit_many_comps(**fit_pars) # TODO: CHECK IF THESE FIT_PARS AREN:T TOO BIG BECAUSE THEY CONTAIN DATA_DICT!

#~ result = {'div_label': fit_pars['div_label'], 'ncomps': fit_pars['ncomps'], 'comps':comps, 'med_and_spans':med_and_spans, 'memb_probs':memb_probs}
result = {'comps':comps, 'med_and_spans':med_and_spans, 'memb_probs':memb_probs}
with open(fit_pars['filename_result'], 'wb') as handle:
    pickle.dump(result, handle)
print('FINISHED EM rank', rank)
