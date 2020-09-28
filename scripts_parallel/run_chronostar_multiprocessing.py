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

from multiprocessing import cpu_count

sys.path.insert(0, '..')
from chronostar.naivefit import NaiveFit

import time
import itertools
import logging

#~ from schwimmbad import MPIPool
#~ from multiprocessing import Pool

from mpi4py import MPI

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

#~ pool = MPIPool()
#~ dir(pool)

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


if rank == 0:
    if len(sys.argv) != 2:
        raise UserWarning('Incorrect usage. Path to parameter file is required'
                          ' as a single command line argument. e.g.\n'
                          '   > python new_run_chronostar.py path/to/parsfile.par')

    fit_par_file = sys.argv[1]

    if not os.path.isfile(fit_par_file):
        raise UserWarning('Provided file does not exist')

    # Init: read the data etc.
    naivefit = NaiveFit(fit_pars=fit_par_file)
    naivefit.first_run_fit()

    # SPLIT DATA into multiple processes
    ncomps = len(naivefit.prev_result['comps'])
    comps = np.array_split(range(ncomps), size)

    #TODO: delete the time line
    print('Start')
    time_start = time.time()
else:
    naivefit = None
    comps = None
    ncomps = None # for sleep

#~ with MPIPool() as pool:
    #~ if not pool.is_master():
        #~ pool.wait()
        #~ sys.exit(0)

#~ pool=Pool(10)

#~ if not pool.is_master():
    #~ pool.wait()
    #~ sys.exit(0)

 #~ MPI.COMM_SELF.Intracomm.Spawn()

while True:
    # BROADCAST CONSTANTS
    naivefit = comm.bcast(naivefit, root=0)  # updated ncomps, prev_results, prev_score
    ncomps = comm.bcast(ncomps, root=0)  # updated ncomps, prev_results, prev_score

    # SCATTER DATA; this will need to be reiterated when a new component is added
    comps = comm.scatter(comps, root=0)
    print('RRRrank', rank)
    if rank < ncomps:
        print('rank', rank)
        # EVERY PROCESS DOES THIS FOR ITS DATA
        all_results_rank = []
        all_scores_rank = []
        for comp in comps:
            result, score = naivefit.run_split_for_one_comp_multiproc(i=comp) # , pool=pool
            all_results_rank.append(result)
            all_scores_rank.append(score)

        # GATHER DATA AND UPDATE NAIVEFIT
        all_results_tmp = comm.gather(all_results_rank, root=0)
        all_scores_tmp = comm.gather(all_scores_rank, root=0)

    if rank == 0:
        all_results = list(itertools.chain.from_iterable(all_results_tmp))
        all_scores = list(itertools.chain.from_iterable(all_scores_tmp))
        
        # 'terminate' tells the loop to terminate if the fit converged.
        terminate = naivefit.run_fit_gather_results_multiproc(all_results, all_scores)
        
        # Introduce a new component. Split comp fitting into multiple processes
        ncomps = len(naivefit.prev_result['comps'])
        comps = np.array_split(range(ncomps), size)
        
        if naivefit.ncomps >= naivefit.fit_pars['max_comp_count']:
            terminate=True
            print('REACHED MAX COMP LIMIT')
            log_message(msg='REACHED MAX COMP LIMIT', symbol='+',
                        surround=True)
    else:
        terminate=None
        naivefit=None
        comps=None
        ncomps=None # for sleep

    terminate = comm.bcast(terminate, root=0)

    if terminate:
        break
    
#~ pool.close()
if rank == 0:
    time_end = time.time()
    print('DONE.', time_end - time_start)
