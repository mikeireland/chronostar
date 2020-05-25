"""
Run the maximisation step for all components at once.
Use one process for each component (np is external parameter).
"""

#~ import subprocess # to call external scripts
import numpy as np
import pickle
import os
import sys
import itertools
from mpi4py import MPI

import sys
sys.path.insert(0, '..')
from chronostar import readparam

USE_C_IMPLEMENTATION = True
try:
    from chronostar._overlap import get_lnoverlaps as c_get_lnoverlaps # TODO: parallelise!!
except ImportError:
    print("C IMPLEMENTATION OF GET_OVERLAP NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    # Read fit parameters from the file        
    filename_data = sys.argv[1]
    filename_result = sys.argv[2]
    
    # Read data
    with open(filename_data, 'rb') as handle:
        d = pickle.load(handle)
    #~ d = np.load(filename_data, allow_pickle=True)
    #~ print(d)
    cov_now = d['cov_now']
    mean_now = d['mean_now']
    star_count = d['star_count']
    star_covs_all = d['star_covs']
    star_means_all = d['star_means']
    
    # Scatter data
    star_covs = np.array_split(star_covs_all, size)
    star_means = np.array_split(star_means_all, size)
    
    print('star_covs', star_covs)
    print('star_means', star_means)

else:
    cov_now = None
    mean_now = None
    star_count = None
    star_covs = None
    star_means = None

# BROADCAST CONSTANTS
cov_now = comm.bcast(cov_now, root=0)
mean_now = comm.bcast(mean_now, root=0)
star_count = comm.bcast(star_count, root=0)

# SCATTER DATA
star_covs = comm.scatter(star_covs, root=0)
star_means = comm.scatter(star_means, root=0)

print(rank, 'start')

# RUN OVERLAPS
if USE_C_IMPLEMENTATION:
    lnols = c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means,
                             star_count)
else:
    lnols = slow_get_lnoverlaps(cov_now, mean_now, star_covs, star_means)


# Gather results
result = comm.gather(lnols, root=0)
if rank == 0:
    result = list(itertools.chain.from_iterable(result))

    # WRITE a file
    np.savetxt(filename_result, result)
    
