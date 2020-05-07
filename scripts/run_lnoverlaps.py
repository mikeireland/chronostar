"""
Get lnoverlaps: this is time consuming and should be parallelised.
"""

"""
Given the parametric description of an origin, calculate star overlaps

Utilises Overlap, a c module wrapped with swig to be callable by python.
This allows a 100x speed up in our 6x6 matrix operations when compared
to numpy.

Parameters
----------
pars: [npars] list
    Parameters describing the origin of group
    typically [X,Y,Z,U,V,W,np.log(dX),np.log(dV),age]
data: dict
    stellar cartesian data being fitted to, stored as a dict:
    'means': [nstars,6] float array
        the central estimates of each star in XYZUVW space
    'covs': [nstars,6,6] float array
        the covariance of each star in XYZUVW space
star_mask: [len(data)] indices
    A mask that excludes stars that have negliglbe membership probablities
    (and thus have their log overlaps scaled to tiny numbers).
comps: [ncomps] list of Component objects
    
"""

import numpy as np
import sys
sys.path.insert(0, '..')

USE_C_IMPLEMENTATION = True
try:
    from chronostar._overlap import get_lnoverlaps as c_get_lnoverlaps
except ImportError:
    print("C IMPLEMENTATION OF GET_OVERLAP NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

from chronostar.component import SphereComponent as Component # TODO:
from chronostar import tabletool
from chronostar import traceorbit
from chronostar import readparam

import itertools
from mpi4py import MPI

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

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

def slow_get_lnoverlaps(g_cov, g_mn, st_covs, st_mns, dummy=None):
    """
    A pythonic implementation of overlap integral calculation.
    Left here in case swigged _overlap doesn't work.

    Paramters
    ---------
    g_cov: ([6,6] float array)
        Covariance matrix of the group
    g_mn: ([6] float array)
        mean of the group
    st_covs: ([nstars, 6, 6] float array)
        covariance matrices of the stars
    st_mns: ([nstars, 6], float array)
        means of the stars
    dummy: {None}
        a place holder parameter such that this function's signature
        matches that of the c implementation, which requires an
        explicit size of `nstars`.

    Returns
    -------
    ln_ols: ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    lnols = []
    for st_cov, st_mn in zip(st_covs, st_mns):
        res = 0
        res -= 6 * np.log(2*np.pi)
        res -= np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        res -= np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        res *= 0.5
        lnols.append(res)
    return np.array(lnols)


if rank == 0:
    # Read fit parameters from the file
    fit_pars = readparam.readParam(sys.argv[1], default_pars=DEFAULT_FIT_PARS)
    
    #~ filename_components = idir + 'best_comps.npy'
    filename_components = 'results3/final_comps.npy'

    # Prepare data
    data = tabletool.build_data_dict_from_table(fit_pars['data_table'])
    star_mask=None
    
    comp = Component.load_raw_components(filename_components)[0] # TODO

    # Prepare star arrays
    if star_mask is not None:
        star_means = data['means'][star_mask]
        star_covs = data['covs'][star_mask]
    else:
        star_means = data['means']
        star_covs = data['covs']

    # Get current day projection of component
    mean_now, cov_now = comp.get_currentday_projection() # This projection should maybe be done somewhere else not here?

    # SPLIT DATA into multiple processes
    star_covs = np.array_split(star_covs, size)
    star_means = np.array_split(star_means, size)
else:
    cov_now = None
    mean_now = None
    star_covs = None
    star_means = None

# BROADCAST CONSTANTS
cov_now = comm.bcast(cov_now, root=0)
mean_now = comm.bcast(mean_now, root=0)

# SCATTER DATA
star_covs = comm.scatter(star_covs, root=0)
star_means = comm.scatter(star_means, root=0)

star_count = len(star_means)

# Calculate overlap integral of each star
if USE_C_IMPLEMENTATION:
    lnols = c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means, star_count)
else:
    lnols = slow_get_lnoverlaps(cov_now, mean_now, star_covs, star_means)

# GATHER DATA
all_lnols_rank = comm.gather(lnols, root=0)
if rank == 0:
    all_lnols = list(itertools.chain.from_iterable(all_lnols_rank))

    # TODO: print, save, ...

