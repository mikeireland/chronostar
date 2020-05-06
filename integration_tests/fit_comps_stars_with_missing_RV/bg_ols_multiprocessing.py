"""
author: Marusa Zerjal 2019 - 07 - 29

Determine background overlaps using means and covariances for both
background and stars.
Covariance matrices for the background are Identity*bandwidth.

Parameters
----------
background_means: [nstars,6] float array_like
    Phase-space positions of some star set that greatly envelops points
    in question. Typically contents of gaia_xyzuvw.npy, or the output of
    >> tabletool.build_data_dict_from_table(
               '../data/gaia_cartesian_full_6d_table.fits',
                historical=True)['means']
star_means: [npoints,6] float array_like
    Phase-space positions of stellar data that we are fitting components to
star_covs: [npoints,6,6] float array_like
    Phase-space covariances of stellar data that we are fitting components to

Output is a file with ln_bg_ols. Same order as input datafile.
No return.

bg_lnols: [nstars] float array_like
    Background log overlaps of stars with background probability density
    function.

Notes
-----
We invert the vertical values (Z and U) because the typical background
density should be symmetric along the vertical axis, and this distances
stars from their siblings. I.e. association stars aren't assigned
higher background overlaps by virtue of being an association star.

Edits
-----
TC 2019-05-28: changed signature such that it follows similar usage as
               get_kernel_densitites
"""

from __future__ import print_function, division

import numpy as np
import itertools
from mpi4py import MPI

import time

import logging

# The placement of logsumexp varies wildly between scipy versions
import scipy
_SCIPY_VERSION= [int(v.split('rc')[0])
                 for v in scipy.__version__.split('.')]
if _SCIPY_VERSION[0] == 0 and _SCIPY_VERSION[1] < 10:
    from scipy.maxentropy import logsumexp
elif ((_SCIPY_VERSION[0] == 1 and _SCIPY_VERSION[1] >= 3) or
    _SCIPY_VERSION[0] > 1):
    from scipy.special import logsumexp
else:
    from scipy.misc import logsumexp


import sys
sys.path.insert(0, '..')
from chronostar import tabletool

from astropy.table import Table

try:
    print('Using C implementation')
    #from _overlap import get_lnoverlaps
    from chronostar._overlap import get_lnoverlaps
except:
    print("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    logging.info("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    from chronostar.likelihood import slow_get_lnoverlaps as get_lnoverlaps


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()


if rank == 0:
    # PREPARE STELLAR DATA
    #~ datafile = '/priv/mulga1/marusa/chronostar/data/ScoCen_box_result_15M_ready_for_bg_ols.fits'
    datafile = 'data/beta_Pictoris_with_gaia_small_everything_final.fits'
    data_table = tabletool.read(datafile)
    historical = 'c_XU' in data_table.colnames
    #data_table = data_table[:20] #TODO for testing
    print('DATA_TABLE READ', len(data_table))

    """
    # Compute overlaps only for the part of the data (chunk)
    # Every 100k stars take about 2 days, so I only want about that many stars in each run, in case something
    # goes wrong
    N=1 # that many chunks
    NI=1#int(sys.argv[1]) # take this chunk #TODO: update this number for every run!
    print('NI=%d'%NI)
    # TAKE ONLY the i-th part of the data
    indices_chunks = np.array_split(range(len(data_table)), N)
    data_table=data_table[indices_chunks[NI]]
    """

    data_dict, good_row_mask = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=False, # bg overlap not available yet
        historical=historical,
        return_good_row_mask=True,
    )
    star_means = data_dict['means']
    star_covs = data_dict['covs']

    # PREPARE BACKGROUND DATA
    print('Read background Gaia data')
    background_means = tabletool.build_data_dict_from_table(
        '/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits',
        only_means=True,
    )

    # Inverting the vertical values
    star_means = np.copy(star_means)
    star_means[:, 2] *= -1
    star_means[:, 5] *= -1

    # Background covs with bandwidth using Scott's rule
    d = 6.0 # number of dimensions
    nstars = background_means.shape[0]
    bandwidth = nstars**(-1.0 / (d + 4.0))
    background_cov = np.cov(background_means.T) * bandwidth ** 2
    background_covs = np.array(nstars * [background_cov]) # same cov for every star


    # SPLIT DATA into multiple processes
    indices_chunks = np.array_split(range(len(star_means)), size)
    star_means = [star_means[i] for i in indices_chunks]
    star_covs = [star_covs[i] for i in indices_chunks]

    #TODO: delete the time line
    print('Start')
    time_start = time.time()
else:
    nstars=None
    star_means=None
    star_covs=None
    background_means=None
    background_covs=None

# BROADCAST CONSTANTS
nstars = comm.bcast(nstars, root=0)
background_means = comm.bcast(background_means, root=0)
background_covs = comm.bcast(background_covs, root=0)

# SCATTER DATA
star_means = comm.scatter(star_means, root=0)
star_covs = comm.scatter(star_covs, root=0)

#print(rank, len(star_means))

# EVERY PROCESS DOES THIS FOR ITS DATA
bg_ln_ols=[]
for star_cov, star_mean in zip(star_covs, star_means):
    try:
        bg_lnol = get_lnoverlaps(star_cov, star_mean, background_covs,
                                 background_means, nstars)
        bg_lnol = logsumexp(bg_lnol)  # sum in linear space
    except:
        # TC: Changed sign to negative (surely if it fails, we want it to
        # have a neglible background overlap?
        print('bg ln overlap failed, setting it to -inf')
        bg_lnol = -np.inf

    bg_ln_ols.append(bg_lnol)
#print(rank, bg_ln_ols)



# GATHER DATA
bg_ln_ols_result = comm.gather(bg_ln_ols, root=0)
if rank == 0:
    bg_ln_ols_result = list(itertools.chain.from_iterable(bg_ln_ols_result))
    
    tab = Table.read(datafile)
    
    print(len(tab), len(bg_ln_ols_result))
    
    tab['background_log_overlap'] = [np.nan]*len(tab)
    tab['background_log_overlap'][good_row_mask] = bg_ln_ols_result
    tab.write(datafile, overwrite=True)
    print('%s written.'%datafile)
    
    time_end = time.time()
    print(rank, 'done', time_end - time_start)
    #print('master collected: ', bg_ln_ols_result)

