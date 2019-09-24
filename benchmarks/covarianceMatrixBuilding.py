"""
OUTDATED
TODO: If deemed important, Tim to come back and update this

Compare to find the fastest means to construct covariance matrices
from data stored in columns

Sample output:
>> Loading ../data/marusa_galah_li_strong_stars_xyzuvw.fits...
>> Reading in took 0.367851 seconds
>> Testing on 2416 stars
>> einsum took 0.003273 seconds
>> tiling took 0.00862 seconds
>> iterative took 1.418085 seconds
>> Loading ../data/gaia_cartesian_full_6d_table.fits...
>> Reading in took 13.310633 seconds
>> Testing on 6376803 stars
>> einsum took 12.041975 seconds
>> tiling took 8.953329 seconds
>> Too many stars to bother with iterative timings
"""

import numpy as np
from astropy.table import Table
import time

import sys
sys.path.insert(0, '..')

from chronostar.tabletool import buildXYZUVWCovMatrices

def buildMeanAndCovMatFromRow(row):
    """
    Build a covariance matrix from a row

    Paramters
    ---------
    row : astropy Table row
        Entries: {X, Y, Z, U, V, W, dX, dY, ..., cXY, cXZ, ...}

    Return
    ------
    cov_mat : [6,6] numpy array
        Diagonal elements are dX^2, dY^2, ...
        Off-diagonal elements are cXY*dX*dY, cXZ*dX*dZ, ...
    """
    dim = 6
    CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
                      'dX', 'dY',   'dZ',   'dU',   'dV',   'dW',
                            'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                                    'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                            'c_ZU', 'c_ZV', 'c_ZW',
                                                    'c_UV', 'c_UW',
                                                            'c_VW']
    mean = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[:6]):
        mean[i] = row[col_name]

    std_vec = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[6:12]):
        std_vec[i] = row[col_name]
    corr_tri = np.zeros((dim,dim))
    # Insert upper triangle (top right) correlations
    for i, col_name in enumerate(CART_COL_NAMES[12:]):
        corr_tri[np.triu_indices(dim,1)[0][i],np.triu_indices(dim,1)[1][i]]\
            =row[col_name]
    # Build correlation matrix
    corr_mat = np.eye(6) + corr_tri + corr_tri.T
    # Multiply through by standard deviations
    cov_mat = corr_mat * std_vec * std_vec.reshape(6,1)
    return mean, cov_mat


def iterativeBuildCovMatrix(data):
    covs = []
    for row in data:
        mean, cov = buildMeanAndCovMatFromRow(row)
        covs.append(cov)
    return np.array(covs)


def alternativeBuildCovMatrix(data):
    nstars = len(data)
    cls = np.array([
        ['X', 'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW'],
        ['c_XY', 'Y', 'c_YZ', 'c_YU', 'c_YV', 'c_YW'],
        ['c_XZ', 'c_YZ', 'Z', 'c_ZU', 'c_ZV', 'c_ZW'],
        ['c_XU', 'c_YU', 'c_ZU', 'U', 'c_UV', 'c_UW'],
        ['c_XV', 'c_YV', 'c_ZV', 'c_UV', 'V', 'c_VW'],
        ['c_XW', 'c_YW', 'c_ZW', 'c_UW', 'c_VW', 'W'],
    ])

    # Construct an [nstars,6,6] array of identity matrices
    covs = np.zeros((nstars,6,6))
    idx = np.arange(6)
    covs[:, idx, idx] = 1.0

    # Insert correlations into off diagonals
    for i in range(0,5):
        for j in range(i+1,5):
            covs[:,i,j] = covs[:,j,i] = data[cls[i,j]]

    # multiply each row and each column by appropriate error
    for i in range(6):
        covs[:,i,:] *= np.tile(data[cls[i,i]], (6,1)).T
        covs[:,:,i] *= np.tile(data[cls[i,i]], (6,1)).T

    return covs

if __name__ == '__main__':
    filenames = ['../data/marusa_galah_li_strong_stars_xyzuvw.fits',
                 '../data/gaia_cartesian_full_6d_table.fits']
    for filename in filenames:
        print("Loading {}...".format(filename))
        t_start = time.clock()
        table = Table.read(filename)
        t_end = time.clock()
        print("Reading in took {} seconds".format(t_end - t_start))
        nstars = len(table)

        functions = [buildXYZUVWCovMatrices,
                     alternativeBuildCovMatrix,
                     iterativeBuildCovMatrix]
        names = ['einsum', 'tiling', 'iterative']

        print("Testing on {} stars".format(len(table)))
        for name, func in zip(names, functions):
            if (name == names[2] and nstars > 5000):
                print("Too many stars to bother with iterative timings")
            else:
                t_start = time.clock()
                func(table)
                t_end = time.clock()
                print('{:10} took {} seconds'.format(name, t_end - t_start))

