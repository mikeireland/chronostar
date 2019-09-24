#! /usr/bin/env python

"""
    A test program completely separate from main chronostar for astr3005,
    in order to test out the performance of swig and suitability of
    the code to find overlap in a simple test.
"""

import sys
sys.path.insert(0,'..')
import time
import argparse

if sys.version[0] == '2':
    timer = time.clock
elif sys.version[0] == '3':
    timer = time.perf_counter

import numpy as np
import chronostar._overlap as overlap

def compute_overlap(A,a,A_det,B,b,B_det):
    """Compute the overlap integral between a star and group mean + covariance matrix
    in six dimensions, including some temporary variables for speed and to match the 
    notes.
    
    This is the first function to be converted to a C program in order to speed up."""
    #Preliminaries - add matrices together. This might make code more readable? 
    #Or might not.
    ApB = A + B
    AapBb = np.dot(A,a) + np.dot(B,b)
    
    #Compute determinants.
    ApB_det = np.linalg.det(ApB)
    
    #Error checking (not needed in C once shown to work?) This shouldn't ever happen, as 
    #the determinants of the sum of positive definite matrices is
    #greater than the sum of their determinants    
    if (ApB_det < 0) | (B_det<0):
        return -np.inf
    
    #Solve for c
    c = np.linalg.solve(ApB, AapBb)
    
    #Compute the overlap formula.
    overlap = np.exp(-0.5*(np.dot(b-c,np.dot(B,b-c)) + \
                           np.dot(a-c,np.dot(A,a-c)) )) 
    overlap *= np.sqrt(B_det*A_det/ApB_det)/(2*np.pi)**3.0
    
    return overlap

def correctness(group_icov, group_mean, group_icov_det, star_icovs,
                                  star_means, star_icov_dets, nstars):
    """
        Displays the result for each function, no differences should
        occur.
    """
    
    # Using swig-numpy module with multiple stars per call
    swig_np_ms_ols = overlap.get_overlaps(
        group_icov, group_mean, group_icov_det, star_icovs,
        star_means, star_icov_dets, nstars)


    # Using swig-numpy module with multiple stars per call and log results
    group_cov = np.linalg.inv(group_icov)
    star_covs = np.zeros(star_icovs.shape)
    for i in range(nstars):
        star_covs[i] = np.linalg.inv(star_icovs[i])
    swig_np_ms_lnols = overlap.get_lnoverlaps(
        group_cov, group_mean, star_covs,
        star_means, nstars
    )

    # Compare with various implementations of calculating one star at a time
    for i in range(nstars):
        # Using numpy
        numpy_ols = compute_overlap(
            group_icov, group_mean, group_icov_det,
            star_icovs[i], star_means[i], star_icov_dets[i])

        # Using swig module (input passed as python list)
        swig_ols = overlap.get_overlap2(
            group_icov.flatten().tolist(), group_mean.flatten().tolist(),
            group_icov_det, star_icovs[i].flatten().tolist(),
            star_means[i].flatten().tolist(), star_icov_dets[i])

        # Using swig module (input passed as numpy arrays)
        swig_np_ols = overlap.get_overlap(
            group_icov, group_mean, group_icov_det, star_icovs[i],
            star_means[i], star_icov_dets[i])

        # # Using swig module (input passed as numpy arrays, output in log)
        # swig_np_lnol = overlap.get_lnoverlap(
        #     group_icov, group_mean, group_icov_det, star_icovs[i],
        #     star_means[i], star_icov_dets[i])

        assert np.isclose(numpy_ols, swig_np_ms_ols[i], rtol=1e-8)
        assert np.isclose(numpy_ols, swig_ols, rtol=1e-8)
        assert np.isclose(numpy_ols, swig_np_ols, rtol=1e-8)
        assert np.isclose(numpy_ols, np.exp(swig_np_ms_lnols[i]), rtol=1e-8)

    print("All implementations return same result to 8 sigfigs")

def timings(group_icov, group_mean, group_icov_det,
              star_icovs, star_means, star_icov_dets, batch_size, noverlaps=10000):
    """
        Executes each function a fixed number of times, timing for how
        long it takes.
    """
    if (noverlaps <= 100000):
        npstart = timer()
        for i in range(noverlaps):
            result = compute_overlap(group_icov, group_mean, group_icov_det,
                                     star_icovs[0], star_means[0],
                                     star_icov_dets[0])
        numpy_time = timer() - npstart
        print("Numpy: " + str(timer() - npstart))
    else:
        numpy_time = None
        print("Numpy: practically infinity seconds")
        print("  -> (approximately 5x 'Swig')")

    swigstart = timer()
    for i in range(noverlaps):
        result =  overlap.get_overlap2(group_icov.flatten().tolist(),
                                       group_mean.flatten().tolist(),
                                       group_icov_det,
                                       star_icovs[0].flatten().tolist(),
                                       star_means[0].flatten().tolist(),
                                       star_icov_dets[0])
    print("Swig: {} s".format(timer() - swigstart))

    swignpstart = timer()
    for i in range(noverlaps):
        result = overlap.get_overlap(group_icov, group_mean, group_icov_det,
                              star_icovs[0], star_means[0], star_icov_dets[0])
    end = timer()
    print("Swigging numpy: {} s".format(end - swignpstart))

    swignpmultistart = timer()
    for i in range(int(noverlaps/batch_size)):
        result = overlap.get_overlaps(
            group_icov, group_mean, group_icov_det,
            star_icovs, star_means, star_icov_dets, batch_size)
    end = timer()

    print("Swigging numpy multi: {} s".format(end - swignpmultistart))
    print("  -> total module calls: {}".format(noverlaps/batch_size))
    print("  -> {} microsec per overlap".\
            format((end - swignpmultistart)/noverlaps*1e6))
    print("  -> {} stars per module call".format(batch_size))

    group_cov = np.linalg.inv(group_icov)
    star_covs = np.linalg.inv(star_icovs)

    newswignpmultistart = timer()
    for i in range(int(noverlaps/batch_size)):
        result = overlap.get_lnoverlaps(group_cov, group_mean,
                              star_covs, star_means, batch_size)
    end = timer()

    print("Swigging numpy multi logged: {} s".format(end - newswignpmultistart))
    print("  -> total module calls: {}".format(noverlaps/batch_size))
    print("  -> {} microsec per overlap".\
            format((end - newswignpmultistart)/noverlaps*1e6))
    print("  -> {} stars per module call".format(batch_size))

    if numpy_time:
        print('Total speed up from basic Numpy is {}'.format(
                numpy_time / (end - newswignpmultistart)
        ))

# ------------- MAIN PROGRAM -----------------------
if __name__ == '__main__':

    print("___ Testing swig module ___")
    #Parsing arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--over', dest='o', default=10000,
                            help='number of overlaps, def: 10000')
    parser.add_argument('-b', '--batch', dest='b', default=10000,
                      help='batch size, must be <= and factor of noverlaps, def: 10000')
    args = parser.parse_args()

    noverlaps = int(args.o)
    batch_size = int(args.b)

    # ensuring batch_size is not greater than noverlaps
    if (batch_size > noverlaps):
      batch_size = noverlaps

    # Readjusting batch_size upwards to next best fitting amount
    batch_size = int(noverlaps/(noverlaps/batch_size))

    #Hard coding some sample data:
    #  1 group inverse covariance matrix and determinant
    #  1 group mean
    #  2 star inverse covariance matrices and determinants
    #  2 star means
    #  n=2:  number of stars in sample data
    group_icov = np.array(
    [[ 0.08169095,-0.08676841, 0.01251394, 0.,         0.,         0.        ],
     [-0.08676841, 0.12519631,-0.03667345, 0.,         0.,         0.        ],
     [ 0.01251394,-0.03667345, 0.02503973, 0.,         0.,         0.        ],
     [ 0.,         0.,         0.,         1.72222567, 0.,         0.        ],
     [ 0.,         0.,         0.,         0.,         1.72222567, 0.        ],
     [ 0.,         0.,         0.,         0.,         0.,         1.72222567]] )
    group_icov_det = 9.06167723629e-05
    group_mean = np.array([ -6.574, 66.56,  23.436, -1.327,-11.427, -6.527])

    star_icovs = np.array(
    [[[ 241.11814038, -20.73085201, -41.76131545, -20.04020342,  39.23379693,
         3.56762733],
      [ -20.73085201, 241.94306462,  65.75059643,  67.93158749,-112.38156699,
        -9.01800703],
      [ -41.76131545,  65.75059643,  93.00901268,  16.28943086,-186.48126616,
       -26.35192182],
      [ -20.04020342,  67.93158749,  16.28943086, 271.35148676,-206.47751678,
         0.59099253],
      [  39.23379693,-112.38156699,-186.48126616,-206.47751678, 533.12434591,
        56.54371174],
      [   3.56762733,  -9.01800703, -26.35192182,   0.59099253,  56.54371174,
         8.7246333 ]],

     [[  3.05924773e+02, -2.14497101e+02,  1.81987150e+02,  2.21167193e+01,
         2.47836028e+01, -1.23364958e+01],
      [ -2.14497101e+02,  3.91116549e+02,  7.84435767e+01,  1.12111433e+00,
         3.67626279e+00,  1.26979547e+01],
      [  1.81987150e+02,  7.84435767e+01,  3.51440781e+02,  3.09116499e-01,
        -1.90331451e+01, -1.68244431e+01],
      [  2.21167193e+01,  1.12111433e+00,  3.09116499e-01,  3.55043182e+01,
         1.69515554e+01, -1.72936911e+01],
      [  2.47836028e+01,  3.67626279e+00, -1.90331451e+01,  1.69515554e+01,
         4.75919822e+01,  1.21941690e+01],
      [ -1.23364958e+01,  1.26979547e+01, -1.68244431e+01, -1.72936911e+01,
         1.21941690e+01,  4.71046181e+01]]]
    )

    star_icov_dets = [ 1315806412.02, 520928339.853 ]

    star_means = np.array(
    [[ -4.76574406, 63.32299927, 39.42994111, -1.31855401,-10.77158563,
       -8.24828843],
     [ 17.58529809,-25.56197368,-20.64041645, -0.86932298, -6.32809279,
       -6.419595  ]] )

    nstars = star_icovs.shape[0]

    print("Testing correctnesss")
    correctness(group_icov, group_mean, group_icov_det, star_icovs,
                                      star_means, star_icov_dets, nstars)
    star_icovs = np.tile(star_icovs[0], (batch_size,1,1))
    star_means = np.tile(star_means[0], (batch_size,1))
    star_icov_dets = np.tile(star_icov_dets[0], batch_size)

    print("Teting timings")
    print("# of overlaps: {}".format(noverlaps))
    timings(
        group_icov, group_mean, group_icov_det, star_icovs,
        star_means, star_icov_dets, batch_size, noverlaps)

    print("___ swig module passsing all unit_tests ___")

