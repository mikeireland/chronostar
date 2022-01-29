"""
If you make any changes in the sourcecode, compile it with
python3 setup.py build_ext -b .
in the main chronostar folder.
"""

import numpy as np

try:
    from chronostar._temporal_propagation import trace_epicyclic_orbit, trace_epicyclic_covmatrix
except ImportError:
    print("C IMPLEMENTATION OF temporal_propagation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

from chronostar import traceorbit
from chronostar import transform

import time


def test_traceorbit():
    age=18.80975968983912
    
    # mean0 (time=0)
    mean = np.array([54.24222826, 125.66013072, -73.70025391, 
        -3.7937685, -6.78350686, -0.22042236])

    # mean_now_python
    start = time.time()
    mean_now_P = traceorbit.trace_epicyclic_orbit(mean, times=age)
    duration_P = time.time()-start
    
    
    
    # C implementation
    start = time.time()
    mean_now_C = trace_epicyclic_orbit(mean, age, len(mean))
    duration_C = time.time()-start
    
    diff = mean_now_C - mean_now_P
    print('diff traceorbit')
    print(diff)

    print('Duration P', duration_P)
    print('Duration C', duration_C)
    print('Duration_P / Duration_C', duration_P / duration_C)


def test_trace_covmatrix():
    age=18.80975968983912

    # mean0 (time=0)
    mean = np.array([-4.042219, -23.352419, -10.544823, 0.802136, -8.495890, 5.565167])


    # It works for this one
    cov = np.array(
        [[650.057168, 78.675453, 0.000000, 14.851775, 3.565047, 0.000000],
        [78.675453, 465.226102, 0.000000, 3.373342, 2.856968, 0.000000],
        [0.000000, 0.000000, 74.980231, 0.000000, 0.000000, -2.568206],
        [14.851775, 3.373342, 0.000000, 0.607276, 0.040567, 0.000000],
        [3.565047, 2.856968, 0.000000, 0.040567, 0.401380, 0.000000],
        [0.000000, 0.000000, -2.568206, 0.000000, 0.000000, 2.286597]])

    #~ cov = np.array(
    #~ [[8.76440841, 0., 0., 0., 0., 0.],
    #~ [0., 8.76440841, 0., 0., 0., 0.],
    #~ [0., 0., 8.76440841, 0., 0., 0.],
    #~ [0., 0., 0., 0.14979915, 0., 0],
    #~ [0., 0., 0., 0., 0.14979915, 0.],
    #~ [0., 0., 0., 0., 0., 0.14979915]])



    covt_dim = 6
    h = 1e-3
    
    start = time.time()
    covt_P = transform.transform_covmatrix(cov, 
        trans_func=traceorbit.trace_epicyclic_orbit, loc=mean, 
        args=(age,),)
    duration_P = time.time()-start
    #~ print(covt_P)
    
    start = time.time()
    covt_C = trace_epicyclic_covmatrix(cov, mean, age, h, covt_dim*covt_dim)
    #~ print('covt_C')
    #~ print(covt_C)
    covt_C = covt_C.reshape(covt_dim, covt_dim)
    duration_C = time.time()-start
    
    diff = covt_C - covt_P
    print('diff trace_covmatrix')
    print(diff)
    mx = np.max(diff)
    if np.any(np.abs(diff)>1e-6):
        print('DISAGREEMENT')
    else:
        print('GOOD AGREEMENT')
    print('Largest difference', mx)
    
    print('Duration P', duration_P)
    print('Duration C', duration_C)
    print('Duration_P / Duration_C', duration_P / duration_C)

test_traceorbit()
test_trace_covmatrix()
