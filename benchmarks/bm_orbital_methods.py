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
from chronostar.component import SphereComponent
from chronostar.traceorbit import trace_cartesian_orbit, trace_epicyclic_orbit

def timings(comp, noverlaps=10000):
    """
        Executes each function a fixed number of times, timing for how
        long it takes.
    """
    comp.trace_orbit_func = trace_cartesian_orbit

    galpystart = timer()
    for i in range(noverlaps):
        comp.get_currentday_projection()
        comp.update_attribute({'age':comp.get_age()+1.e-10})

    galpy_time = timer() - galpystart
    print("GALPY")
    print("Time: %.5f s"%galpy_time)
    print("Time per projection: %.5f micros"%(galpy_time/nprojections * 1e6))

    comp.trace_orbit_func = trace_epicyclic_orbit
    epicycstart = timer()
    for i in range(noverlaps):
        comp.get_currentday_projection()
        comp.update_attribute({'age':comp.get_age()+1.e-10})
    epicyctime = timer() - epicycstart
    print("EPICYCLIC")
    print("Time: " + str(epicyctime))
    print("Time per projection: %.3f micros"%(epicyctime/nprojections * 1e6))


# ------------- MAIN PROGRAM -----------------------
if __name__ == '__main__':

    print("___ Testing swig module ___")
    #Parsing arguments

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--over', dest='o', default=10000,
                            help='number of overlaps, def: 10000')
    # parser.add_argument('-b', '--batch', dest='b', default=10000,
    #                   help='batch size, must be <= and factor of noverlaps, def: 10000')
    args = parser.parse_args()

    nprojections = int(args.o)

    pars = np.hstack((np.zeros(6),[10.,5,100]))
    my_comp = SphereComponent(pars)

    print("Testing timings")
    print("# of projections: {}".format(nprojections))
    timings(my_comp, nprojections)


