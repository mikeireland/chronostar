'''
Parallelism tutorial copied from
https://emcee.readthedocs.io/en/latest/tutorials/parallel/
credit: Dan Foreman-Mackey
'''
import os
from multiprocessing import cpu_count
from multiprocessing import Pool
import emcee
import time
import numpy as np
import sys

os.environ["OMP_NUM_THREADS"] = "1"
print(emcee.__version__)
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

# Handle some simple input
if len(sys.argv) > 1:
    nthreads = min(int(sys.argv[1]),ncpu)
else:
    nthreads = ncpu
print('Using {} threads'.format(nthreads))
if len(sys.argv) > 2:
    scale = float(sys.argv[2])
else:
    scale = 1.0
print('Scale factor of {}'.format(scale))

def log_prob(theta):
    t = time.time() + scale*np.random.uniform(0.005, 0.008)
    while time.time() < t:
        pass
    return -0.5*np.sum(theta**2)

np.random.seed(42)
initial = np.random.randn(32, 5)
nwalkers, ndim = initial.shape
nsteps = 100

# with Pool() as pool: (not consistent with python2)
# pool = Pool(nthreads)
pool = None
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool, threads=nthreads)
start = time.time()
sampler.run_mcmc(initial, nsteps)
end = time.time()
multi_time = end - start
print("Multiprocessing took {0:.1f} seconds".format(multi_time))
# pool.close()


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
start = time.time()
sampler.run_mcmc(initial, nsteps)
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))



print("{0:.1f} times faster than serial".format(serial_time / multi_time))

