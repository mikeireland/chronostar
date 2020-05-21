"""
Run the maximisation step for all components at once.
Use one process for each component (np is external parameter).
"""

import subprocess # to call external scripts
import numpy as np
import os
import sys
import itertools
from mpi4py import MPI

import sys
sys.path.insert(0, '..')
from chronostar import readparam

comm = MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

if rank==0:
    # Read fit parameters from the file    
    filename_fit_pars = sys.argv[1]
    fit_pars = readparam.readParam(filename_fit_pars)
    
    filenames_pars = np.load(sys.argv[2])
    
    # Scatter data
    filename_params = np.array_split(filenames_pars, size)

else:
    filename_params = None
    filename_fit_pars = None

filename_fit_pars = comm.bcast(filename_fit_pars, root=0)

# SCATTER DATA
filename_params = comm.scatter(filename_params, root=0)
if len(filename_params)==1:
    filename_params=filename_params[0]
else:
    print('run_maximisation_all_comps: LEN FILENAME_PARAMS>1; increase number of processes!')

# RUN MAXIMISATION
bashCommand = 'python run_maximisation_1_comp.py %s %s'%(filename_fit_pars, filename_params)
#~ bashCommand = 'python run_maximisation_1_comp_scipy_optimise.py %s %s'%(filename_fit_pars, filename_params)
print(bashCommand)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#~ output, error = process.communicate()
#~ _, _ = process.communicate()
process_output, _ = process.communicate()
print('process_output', process_output)

# Gather results: this makes the code wait until all ranks are finished and then exit.
result = comm.gather([True], root=0)
if rank == 0:
    result = list(itertools.chain.from_iterable(result))


    
