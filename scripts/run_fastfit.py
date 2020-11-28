#! /usr/bin/env python
"""
A helper script that performs a brute force Chronostar fit to
some pre-prepared data.

Accepts as a command line argument a path to a parameter file
TODO: Update README.md with description for NaiveFit parameters
(see README.md for how to structure the parameter file and
various available parameters).
"""

from mpi4py import MPI
from multiprocessing import cpu_count

import os.path
import sys
sys.path.insert(0, '..')
from chronostar.naivefit import NaiveFit
from chronostar import expectmax



# Parallelism
def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START', 'FINISH')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes: e.g. np -4
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

num_workers = size - 1 # Maybe increase this to 'size' as rank=0 could also do some work.




# READ parameter file
if len(sys.argv) != 2:
    if rank==0:
        raise UserWarning('Incorrect usage. Path to parameter file is required'
                          ' as a single command line argument. e.g.\n'
                          '   > python new_run_chronostar.py path/to/parsfile.par')

fit_par_file = sys.argv[1]

if not os.path.isfile(fit_par_file):
    raise UserWarning('Provided file does not exist')

# ALL ranks need to read-in data and params
print('Init naivefit for rank', rank)
naivefit = NaiveFit(fit_pars=fit_par_file)


# Master rank runs the main code
if rank == 0:
    naivefit.run_fit(comm=comm, tags=tags)



    

    ### CLOSE WORKERS after finished ######
    # This enables run_fastfit.py to exit.
    # Workers need to be closed otherwise they will continue waiting
    # to receive tasks to do.
    print('Start closing workers...')
    closed_workers = 0
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        
        # Tell workers to stop waiting for the next task and exit.
        comm.send(None, dest=source, tag=tags.EXIT)

        if tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1

    print("Master finishing")





# Workers maximise the components
else:
    """
    Workers do the MAXIMISATION step here.
    This maximises one component only.
    """
    #~ name = MPI.Get_processor_name()
    #~ print("I am a worker with rank %d on %s." % (rank, name))
    
    data_dict = naivefit.data_dict
    
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.START:
            #~ print('Worker %d START'%rank)
            #~ print(rank, task)
            args = task#[1]
            i = args['i']
            memb_probs = args['memb_probs']
            idir = args['idir']

            print('Worker %d maximising component %g'%(rank, i), idir)
            

            #~ best_comp, chain, lnprob, final_pos, i = expectmax.maximise_one_comp(data_dict,
                #~ memb_probs, i, idir, **args) # Rather pass **args than **naivefit.fit_pars because **args are updated with current info.

            best_comp, chain, lnprob, final_pos, iresult = expectmax.maximise_one_comp(data_dict,
                **args) # Rather pass **args than **naivefit.fit_pars because **args are updated with current info.
            
            
            if i!=iresult:
                print('i!=iresult')

            result = dict()
            result['i'] = i
            result['best_comp'] = best_comp
            result['chain'] = chain
            result['lnprob'] = lnprob
            result['final_pos'] = final_pos

            
            # Send result to master.
            comm.send(result, dest=0, tag=tags.DONE)
            
            
        elif tag == tags.FINISH:
            comm.send(None, dest=0, tag=tags.FINISH)
            
        elif tag == tags.EXIT:
            break

    comm.send(result, dest=0, tag=tags.EXIT)

