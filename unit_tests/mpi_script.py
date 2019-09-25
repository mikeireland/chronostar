#from schwimmbad import MPIPool, MultiPool
import emcee
from emcee.utils import MPIPool
import sys
if sys.version[0] == '2':
    from cPickle import PicklingError
else:
    from _pickle import PicklingError
import numpy as np

class class_with_method(object):
    '''A class with method either defined externally or internally'''
    def __init__(self, give_me_data, lnprobfunc=None):
        _data = give_me_data
        if lnprobfunc is None:
            self.ln_probability = self.internal_ln_probability
        else:
            self.ln_probability = lnprobfunc
        
    def internal_ln_probability(self, p, mean, var):
        x = p[0]
        return -0.5 * (x-mean)**2 / var

def global_ln_probability(p, mean, var):
    '''A globally defined logprob function'''
    x = p[0]
    return -0.5 * (x-mean)**2 / var

def test_mpi(pool, the_func):
    sampler = emcee.EnsembleSampler(n_walkers, dim=1,
                                    lnpostfn=the_func,
                                    args=(5, 1.2),
                                    pool=pool) # the important line

    pos,_,_ = sampler.run_mcmc(p0, 500)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

if __name__ == '__main__':
    print('mpi_script.py using python {}.{}'.format(
        *sys.version.split('.')[:2]
        ))
    n_walkers = 16
    p0 = np.random.uniform(0, 10, (n_walkers, 1))
    my_object_int = class_with_method('dummy_data')
    my_object_ext = class_with_method('dummy_data',
                                      lnprobfunc=global_ln_probability)

    # In the case of non-concurrency, everything works
    for lnprob in [global_ln_probability, my_object_int.ln_probability,
                   my_object_ext.ln_probability]:
        test_mpi(pool=None, the_func=lnprob)
    print('Everything works if MPI is not used')

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        # Try with global function
        test_mpi(pool=pool, the_func=global_ln_probability)

        # Try with externally defined method
        test_mpi(pool=pool, the_func=my_object_ext.ln_probability)
        print('Pickling a class with externally defined lnprob is fine')

        # Try with internally defined method
        try:
            test_mpi(pool=pool, the_func=my_object_int.ln_probability)
            print('Pickling class with internally defined lnprob is fine on python 3')
        except PicklingError:
            assert sys.version[0] == '2'
            print('Pickling class with internally defined lnprob breaks on python 2')
