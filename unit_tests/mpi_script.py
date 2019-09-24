#from schwimmbad import MPIPool, MultiPool
import emcee
from emcee.utils import MPIPool
import sys
if sys.version[0] == '2':
    from cPickle import PicklingError
else:
    from _pickle import PicklingError
import numpy as np

class not_really_a_class(object):
    def __init__(self, give_me_data):
        _data = give_me_data
        
    def ln_probability(self, p, mean, var):
        x = p[0]
        return -0.5 * (x-mean)**2 / var

def ln_probability(p, mean, var):
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
    my_object = not_really_a_class(6)
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        test_mpi(pool=pool, the_func=ln_probability)
        try:
            test_mpi(pool=pool, the_func=my_object.ln_probability)
            print('Pickling class is fine on python 3')
        except PicklingError:
            assert sys.version[0] == '2'
            print('Pickling class breaks on python 2')

