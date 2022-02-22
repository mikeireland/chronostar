try:
    from chronostar._likelihood import lnprob_func_gradient_descent
except ImportError:
    print("C IMPLEMENTATION OF lnprob_func_gradient_descent NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?

from chronostar import likelihood
from chronostar import traceorbit

import numpy as np
import time
import scipy.optimize

import pickle
with open('data_for_testing/lnprob_func_gradient_descent_10.pkl', 'rb') as h:
    d = pickle.load(h)

data=d[0]
memb_probs=d[1]
init_pos=d[2]

# data['means']: [nstars, 6]
# data['covs']: [nstars, 6, 6]
# data['bg_lnols']: [nstars, 1]
# memb_probs: [nstars, 1]

# Construct an array that contains all the data needed for lnprob
# THIS IS SOMETHING THAT NEEDS TO BE DONE ONCE AT THE BEGINNING, + add memb_probs at the end each time
a = []
memb_threshold=1e-5
nearby_star_mask = np.where(memb_probs > memb_threshold)
#~ for i in range(len(memb_probs)):
for i in nearby_star_mask[0]:
    tmp = np.hstack((data['means'][i], data['covs'][i].flatten(), memb_probs[i]))
    a.append(tmp)
a=np.array(a)

start = time.time()
resultC = lnprob_func_gradient_descent(init_pos, a)
durationC = time.time()-start
print('result C', resultC)
#~ print('Duration C:', durationC)

start = time.time()
trace_orbit_func=traceorbit.trace_epicyclic_orbit
optimisation_method='Nelder-Mead'
from chronostar.component import SphereComponent
Component = SphereComponent
#~ args = [data, memb_probs, trace_orbit_func, optimisation_method] # likelihood2
#~ resultP = likelihood.lnprob_func_gradient_descent(init_pos, args, 
    #~ memb_probs=memb_probs, 
    #~ trace_orbit_func=traceorbit.trace_epicyclic_orbit)
argsP = [data, memb_probs, trace_orbit_func, Component]
resultP = likelihood.lnprob_func_gradient_descent(init_pos, argsP)
durationP = time.time()-start
print('result P', resultP)
#~ print('Duration P:', durationP)

print('likelihood evaluation: DurationP / DurationC', durationP/durationC)

# I don't really need to test minimisation but just the function evaluation.
convergence_tol=1
optimisation_method = 'Nelder-Mead'

# C
start = time.time()
result = scipy.optimize.minimize(lnprob_func_gradient_descent, init_pos, 
    args=a, tol=convergence_tol, method=optimisation_method)
durationC = time.time()-start
print('Result C')
print(result.x)


# Python
start = time.time()
result2 = scipy.optimize.minimize(
    likelihood.lnprob_func_gradient_descent, init_pos, 
    args=argsP, tol=convergence_tol, method=optimisation_method)
durationP = time.time()-start
print('Result P')
print(result2.x)

print('MINIMIZE: DurationP / DurationC', durationP/durationC)
