try:
    from chronostar._likelihoodc import lnprob_func_gradient_descent
except ImportError:
    print("C IMPLEMENTATION OF lnprob_func_gradient_descent NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?

import numpy as np
import scipy.optimize

import pickle
with open('lnprob_func_gradient_descent_10.pkl', 'rb') as h:
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
for i in range(len(memb_probs)):
    tmp = np.hstack((data['means'][i], data['covs'][i].flatten(), data['bg_lnols'][i], memb_probs[i]))
    a.append(tmp)
a=np.array(a)


test = lnprob_func_gradient_descent(init_pos, a)


#~ result = scipy.optimize.minimize(lnprob_test, init_pos, args=args, tol=1, method='Nelder-Mead')

#~ result = scipy.optimize.minimize(
    #~ lnprob_func_gradient_descent_sphere_component, 
    #~ init_pos, args=args, 
    #~ tol=1, method='Nelder-Mead')

