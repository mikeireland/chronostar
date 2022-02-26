"""
If you make any changes in the expectation sourcecode, compile it with
python3 setup.py build_ext -b .
in the main chronostar folder.
"""

import numpy as np

try:
    from chronostar._expectation import expectation
    from chronostar._expectation import expectation_iterative_component_amplitudes
    from chronostar._expectation import print_bg_lnols
except ImportError:
    print("C IMPLEMENTATION OF expectation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

# For testing purposes. Remove later.
try:
    from chronostar._overlap import get_lnoverlaps as c_get_lnoverlaps
except ImportError:
    print("C IMPLEMENTATION OF GET_OVERLAP NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

# Read in stellar data
from chronostar import tabletool
from chronostar.run_em_files_python import expectation_marusa as expectationP

from chronostar import expectmax

import time

####################################################################
#### DATA ##########################################################
####################################################################
import pickle
#~ with open('data_for_testing/input_data_to_expectation.pkl', 'rb') as f:
    #~ d = pickle.load(f)
with open('data_for_testing/input_data_to_expectation_segmentation_fault.pkl', 'rb') as f:
    d = pickle.load(f)
    #~ # old_memb_probs: probabilities do not sum up to 1!!

data_dict = d[0]
comps = d[1]
old_memb_probs = d[2]
inc_posterior = d[3]
use_box_background = d[4]


# Components should be provided at time NOW.
gr_mns = [c.get_mean_now() for c in comps]
gr_covs = [c.get_covmatrix_now() for c in comps]
comps = [[m, co] for m, co in zip(gr_mns, gr_covs)]
bg_lnols = data_dict['bg_lnols']

#~ comps = comps[0]
use_box_background = False # Because it's false in C



####################################################################
#### Python - not iterative ########################################
####################################################################
# Components are Component objects. Traceforward is done somewhere 
# within this function.
time_start = time.time()
memb_probs_newP = expectationP.expectation(data_dict, 
    comps, old_memb_probs, inc_posterior=inc_posterior, 
    use_box_background=use_box_background)
duration_P = time.time()-time_start
print('Duration python:', duration_P)


memb_probs_newP_NOTiterative = memb_probs_newP
####################################################################
#### C - not iterative #############################################
####################################################################
st_mns = data_dict['means']
st_covs = data_dict['covs']

#~ gr_mns = [tp.trace_epicyclic_orbit(xyzuvw_start, times=None)]
#~ gr_cov = [tp.trace_epicyclic_covmatrix(cov, loc, age=None)]
nstars = len(st_mns)
ncomps = len(gr_mns)+1 # for bg

time_start = time.time()
memb_probs_newC = expectation(st_mns, st_covs, gr_mns, gr_covs, 
    bg_lnols, old_memb_probs, nstars*ncomps)
memb_probs_newC = memb_probs_newC.reshape(nstars, ncomps)
duration_C = time.time()-time_start
print('Duration C:', duration_C)

print('Duration_python / Duration_C', duration_P/duration_C)

####################################################################
#### COMPARE RESULTS ###############################################
####################################################################
diff = np.abs(memb_probs_newC-memb_probs_newP)
print('DIFF')
print(diff)
mx = np.max(diff)
print('max diff %e'%mx)

if np.any(diff>1e-12):
    print('NO AGREEMENT!!')
else:
    print('PERFECT AGREEMENT')


print('')
print('')
print('')
####################################################################
#### Python - iterative ############################################
####################################################################
# Components are Component objects. Traceforward is done somewhere 
# within this function.
time_start = time.time()
memb_probs_newP = expectmax.expectation(data_dict, d[1], 
    old_memb_probs=old_memb_probs, inc_posterior=inc_posterior, 
    amp_prior=None, use_box_background=use_box_background)
duration_P = time.time()-time_start
print('Duration python:', duration_P)

memb_probs_newP_iterative = memb_probs_newP

####################################################################
#### C - iterative #################################################
####################################################################
st_mns = data_dict['means']
st_covs = data_dict['covs']

#~ gr_mns = [tp.trace_epicyclic_orbit(xyzuvw_start, times=None)]
#~ gr_cov = [tp.trace_epicyclic_covmatrix(cov, loc, age=None)]
nstars = len(st_mns)
ncomps = len(gr_mns)+1 # for bg

time_start = time.time()
memb_probs_newC = expectation_iterative_component_amplitudes(st_mns, 
    st_covs, gr_mns, gr_covs, bg_lnols, old_memb_probs, nstars*ncomps)
memb_probs_newC = memb_probs_newC.reshape(nstars, ncomps)
duration_C = time.time()-time_start
print('Duration C:', duration_C)

print('Duration_python / Duration_C', duration_P/duration_C)

####################################################################
#### COMPARE RESULTS ###############################################
####################################################################
diff = np.abs(memb_probs_newC-memb_probs_newP)
print('DIFF')
print(diff)
mx = np.max(diff)
print('max diff %e'%mx)

if np.any(diff>1e-12):
    print('NO AGREEMENT!!')
else:
    print('PERFECT AGREEMENT')

#### Check the difference between iterative and NONiterative ###########
diff = np.abs(memb_probs_newP_iterative - memb_probs_newP_NOTiterative)
print('diff (iterative - NONiterative')
print(diff)
mx = np.max(diff)
print('max diff %e'%mx)

print("TESTS FINISHED. \n")

print('python bg_lnols')
print(bg_lnols)
print(type(bg_lnols))

print('bg_lnols C')
print_bg_lnols(bg_lnols)
