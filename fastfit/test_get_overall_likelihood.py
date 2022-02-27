import numpy as np

# C modules
try:
    from chronostar._overall_likelihood import get_overall_lnlikelihood_for_fixed_memb_probs
except ImportError:
    print("C IMPLEMENTATION OF expectation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?


# Read in stellar data
from chronostar import tabletool
from chronostar.run_em_files_python import expectation_marusa as expectationP

import time

####################################################################
#### DATA ##########################################################
####################################################################
import pickle
with open('data_for_testing/input_data_to_get_overall_lnlikelihood_for_fixed_memb_probs.pkl', 'rb') as f:
    d = pickle.load(f)

st_mns = d[0]
st_covs = d[1]
gr_mns = d[2]
gr_covs = d[3]
bg_lnols = d[4]
memb_probs_new = d[5]
data_dict = d[6]
comps_new = d[7]
memb_probs_new = d[8] # same as d[5]
inc_posterior = d[9]
use_box_background = d[10]


####################################################################
#### Python ########################################################
####################################################################
#~ comps_new_list = [[comp.get_mean(), comp.get_covmatrix()] for comp in comps_new]
comps_new_list = [[comp.get_mean_now(), comp.get_covmatrix_now()] for comp in comps_new]
time_start = time.time()

overall_lnlikeP = expectationP.get_overall_lnlikelihood_for_fixed_memb_probs(
    data_dict, comps_new_list, memb_probs=memb_probs_new, 
    inc_posterior=inc_posterior, use_box_background=use_box_background)

#~ overall_lnlikeP = expectationP.get_overall_lnlikelihood(
    #~ data_dict, comps_new_list, old_memb_probs=memb_probs_new, 
    #~ inc_posterior=inc_posterior, use_box_background=use_box_background)

print('overall_lnlikeP', overall_lnlikeP)

duration_P = time.time()-time_start
print('Duration python:', duration_P)


####################################################################
#### C #############################################################
####################################################################
time_start = time.time()

overall_lnlikeC = get_overall_lnlikelihood_for_fixed_memb_probs(
    st_mns, st_covs, gr_mns, gr_covs, bg_lnols, memb_probs_new)
#~ overall_lnlikeC = overall_lnlikeC[0] # TODO
print('overall_lnlikeC', overall_lnlikeC)

duration_C = time.time()-time_start
print('Duration C:', duration_C)


####################################################################
#### COMPARE TIME ##################################################
####################################################################
print('Duration_python / Duration_C', duration_P/duration_C)

####################################################################
#### COMPARE RESULTS ###############################################
####################################################################
diff = np.abs(overall_lnlikeC-overall_lnlikeP)
print('DIFF', diff)

print("TESTS FINISHED. \n")
