"""
If you make any changes in the expectation sourcecode, compile it with
python3 setup.py build_ext -b .
in the main chronostar folder.
"""

import numpy as np

try:
    from chronostar._expectation import expectation
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

def test_expectation():
    
    data = tabletool.build_data_dict_from_table('example2_XYZUVW.fits')
 
    """
    This is translated from C
    """

    st_mns = data['means']
    st_covs = data['covs']


    ########
    #### COMPONENT DATA ////////////////////////////////////////////////
    #######
    gr_mn = [-4.04221889, -23.35241922, -10.54482267, 
        0.80213607, -8.49588952, 5.56516729]  

    gr_cov = np.zeros((6,6));
    gr_cov[0, 0] = 6.50057168e+02;
    gr_cov[0, 1] = 7.86754530e+01;
    gr_cov[0, 2] = 0.00000000e+00;  
    gr_cov[0, 3] = 1.48517752e+01;
    gr_cov[0, 4] = 3.56504748e+00;
    gr_cov[0, 5] = 0.00000000e+00;
    gr_cov[1, 0] = 7.86754530e+01;
    gr_cov[1, 1] = 4.65226102e+02;
    gr_cov[1, 2] = 0.00000000e+00;
    gr_cov[1, 3] = 3.37334164e+00;
    gr_cov[1, 4] = 2.85696753e+00;
    gr_cov[1, 5] = 0.00000000e+00;
    gr_cov[2, 0] = 0.00000000e+00;
    gr_cov[2, 1] = 0.00000000e+00;
    gr_cov[2, 2] = 7.49802305e+01;
    gr_cov[2, 3] = 0.00000000e+00;
    gr_cov[3, 4] = 0.00000000e+00;
    gr_cov[2, 5] = -2.56820590e+00;
    gr_cov[3, 0] = 1.48517752e+01;
    gr_cov[3, 1] = 3.37334164e+00;
    gr_cov[3, 2] = 0.00000000e+00; 
    gr_cov[3, 3] = 6.07275584e-01;
    gr_cov[3, 4] = 4.05668315e-02;
    gr_cov[3, 5] = 0.00000000e+00;
    gr_cov[4, 0] = 3.56504748e+00;
    gr_cov[4, 1] = 2.85696753e+00;
    gr_cov[4, 2] = 0.00000000e+00;
    gr_cov[4, 3] = 4.05668315e-02;
    gr_cov[4, 4] = 4.01379894e-01;
    gr_cov[4, 5] = 0.00000000e+00;
    gr_cov[5, 0] = 0.00000000e+00;
    gr_cov[5, 1] = 0.00000000e+00;
    gr_cov[5, 2] = -2.56820590e+00;
    gr_cov[5, 3] = 0.00000000e+00;
    gr_cov[5, 4] = 0.00000000e+00;
    gr_cov[5, 5] = 2.28659744e+00;

    nstars = len(st_mns)
    ncomps = 1


    ######################
    cov_now = gr_cov
    mean_now = gr_mn
    star_covs = data['covs']
    star_means = data['means']
    star_count = nstars

    lnols = c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means, 
        star_count)

    ######################


    
    old_memb_probs = np.ones((nstars, ncomps)) / ncomps

    memb_probs = expectation(st_mns, st_covs, gr_mn, gr_cov, 
        old_memb_probs, nstars, ncomps, old_memb_probs)
    



test_expectation()

print("TESTS FINISHED. \n")
