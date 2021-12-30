// clang -L/usr/local/lib test_expectation.c read_input_data_ascii.c ../chronostar/expectation.c -o test_expectation -lgsl -lgslcblas -lm

// ./test_expectation

#include <stdio.h>
#include <stdlib.h>

#include "../chronostar/expectation.h" // Sort out these paths
#include "read_input_data_ascii.h"


void test_get_overlaps() {
    /* ----------
    *  gr_cov        (6*6 npArray)   group's covariance matrix
    *  gr_mn         (1*6 npArray)   group's central estimate (mean)
    *  st_covs       (n*6*6 npArray) array of each star's cov matrix
    *  st_mns:       (n*6 npArray)   array of each star's central estimate
    *  lnols_output: (n npArray)     used to store and return calculated overlaps
    *  nstars:       (int)           number of stars, used for array dimensions
    *
    * Returns
    * -------
    *  (nstars) array of calculated log overlaps of every star with component
    *      thanks to Swig magic the result is stored in `lnols_output` which is
    *      returned as output to the python call as a numpy array
    */
    

    ////////////////////////////////////////////////////////////////////
    //// READ INPUT STELLAR DATA (ASCII) ///////////////////////////////
    ////////////////////////////////////////////////////////////////////
    char* filename = 
        "/Users/marusa/chronostar/fastfit/example2_XYZUVW.dat";    
    
    int nstars = count_number_of_lines(filename);
    const int data_dim = 27;

    double data[nstars*data_dim];
    read_input_gaia_ascii_table(filename, data, data_dim, nstars);

    const int dim = 6; 
    const int st_mn_dim1 = nstars;
    const int st_mn_dim2 = dim;
    const int st_dim1 = nstars;
    const int st_dim2 = dim;
    const int st_dim3 = dim;

    // Means
    double st_mns[nstars*dim];
    generate_means(data, nstars, st_mns, data_dim, dim);

    // Covariance matrices
    double st_covs[nstars*dim*dim];
    generate_covariance_matrices(data, nstars, st_covs, data_dim, dim);


    ////////////////////////////////////////////////////////////////////
    //// COMPONENT DATA ////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    const int gr_mn_dim = 6;
    double gr_mn[gr_mn_dim] = {-4.04221889, -23.35241922, -10.54482267, 
        0.80213607, -8.49588952, 5.56516729};    
    
    const int gr_dim1 = 6;
    const int gr_dim2 = 6;
    
    //~ double gr_cov[gr_dim1][gr_dim2] = {
    //~ double* gr_cov = {
        //~ {6.50057168e+02, 7.86754530e+01, 0.00000000e+00,  
        //~ 1.48517752e+01,3.56504748e+00, 0.00000000e+00}, 
        //~ {7.86754530e+01, 4.65226102e+02, 0.00000000e+00,
        //~ 3.37334164e+00, 2.85696753e+00, 0.00000000e+00},
        //~ {0.00000000e+00, 0.00000000e+00, 7.49802305e+01,
        //~ 0.00000000e+00, 0.00000000e+00, -2.56820590e+00},
        //~ {1.48517752e+01, 3.37334164e+00, 0.00000000e+00, 
        //~ 6.07275584e-01, 4.05668315e-02, 0.00000000e+00},
        //~ {3.56504748e+00, 2.85696753e+00, 0.00000000e+00, 
        //~ 4.05668315e-02, 4.01379894e-01, 0.00000000e+00},
        //~ {0.00000000e+00, 0.00000000e+00, -2.56820590e+00,
        //~ 0.00000000e+00, 0.00000000e+00, 2.28659744e+00}};


    //~ double* gr_cov;
    double gr_cov[gr_dim2*gr_dim2];
    //~ double gr_cov[gr_dim2][gr_dim2];
    gr_cov[0] = 6.50057168e+02;
    gr_cov[1] = 7.86754530e+01;
    gr_cov[2] = 0.00000000e+00;  
    gr_cov[3] = 1.48517752e+01;
    gr_cov[4] = 3.56504748e+00;
    gr_cov[5] = 0.00000000e+00;
    gr_cov[6] = 7.86754530e+01;
    gr_cov[7] = 4.65226102e+02;
    gr_cov[8] = 0.00000000e+00;
    gr_cov[9] = 3.37334164e+00;
    gr_cov[10] = 2.85696753e+00;
    gr_cov[11] = 0.00000000e+00;
    gr_cov[12] = 0.00000000e+00;
    gr_cov[13] = 0.00000000e+00;
    gr_cov[14] = 7.49802305e+01;
    gr_cov[15] = 0.00000000e+00;
    gr_cov[16] = 0.00000000e+00;
    gr_cov[17] = -2.56820590e+00;
    gr_cov[18] = 1.48517752e+01;
    gr_cov[19] = 3.37334164e+00;
    gr_cov[20] = 0.00000000e+00; 
    gr_cov[21] = 6.07275584e-01;
    gr_cov[22] = 4.05668315e-02;
    gr_cov[23] = 0.00000000e+00;
    gr_cov[24] = 3.56504748e+00;
    gr_cov[25] = 2.85696753e+00;
    gr_cov[26] = 0.00000000e+00;
    gr_cov[27] = 4.05668315e-02;
    gr_cov[28] = 4.01379894e-01;
    gr_cov[29] = 0.00000000e+00;
    gr_cov[30] = 0.00000000e+00;
    gr_cov[31] = 0.00000000e+00;
    gr_cov[32] = -2.56820590e+00;
    gr_cov[33] = 0.00000000e+00;
    gr_cov[34] = 0.00000000e+00;
    gr_cov[35] = 2.28659744e+00;

    
    // Output
    int n = nstars;
    double lnols_output[n];

    // Compute overlaps
    get_lnoverlaps(gr_cov, gr_dim1, gr_dim2, gr_mn, gr_mn_dim, st_covs, 
        st_dim1, st_dim2, st_dim3, st_mns, st_mn_dim1, st_mn_dim2, 
        lnols_output, n);
    
    for (int i=0; i<n; i++) {
        printf("%d %f\n", i, lnols_output[i]);
    }
    
    
}


void test_expectation() {
    ////////////////////////////////////////////////////////////////////
    //// READ INPUT STELLAR DATA (ASCII) ///////////////////////////////
    ////////////////////////////////////////////////////////////////////
    char* filename = 
        "/Users/marusa/chronostar/fastfit/example2_XYZUVW.dat";    
    
    int nstars = count_number_of_lines(filename);
    const int data_dim = 27;

    double data[nstars*data_dim];
    read_input_gaia_ascii_table(filename, data, data_dim, nstars);

    const int dim = 6; 
    const int st_mn_dim1 = nstars;
    const int st_mn_dim2 = dim;
    const int st_dim1 = nstars;
    const int st_dim2 = dim;
    const int st_dim3 = dim;

    // Means
    double st_mns[nstars*dim];
    generate_means(data, nstars, st_mns, data_dim, dim);
    
    //~ printf("print st_mns");
    //~ for (int i=0; i<nstars; i++) {
        //~ for (int j=0; j<6; j++) {
            //~ printf("%f ", st_mns[i*6+j]);
        //~ }
        //~ printf("\n");
    //~ }
    

    // Covariance matrices
    double st_covs[nstars*dim*dim];
    generate_covariance_matrices(data, nstars, st_covs, data_dim, dim);


    ////////////////////////////////////////////////////////////////////
    //// COMPONENT DATA ////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    const int gr_mn_dim = 6;
    double gr_mn[gr_mn_dim] = {-4.04221889, -23.35241922, -10.54482267, 
        0.80213607, -8.49588952, 5.56516729};    
    
    const int gr_dim1 = 6;
    const int gr_dim2 = 6;
    
    //~ double gr_cov[gr_dim1][gr_dim2] = {
    //~ double* gr_cov = {
        //~ {6.50057168e+02, 7.86754530e+01, 0.00000000e+00,  
        //~ 1.48517752e+01,3.56504748e+00, 0.00000000e+00}, 
        //~ {7.86754530e+01, 4.65226102e+02, 0.00000000e+00,
        //~ 3.37334164e+00, 2.85696753e+00, 0.00000000e+00},
        //~ {0.00000000e+00, 0.00000000e+00, 7.49802305e+01,
        //~ 0.00000000e+00, 0.00000000e+00, -2.56820590e+00},
        //~ {1.48517752e+01, 3.37334164e+00, 0.00000000e+00, 
        //~ 6.07275584e-01, 4.05668315e-02, 0.00000000e+00},
        //~ {3.56504748e+00, 2.85696753e+00, 0.00000000e+00, 
        //~ 4.05668315e-02, 4.01379894e-01, 0.00000000e+00},
        //~ {0.00000000e+00, 0.00000000e+00, -2.56820590e+00,
        //~ 0.00000000e+00, 0.00000000e+00, 2.28659744e+00}};


    //~ double* gr_cov;
    double gr_cov[gr_dim2*gr_dim2];
    //~ double gr_cov[gr_dim2][gr_dim2];
    gr_cov[0] = 6.50057168e+02;
    gr_cov[1] = 7.86754530e+01;
    gr_cov[2] = 0.00000000e+00;  
    gr_cov[3] = 1.48517752e+01;
    gr_cov[4] = 3.56504748e+00;
    gr_cov[5] = 0.00000000e+00;
    gr_cov[6] = 7.86754530e+01;
    gr_cov[7] = 4.65226102e+02;
    gr_cov[8] = 0.00000000e+00;
    gr_cov[9] = 3.37334164e+00;
    gr_cov[10] = 2.85696753e+00;
    gr_cov[11] = 0.00000000e+00;
    gr_cov[12] = 0.00000000e+00;
    gr_cov[13] = 0.00000000e+00;
    gr_cov[14] = 7.49802305e+01;
    gr_cov[15] = 0.00000000e+00;
    gr_cov[16] = 0.00000000e+00;
    gr_cov[17] = -2.56820590e+00;
    gr_cov[18] = 1.48517752e+01;
    gr_cov[19] = 3.37334164e+00;
    gr_cov[20] = 0.00000000e+00; 
    gr_cov[21] = 6.07275584e-01;
    gr_cov[22] = 4.05668315e-02;
    gr_cov[23] = 0.00000000e+00;
    gr_cov[24] = 3.56504748e+00;
    gr_cov[25] = 2.85696753e+00;
    gr_cov[26] = 0.00000000e+00;
    gr_cov[27] = 4.05668315e-02;
    gr_cov[28] = 4.01379894e-01;
    gr_cov[29] = 0.00000000e+00;
    gr_cov[30] = 0.00000000e+00;
    gr_cov[31] = 0.00000000e+00;
    gr_cov[32] = -2.56820590e+00;
    gr_cov[33] = 0.00000000e+00;
    gr_cov[34] = 0.00000000e+00;
    gr_cov[35] = 2.28659744e+00;

    int ncomps = 1;
    
    double memb_probs[nstars*ncomps]; // result goes here
    double old_memb_probs[nstars*ncomps];
    
    for (int i=0; i<nstars; i++) {
        for (int j=0; j<ncomps; j++) {
            old_memb_probs[i*ncomps+j] = 1.0 / ncomps;
        }
    }

    expectation(st_mns, st_covs, nstars, gr_mn, gr_cov, ncomps, 
        memb_probs, old_memb_probs);
    
    //~ for (int i=0; i<nstars; i++) {
        //~ for (int j=0; j<ncomps; j++) {
            //~ printf("%f ", memb_probs[i*ncomps+j]);
        //~ }
        //~ printf("\n");
    //~ }   
    
}


int main() {
    //~ test_get_overlaps();
    
    test_expectation();
    
    printf("TESTS FINISHED. \n");

    return 0;
}
