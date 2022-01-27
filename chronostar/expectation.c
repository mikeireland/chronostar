#include <stdio.h>
#include <stddef.h>
#include <Python.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <string.h>
#include <math.h>


void get_lnoverlaps(
    double* gr_cov, int gr_dim1, int gr_dim2, 
    double* gr_mn, int gr_mn_dim, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3, 
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* lnols_output, int n) {
    
    /* Function: get_lnoverlaps
    * ------------------------
    *   Calculates the log overlap (convolution) with a set of 6D Gaussians with
    *   a single 6D Gaussian, each in turn.
    *
    *   In Chronostar, this is used to see how well the kinematic properties
    *   of stars overlap with a proposed Gaussian distribution.
    *
    * Paramaters
    *  name      type            description
    * ----------
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
    *
    * Notes
    * -----
    *   For each star calculates:
    *     log(1/sqrt( (2*PI)^6*det(C) ) * exp( -0.5*(b-a)^T*(C)^-1*(b-a) )
    *   where
    *     C = st_cov + gr_cov
    *   and
    *     a = gr_mn, b = st_mn
    *   Expanding and simplifying this becomes:
    *     -0.5[ 6*ln(2*PI) + ln(|C|) + (b-a)^T*(C^-1)*(b-a) ]
    *
    *   Stark improvement on previous implementations. Doesn't require input as
    *   inverse covariance matrices. Never performs a matrix inversion.
    */
    // ALLOCATE MEMORY
    int star_count = 0;
    int MAT_DIM = gr_dim1; //Typically set to 6
    int i, j, signum;
    double d_temp, result, ln_det_BpA;
    gsl_permutation *p1;

    gsl_matrix *BpA      = gsl_matrix_alloc(MAT_DIM, MAT_DIM); //will hold (B+A)
    gsl_vector *bma      = gsl_vector_alloc(MAT_DIM);          //will hold b - a
    gsl_vector *v_temp   = gsl_vector_alloc(MAT_DIM);

    p1 = gsl_permutation_alloc(BpA->size1);

    // Go through each star, calculating and storing overlap
    for (star_count=0; star_count<n; star_count++) {
        // INITIALISE STAR MATRIX
        for (i=0; i<MAT_DIM; i++)
            for (j=0; j<MAT_DIM; j++)
            //performing st_cov+gr_cov as part of the initialisation
            gsl_matrix_set(BpA,i,j,
                st_covs[star_count*MAT_DIM*MAT_DIM+i*MAT_DIM+j] +
                gr_cov[i*MAT_DIM+j]);

        // INITIALISE CENTRAL ESTIMATES
        // performing st_mn - gr_mn as part of the initialisation
        for (i=0; i<MAT_DIM; i++) {
            gsl_vector_set(bma, i, 
                st_mns[star_count*MAT_DIM + i] - gr_mn[i]);
        }

        // CALCULATE OVERLAPS
        // Performed in 4 stages
        // Calc and sum up the inner terms:
        // 1) 6 ln(2pi)
        // 2) ln(|C|)
        // 3) (b-a)^T(C^-1)(b-a)
        // Then apply -0.5 coefficient

        // 1) Calc 6 ln(2pi)
        result = 6*log(2*M_PI);

        // 2) Get log determiant of C
        gsl_linalg_LU_decomp(BpA, p1, &signum);
        ln_det_BpA = log(fabs(gsl_linalg_LU_det(BpA, signum)));
        result += ln_det_BpA;

        // 3) Calc (b-a)^T(C^-1)(b-a)
        gsl_vector_set_zero(v_temp);
        gsl_linalg_LU_solve(BpA, p1, bma, v_temp); /* v_temp holds (B+A)^-1 (b-a) *
                                                    * utilises `p1` as calculated *
                                                    * above                       */
        gsl_blas_ddot(v_temp, bma, &d_temp); //d_temp holds (b-a)^T (B+A)-1 (b-a)
        result += d_temp;

        // 4) Apply coefficient
        result *= -0.5;

        // STORE RESULT 'lnols_output'
        lnols_output[star_count] = result;
        //~ printf("%d lnols_output %f \n", star_count, result);
    }

    // DEALLOCATE THE MEMORY
    gsl_matrix_free(BpA);
    gsl_vector_free(bma);
    gsl_vector_free(v_temp);
    gsl_permutation_free(p1);
}




// No iterations - old version
void get_all_lnoverlaps(
    double* st_mns, int st_mn_dim1, int st_mn_dim2,
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mn_dim1, int gr_mn_dim2,
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3,
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int memb_dim1, int memb_dim2,
    int inc_posterior, int amp_prior, double use_box_background, 
    double* lnols, int lnols_dim1, int lnols_dim2,
    int using_bg) {
    /*
     * lnols: result is saved here. This is an array [nstars, ncomps+1]
     * comps_means: an array [ncomps][ndim] where ndim=9 for spherical components
     * comps_covs: an array [ncomps][row][column] for covariance matrices NOW


    Get the log overlap integrals of each star with each component
    
    Parameters
    ----------
    data: 
    comps_list: 
    old_memb_probs: 
    inc_posterior: 
    amp_prior: 
    * 
    Returns
    -------
    lnols: 
    * 
     */
    
    
    int i, j;


    // Compute weights. These are amplitudes of components and are equal to the number of members in each of the components.
    double weights[gr_mn_dim1];
    
    double sum;
    for (j=0; j<memb_dim2-1; j++) {
        sum=0.0;
        for (i=0; i<memb_dim1; i++) {
            sum+=old_memb_probs[i*memb_dim2+j];
        }
        weights[j] = sum;
    }
  
    // Optionally scale each weight by the component prior, then rebalance
    // such that total expected stars across all components is unchanged
    // TODO: LATER
//~ //    if inc_posterior:
//~ //        comp_lnpriors = np.zeros(ncomps)
//~ //        for i, comp in enumerate(comps):
//~ //            comp_lnpriors[i] = likelihood2.ln_alpha_prior(
//~ //                    comp, memb_probs=old_memb_probs
//~ //            )
//~ //        assoc_starcount = weights.sum()
//~ //        weights *= np.exp(comp_lnpriors)
//~ //        weights = weights / weights.sum() * assoc_starcount

    // Optionally scale each weight such that the total expected stars
    // is equal to or greater than `amp_prior`
    // TODO: LATER
//~ //    if amp_prior:
//~ //        if weights.sum() < amp_prior:
//~ //            weights *= amp_prior / weights.sum()

    // For each component, get log overlap with each star, scaled by
    // amplitude (weight) of each component's pdf
    
    
    double* gr_cov;
    double* gr_mn;   

    double lnols_comp[st_mn_dim1]; // for every component
    for (i=0; i<gr_mn_dim1; i++) {
        gr_mn = &gr_mns[i*gr_dim2];
        gr_cov = &gr_covs[i*gr_dim2*gr_dim3];
                
        get_lnoverlaps(gr_cov, gr_dim2, gr_dim3, 
            gr_mn, gr_mn_dim2, 
            st_covs, st_dim1, st_dim2, st_dim3, 
            st_mns, st_mn_dim1, st_mn_dim2, 
            lnols_comp, st_mn_dim1);
        
        for (j=0; j<st_mn_dim1; j++) {
            lnols[i*st_mn_dim1+j] = log(weights[i]) + lnols_comp[j];
        }
    }

    // Insert one time calculated background overlaps
    for (i=0; i<bg_dim; i++) {
        lnols[st_mn_dim1*gr_mn_dim1+i] = bg_lnols[i];
    }

}


double get_overall_lnlikelihood(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int omemb_dim1, int omemb_dim2,
    double* memb_probs, int memb_dim1) {
        
    //~ Get overall likelihood for a proposed model.

    //~ Evaluates each star's overlap with every component and background
    //~ If only fitting one group, inc_posterior does nothing

    //~ Parameters
    //~ ----------
    //~ data: (dict)
        //~ See fit_many_comps
    //~ comps: [ncomps] list of Component objects
        //~ See fit_many_comps
    //~ return_memb_probs: bool {False}
        //~ Along with log likelihood, return membership probabilites

    //~ Returns
    //~ -------
    //~ overall_lnlikelihood: float


    //~ double memb_probs[st_mn_dim1*(gr_mns_dim1+1)];
    
    // TODO: UNCOMMENT THIS! It is commented for now because it doesn;t work and won't compile
    //~ expectation(st_mns, st_mn_dim1, st_mn_dim2, 
        //~ st_covs, st_dim1, st_dim2, st_dim3,
        //~ gr_mns, gr_mns_dim1, gr_mns_dim2, 
        //~ gr_covs, gr_dim1, gr_dim2, gr_dim3, 
        //~ bg_lnols, bg_dim,
        //~ old_memb_probs, omemb_dim1, omemb_dim2,
        //~ memb_probs, memb_dim1);


    // UNCOMMENT THIS
    //~ int lnols_dim1 = st_mn_dim1;
    //~ int lnols_dim2 = (gr_mns_dim1+1);
    //~ double all_ln_ols[lnols_dim1*lnols_dim2]; // TODO: check dimension
    //~ int inc_posterior = 0;
    //~ int amp_prior = 0;
    //~ int use_box_background = 0;
    //~ int using_bg = 1;
    //~ get_all_lnoverlaps(st_mns, st_mn_dim1, st_mn_dim2,
        //~ st_covs, st_dim1, st_dim2, st_dim3,
        //~ gr_mns, gr_mns_dim1, gr_mns_dim2,
        //~ gr_covs, gr_dim1, gr_dim2, gr_dim3,
        //~ bg_lnols, bg_dim,
        //~ old_memb_probs, omemb_dim1, omemb_dim2,
        //~ inc_posterior, amp_prior, use_box_background, 
        //~ all_ln_ols, lnols_dim1, lnols_dim2,
        //~ using_bg);
    


    //~ Multiplies each log overlap by the star's membership probability
    //~ (In linear space, takes the star's overlap to the power of its
    //~ membership probability)

    //~ #einsum is an Einstein summation convention. Not suer why it is used here???
    //~ #weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)

    // Compute weighted_lnols = all_ln_ols * memb_probs
    double sum=0.0;
    //~ for (int i=0; i<st_mn_dim1*(gr_mns_dim1+1); i++) { // CHECK DIMENSION
        //weighted_lnols[i] = all_ln_ols[i] * memb_probs[i];
        //~ sum+=all_ln_ols[i] * memb_probs[i];
    //~ }

    //~ #if np.sum(weighted_lnols) != np.sum(weighted_lnols):
    //~ #    import pdb; pdb.set_trace() #!!!!

    // READ THIS: can 'return' memb_probs in any case, because they
    // get updated here in any case

    // return_memb_probs=True only in expectmax that reads in all the previous fits. We skip this in C.
    //~ if return_memb_probs:
        //~ return np.sum(weighted_lnols), memb_probs
    //~ else:
        //~ return np.sum(weighted_lnols)
    
    return sum;
}


void calc_membership_probs(double *star_lnols, int ncomps, 
    double *star_memb_probs) {
    /*
     * Calculate probabilities of membership for a **SINGLE** star from 
     * overlaps
     * 
     * Parameters
     * ----------
     * star_lnols : [ncomps] array
     *      The log of the overlap of a star with each group
     * 
     * Returns
     * -------
     * star_memb_probs : [ncomps] array
     *     The probability of membership to each group, normalised to 
     *     sum to 1
     *   
     */
    
    // TODO: Should use GSL vectors to speed this up!

    //~ printf("start calc_membership_probs\n");

    int j;
    double sum;
    double si;
    for (int i=0; i<ncomps; i++) {
        si = star_lnols[i];
        sum=0;
        for(j=0; j<ncomps; j++) {
            sum += exp(star_lnols[j]-si);
        }
        star_memb_probs[i] = 1. / sum;
    }
    
    //~ printf("end calc_membership_probs\n");
}


void expectation(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int omemb_dim1, int omemb_dim2,
    double* memb_probs, int memb_dim1) {

    /*
     * Take stellar data and components, compute overlaps and return
     * membership probabilities for these stars to be in the components.
     * 
     * --> Cannot return 2D array with swig, but pack everything into 1D 
     * array and reshape later in python!
     * Result: memb_probs[nstars+ncomps(+1 for bg?)]
     */


    int inc_posterior = 0;
    int amp_prior = 0;
    int use_box_background = 0;
    int using_bg = 1;
    
    // Calculate all log overlaps
    int lnols_dim1 = st_mn_dim1;  
    int lnols_dim2 = gr_mns_dim1+1;
    double lnols[lnols_dim1*lnols_dim2];

    //~ printf("C start get_all_lnoverlaps\n");
    get_all_lnoverlaps(
        st_mns, st_mn_dim1, st_mn_dim2,
        st_covs, st_dim1, st_dim2, st_dim3,
        gr_mns, gr_mns_dim1, gr_mns_dim2,
        gr_covs, gr_dim1, gr_dim2, gr_dim3,
        bg_lnols, bg_dim,
        old_memb_probs, omemb_dim1, omemb_dim2,
        inc_posterior, amp_prior, use_box_background, 
        lnols, lnols_dim1, lnols_dim2,
        using_bg);        
    //~ printf("C end get_all_lnoverlaps\n");
 
    // Calculate membership probabilities, tidying up 'nan's as required
    int i, j;
    int ncomps = gr_mns_dim1+1;
    double memb_probs_i[ncomps];
    double lnols_i[gr_mns_dim1+1];
    
    for (i=0; i<st_mn_dim1; i++) {
        // TODO: AVOID THIS FOR LOOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // Get lnols
        for (j=0; j<ncomps; j++) {
            lnols_i[j] = lnols[j*st_mn_dim1+i];
        }
 
        calc_membership_probs(lnols_i, ncomps, memb_probs_i);
        //~ printf("END calc_membership_probs(lnols_i, ncomps, memb_probs_i), i=%d\n", i);
        
        for (j=0; j<ncomps; j++) {
            //~ printf("insert memb_probs i=%d, j=%d, %d, memb_probs_i[j]=%f\n",
                //~ i, j, i*ncomps+j, memb_probs_i[j]);
            memb_probs[i*ncomps+j] = memb_probs_i[j];
            //~ printf("memb_probs_i[j] INSERTED\n");
        }
        //~ printf("END memb_probs[i*ncomps+j] = memb_probs_i[j], i=%d, j=%d\n", i, j);
    }

    //~ printf("last line in expectation in C\n");

    //~ // Check if any of the probabilities is nan
    //~ if np.isnan(memb_probs).any():
        //~ print('memb_probs')
        //~ print(memb_probs)
        //~ #~ for comp in comps_list:
            //~ #~ print(comp)
        //~ #~ log_message('AT LEAST ONE MEMBERSHIP IS "NAN"', symbol='!')
        //~ print('AT LEAST ONE MEMBERSHIP IS "NAN"')
        //~ memb_probs[np.where(np.isnan(memb_probs))] = 0. # TODO: remove the if sentence and do this in any case???
}
