/*
 * Expectation module. Compute membership probabilities here.
 * Overlaps are computed here.
*/

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>


void get_lnoverlaps(double* gr_cov, int gr_dim1, int gr_dim2, 
    double* gr_mn, int gr_mn_dim, double* st_covs, int st_dim1, 
    int st_dim2, int st_dim3, double* st_mns, int st_mn_dim1, 
    int st_mn_dim2, double* lnols_output, int n) {
    
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
void get_all_lnoverlaps(double* st_mns, double* st_covs,
    double* gr_mns, double* gr_covs, double* old_memb_probs, 
    int inc_posterior, int amp_prior, double use_box_background, 
    int nstars, int ncomps, double* lnols, int using_bg) {
    /*
     * lnols: result is saved here. This is an array [nstars, ncomps+1]
     * comps_means: an array [ncomps][ndim] where ndim=9 for spherical components
     * comps_covs: an array [ncomps][row][column] for covariance matrices NOW
     */ 
    
    
    /*
    Get the log overlap integrals of each star with each component
    
    Parameters
    ----------
    data: dict -or- astropy.table.Table -or- path to astrop.table.Table
        if dict, should have following structure:
            'means': [nstars,6] float array_like
                the central estimates of star phase-space properties
            'covs': [nstars,6,6] float array_like
                the phase-space covariance matrices of stars
            'bg_lnols': [nstars] float array_like (opt.)
                the log overlaps of stars with whatever pdf describes
                the background distribution of stars.
        if table, see tabletool.build_data_dict_from_table to see
        table requirements.
    comps_list: [ncomps]
        List of components. Every element of this list has mean and cov
        for a component, e.g. [mean, cov]. 
        E.g. comps_list = [[mean, cov], [mean, cov], ...]
        This replaces 'comps' from the old chronostar
    # comps: [ncomps] syn.Group object list
    #    a fit for each comp (in internal form)
    old_memb_probs: [nstars, ncomps] float array {None}
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each comp. Each
        element is between 0.0 and 1.0 such that each row sums to 1.0
        exactly.
        If bg_hists are also being used, there is an extra column for the
        background (but note that it is not used in this function)
    inc_posterior: bool {False}
        If true, includes prior on groups into their relative weightings
    amp_prior: int {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`
        
    Returns
    -------
    lnols: [nstars, ncomps (+1)] float array
        The log overlaps of each star with each component, optionally
        with the log background overlaps appended as the final column
     */
    
    int i, j;

    // Tidy input, infer some values
    //~ // using_bg = 'bg_lnols' in data.keys()

//~ //    lnols = np.zeros((nstars, ncomps + using_bg))

    // Set up old membership probabilities. # THIS SHOULD HAPPEN AT AN UPPER LEVEL IN THE CODE! Make sure you always provide old_memb_probs
//~ //    if old_memb_probs is None:
//~ //        old_memb_probs = np.ones((nstars, ncomps)) / ncomps
    
    // Compute weights. These are amplitudes of components and are equal to the number of members in each of the components.
    double weights[ncomps];
    double sum;
    for (i=0; i<ncomps; i++) {
        sum=0.0;
        for (j=0; j<nstars; j++) {
            sum+=old_memb_probs[j*ncomps+i];
        }
        weights[i] = sum;
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
    int gr_dim1=6; // number of components
    int gr_dim2=6; 
    double* gr_mn;
    int gr_mn_dim=6;
    //~ double* st_covs; 
    int st_dim1=nstars;
    int st_dim2=6;
    int st_dim3=6;
    //~ double* st_mns; 
    int st_mn_dim1=nstars;
    int st_mn_dim2=6;
    int n=nstars;    
    
    double lnols_comp[nstars]; // for every component
    for (i=0; i<ncomps; i++) {
        gr_mn = &gr_mns[i*gr_dim2]; // does this work?
        gr_cov = &gr_covs[i*gr_dim2]; // does this work?
        
        get_lnoverlaps(gr_cov, gr_dim1, gr_dim2, 
            gr_mn, gr_mn_dim, 
            st_covs, st_dim1, st_dim2, st_dim3, 
            st_mns, st_mn_dim1, st_mn_dim2, 
            lnols_comp, n);
        
        // lnols_output is multidim...
        // lnols: result is saved here. This is an array [nstars, ncomps+1]
        for (j=0; j<nstars; j++) {
            lnols[j*ncomps+i] = log(weights[i]) + lnols_comp[j];
        }
    }

    // TODO
    //~ // insert one time calculated background overlaps
    //~ if using_bg:
        //~ lnols[:,-1] = data['bg_lnols']


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
    
    //~ for i in range(ncomps):
        //~ star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))

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
}

void expectation(double* means_stars, double* covs_stars, int nstars,
    double* means_comps, double* covs_comps, int ncomps, 
    double* memb_probs, double* old_memb_probs) {
    /*
     * Take stellar data and components, compute overlaps and return
     * membership probabilities for these stars to be in the components.
     * 
     * Result: memb_probs [nstars * ncomps(+1?)]
     */

    int inc_posterior = 0;
    int amp_prior = 0;
    int use_box_background = 0;
    int using_bg = 1;
    
    // Calculate all log overlaps
    double lnols[nstars*ncomps]; // ncomps+1?    
    get_all_lnoverlaps(means_stars, covs_stars, means_comps, covs_comps,
        old_memb_probs, inc_posterior, amp_prior, use_box_background, 
        nstars, ncomps, lnols, using_bg);

    // Calculate membership probabilities, tidying up 'nan's as required
    
    //~ memb_probs = np.zeros((nstars, ncomps + using_bg))
    //~ memb_probs is a matrix!!
    
    // for i in range(ncomps):
    //        star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))
    
    double memb_probs_i[ncomps];
    for (int i=0; i<nstars; i++) {
        calc_membership_probs(&lnols[i*ncomps], ncomps, memb_probs_i);
        for (int j=0; j<ncomps; j++) {
            memb_probs[i*ncomps+j] = memb_probs_i[j];
        }
    }
    
    // PRINT MEMBERSHIPS
    //~ for (int i=0; i<nstars; i++) {
        //~ for (int j=0; j<ncomps; j++) {
            //~ printf("%lf ", memb_probs[i*ncomps+j]);
        //~ }
        //~ printf("\n");
    //~ }
    
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
