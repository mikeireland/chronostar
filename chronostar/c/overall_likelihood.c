#include <stdio.h>
#include <stddef.h>
#include <Python.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "expectation.h"

// Why is this not in expectation.c? Answer: Swig issue.
// Swig issue: These functions need to be in a separate file because 
// they return a scalar number. If they are in expectation.c, for some
// reason swig thinks they return number + memb_probs, and this breaks
// arrays in python.

double get_overall_lnlikelihood_for_fixed_memb_probs(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* memb_probs, int memb_dim1, int memb_dim2) {
    //~ run_em.py computes expectation just before this step, so we can
    //~ take that result. No need to compute expectation again.
    //~ For now keep this separate from get_overall_lnlikelihood
    //~ because I don't know if this is called from other parts of
    //~ Chronostar and requires new memb_probs determination.
        
    //~ Get overall likelihood for a proposed model.

    //~ Evaluates each star's overlap with every component and background
    //~ If only fitting one group, inc_posterior does nothing

    //~ Parameters
    //~ ----------
    //~ data: (dict)
        //~ See fit_many_comps
    //~ comps: [ncomps] list of Component objects
        //~ See fit_many_comps
    //~ Returns
    //~ -------
    //~ overall_lnlikelihood: float


    // Output of get_all_lnoverlaps
    int lnols_dim1 = st_mn_dim1;
    int lnols_dim2 = (gr_mns_dim1+1);
    double all_ln_ols[lnols_dim1*lnols_dim2]; // TODO: check dimension
    
    // TODO: all these params?
    int inc_posterior = 0;
    int amp_prior = 0;
    int use_box_background = 0;
    int using_bg = 1;
    
    get_all_lnoverlaps(st_mns, st_mn_dim1, st_mn_dim2,
        st_covs, st_dim1, st_dim2, st_dim3,
        gr_mns, gr_mns_dim1, gr_mns_dim2,
        gr_covs, gr_dim1, gr_dim2, gr_dim3,
        bg_lnols, bg_dim,
        memb_probs, memb_dim1, memb_dim2,
        inc_posterior, amp_prior, use_box_background, 
        all_ln_ols, lnols_dim1, lnols_dim2,
        using_bg);

    // Multiplies each log overlap by the star's membership probability
    // (In linear space, takes the star's overlap to the power of its
    // membership probability)

    // #einsum is an Einstein summation convention. Not suer why it is used here???
    // #weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)

    // Compute weighted_lnols = all_ln_ols * memb_probs
    //TODO some ols and/or memb_probs are 0. Can I skip them here to speed it up?
    double result=0.0;
    for (int i=0; i<st_mn_dim1; i++) {
        for(int j=0; j<lnols_dim2; j++) {
            result += all_ln_ols[j*st_mn_dim1+i] * memb_probs[i*memb_dim2+j];
        }
    }

    return result;
}


double get_overall_lnlikelihood( // not finished
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
    
    // WHY DO WE NEED MEMB_PROBS HERE? TODO
    expectation(st_mns, st_mn_dim1, st_mn_dim2, 
        st_covs, st_dim1, st_dim2, st_dim3,
        gr_mns, gr_mns_dim1, gr_mns_dim2, 
        gr_covs, gr_dim1, gr_dim2, gr_dim3, 
        bg_lnols, bg_dim,
        old_memb_probs, omemb_dim1, omemb_dim2,
        memb_probs, memb_dim1);


    // UNCOMMENT THIS
    int lnols_dim1 = st_mn_dim1;
    int lnols_dim2 = (gr_mns_dim1+1);
    double all_ln_ols[lnols_dim1*lnols_dim2]; // TODO: check dimension
    int inc_posterior = 0;
    int amp_prior = 0;
    int use_box_background = 0;
    int using_bg = 1;
    get_all_lnoverlaps(st_mns, st_mn_dim1, st_mn_dim2,
        st_covs, st_dim1, st_dim2, st_dim3,
        gr_mns, gr_mns_dim1, gr_mns_dim2,
        gr_covs, gr_dim1, gr_dim2, gr_dim3,
        bg_lnols, bg_dim,
        old_memb_probs, omemb_dim1, omemb_dim2,
        inc_posterior, amp_prior, use_box_background, 
        all_ln_ols, lnols_dim1, lnols_dim2,
        using_bg);
    


    // Multiplies each log overlap by the star's membership probability
    // (In linear space, takes the star's overlap to the power of its
    // membership probability)

    // #einsum is an Einstein summation convention. Not suer why it is used here???
    // #weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)

    // Compute weighted_lnols = all_ln_ols * memb_probs
    double sum=0.0;
    for (int i=0; i<st_mn_dim1*(gr_mns_dim1+1); i++) { // CHECK DIMENSION
        //~ //weighted_lnols[i] = all_ln_ols[i] * memb_probs[i];
        sum+=all_ln_ols[i] * memb_probs[i];
    }

    //#if np.sum(weighted_lnols) != np.sum(weighted_lnols):
    //#    import pdb; pdb.set_trace() #!!!!

    //~ // READ THIS: can 'return' memb_probs in any case, because they
    //~ // get updated here in any case

    //~ // return_memb_probs=True only in expectmax that reads in all the previous fits. We skip this in C.
    //~ if return_memb_probs:
        //~ return np.sum(weighted_lnols), memb_probs
    //~ else:
        //~ return np.sum(weighted_lnols)
    
    return sum;
}

