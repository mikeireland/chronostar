#ifndef EXPECTATION_H_
#define EXPECTATION_H_


void get_lnoverlaps(double* gr_cov, int gr_dim1, int gr_dim2, 
    double* gr_mn, int gr_mn_dim, double* st_covs, int st_dim1, 
    int st_dim2, int st_dim3, double* st_mns, int st_mn_dim1, 
    int st_mn_dim2, double* lnols_output, int n);


// No iterations - old version
void get_all_lnoverlaps(double st_mns, double* st_covs,
    double* gr_mn, double* gr_cov, double* old_memb_probs, 
    int inc_posterior, int amp_prior, double use_box_background, 
    int nstars, int ncomps, double* lnols, int using_bg);


void calc_membership_probs(double *star_lnols, int ncomps, 
    double *star_memb_probs);


void expectation(double* means_stars, double* covs_stars, int nstars,
    double* means_comps, double* covs_comps, int ncomps, 
    double* memb_probs, double* old_memb_probs);
    
#endif // EXPECTATION_H_
