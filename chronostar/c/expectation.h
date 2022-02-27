#ifndef EXPECTATION_H_
#define EXPECTATION_H_


void get_lnoverlaps(
    double* gr_cov, int gr_dim1, int gr_dim2, 
    double* gr_mn, int gr_mn_dim, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3, 
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* lnols_output, int n);


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
    int using_bg);

void print_bg_lnols(double* bg_lnols, int bg_dim);

void expectation_iterative_component_amplitudes(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int omemb_dim1, int omemb_dim2,
    double* memb_probs, int memb_dim1);


void calc_membership_probs(double *star_lnols, int ncomps, 
    double *star_memb_probs);


void expectation(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int omemb_dim1, int omemb_dim2,
    double* memb_probs, int memb_dim1);
    
#endif // EXPECTATION_H_
