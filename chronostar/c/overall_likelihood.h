#ifndef OVERALL_LIKELIHOOD_H_
#define OVERALL_LIKELIHOOD_H_


double get_overall_lnlikelihood_for_fixed_memb_probs(
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* memb_probs, int memb_dim1, int memb_dim2);

double get_overall_lnlikelihood( // not finished
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3,
    double* gr_mns, int gr_mns_dim1, int gr_mns_dim2, 
    double* gr_covs, int gr_dim1, int gr_dim2, int gr_dim3, 
    double* bg_lnols, int bg_dim,
    double* old_memb_probs, int omemb_dim1, int omemb_dim2,
    double* memb_probs, int memb_dim1);

#endif // OVERALL_LIKELIHOOD_H_
