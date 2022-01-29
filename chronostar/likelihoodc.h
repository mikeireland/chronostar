#ifndef LIKELIHOODC_H_
#define LIKELIHOODC_H_


double calc_alpha(double dx, double dv, int nstars);


double lnlognormal(double x, double mu, double sig);

    
double ln_alpha_prior(double dx, double dv, double* memb_probs, 
    double sig, int nstars);


double lnprior(double* mean, int mean_dim, double* covmatrix, 
    double dx, double dv, double age, double* memb_probs, int nstars);


double lnlike(double* gr_mn, int gr_mn_dim, 
    double* gr_cov, int gr_dim1, int gr_dim2,
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3, 
    double* memb_probs, int nstars);


double lnprob_func_gradient_descent(double* pars, int pars_dim, 
    double* data, int data_dim1, int data_dim2);


#endif // LIKELIHOODC_H_
