#ifndef LIKELIHOODC_H_
#define LIKELIHOODC_H_


double calc_alpha(double dx, double dv, int nstars);


double lnlognormal(double x, double mu, double sig);

    
double ln_alpha_prior(double dx, double dv, double* memb_probs, 
    double sig, int nstars);


double lnprior(double* mean, double* covmatrix, double dx, double dv, 
    double age, double* memb_probs, int nstars);


double lnlike(double* mean_now, double* cov_now, 
    double* st_mns, double* st_covs, double* bg_lnols,
    double* memb_probs, int nstars);


double lnprob_func_gradient_descent(double* pars, int pars_dim, 
    double* data, int data_dim1, int data_dim2);


#endif // LIKELIHOODC_H_
