#include <stdio.h>
#include <math.h>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "expectation.h" // Sort out these paths
#include "temporal_propagation.h" // Sort out these paths

#define MAX_AGE 500 // Max allowed age
#define G_const 0.004300917270069976  // pc (km/s)^2 / Msun
#define H 0.001 // transformation of covmatrix
// TODO: CHECK the value of H in python's chronostar and make sure they are the same!

double calc_alpha(double dx, double dv, int nstars) {
    /*
     * Assuming we have identified 100% of star mass with a 1 M_sun 
     * missing mass offset, and that average star mass is 1 M_sun.
     * 
     * alpha>1: gravitationally unbound, it is expanding.
     * alpha<1: gravitationally bound, it is collapsing.
     * 
     * Calculated alpha is unitless
     */


    // M_sol = 1 // Msun
    // return (dv**2 * dx) / (G_const * (nstars+1) * M_sol) // No need to multiply by M_sol=1
    
    return (dv*dv * dx) / (G_const * (nstars+1));
}


double lnlognormal(double x, double mu, double sig) {
    // mu=2.1, double sig=1.0
    return -log(x*sig*sqrt(2*M_PI)) - pow(log(x)-mu, 2)/(2*sig*sig); //TODO 2*M_PI could be precomputed as a header constant. What about 2*sig*sig?
}

    
double ln_alpha_prior(double dx, double dv, double* memb_probs, 
    double sig, int nstars) {
        // sig=1.0
    /*
    A very approximate, gentle prior preferring super-virial distributions

    Since alpha is strictly positive, we use a lognormal prior. We then
    take the log of the result to incorporate it into the log likelihood
    evaluation.

    Mode is set at 3, when `sig` is 1, this corresponds to a FWHM of 1 dex
    (AlphaPrior(alpha=1, sig=1.) =     AlphaPrior(alpha=11,sig=1.)
                                 = 0.5*AlphaPrior(alpha=3, sig=1.)

    Parameters
    ----------
    comp: Component object
        An object from an implementation of the AbstractComponent class.
        Encapsulates the parameters describing a component fit.
    memb_probs: [nstars] float array
        membership array
    */
    
    //~ #~ dx = comp.get_sphere_dx()
    //~ #~ dv = comp.get_sphere_dv()
    
    //~ nstars = np.sum(memb_probs)
    
    double alpha = calc_alpha(dx, dv, nstars);
    
    // TODO: hardcoded...
    double mu=2.1;
    //~ double sig=1.0;
    
    return lnlognormal(alpha, mu, sig);
}


double lnprior(double* mean, int mean_dim, double* covmatrix, 
    double dx, double dv, double age, double* memb_probs, int nstars) {
    /*
    Computes the prior of the group models constraining parameter space.

    Parameters
    ----------
    comp: Component object
        Component object encapsulating the component model
    memb_probs: [nstars] float array
        array of weights [0.0 - 1.0] for each star, describing 
        probabilty of each star being a member of component beign 
        fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters
    */
    
    int i;
    
    for (i=0; i<mean_dim; i++) {
        if ((mean[i]<-1e+5) || (mean[i]>1e+5)) return -INFINITY;
    }
    
    //~ if np.min(mean) < -100000 or np.max(mean) > 100000:
        //~ return -INFINITY
    // Components can be quite large. Lets let them be as large as they like.
    // if np.min(stds) <= 0.0: # or np.max(stds) > 10000.0:


    ////#~ covmatrix = comp.get_covmatrix()
    // NOTE: THIS SHOULD BE COVMATRIX AT TIME 0!!
    //~ stds = np.linalg.eigvalsh(covmatrix)
    // This is a diagonal matrix, so its eigenvalues are elements of the diagonal

    double dx2 = dx*dx;
    double dv2 = dv*dv;
    if ((dx<0.0) || (dv<0.0) || (dx2>1e+6) || (dv2>1e+6)) return -INFINITY;
    //~ if np.min(stds) <= 0.0 or np.max(stds) > 1e+6:
        //~ return -INFINITY

    // MZ: This is the same as above
    // Check correlations are valid
    //~ if not np.all(np.linalg.eigvals(covmatrix) > 0):
        //~ return -INFINITY

    // Check covariance matrix is transform of itself
    // MZ: this is a diagonal square matrix, and its transpose should be equal to the original matrix.
    //~ if not np.allclose(covmatrix, covmatrix.T):
        //~ return -INFINITY
    
    if ((age < 0.0) || (age > MAX_AGE)) return -INFINITY;


    // Estimated number of members. This SHOULD be done outside the lnprob function!!! TODO
    int nstrs=0;
    for (i=0; i<nstars; i++) {
        nstrs+=memb_probs[i];
    }

    double sig=1.0;
    return ln_alpha_prior(dx, dv, memb_probs, sig, nstrs);

}


double lnlike(double* gr_mn, int gr_mn_dim, 
    double* gr_cov, int gr_dim1, int gr_dim2,
    double* st_mns, int st_mn_dim1, int st_mn_dim2, 
    double* st_covs, int st_dim1, int st_dim2, int st_dim3, 
    double* memb_probs, int nstars) {

    /*
    Computes the log-likelihood for a fit to a group.

    The emcee parameters encode the modelled origin point of the stars.
    Using the parameters, a mean and covariance in 6D space are 
    constructed as well as an age. The kinematics are then projected 
    forward to the current age and compared with the current stars' 
    XYZUVW values (and uncertainties).

    P(D|G) = prod_i[P(d_i|G)^{z_i}]
    ln P(D|G) = sum_i z_i*ln P(d_i|G)

    Parameters
    ----------
    pars: [npars] list
        Parameters describing the group model being fitted
    data: dict
        traceback data being fitted to, stored as a dict:
        'means': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'covs': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    memb_probs: [nstars] float array
        array of weights [0.0 - 1.0] for each star, describing how 
        likely they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit

    */
    
    
    
    //~ // Boost expect star count to some minimum threshold
    //~ // This is a bit of a hack to prevent component amplitudes dwindling
    //~ // to nothing
    //~ // TODO: Check if this effect is ever actually triggered...
    
    //~ double memb_threshold=1e-5;
    //~ double minimum_exp_starcount=10.0;
    
    //~ int exp_starcount = 0;
    //~ for (i=0; i<nstars; i++) {
        //~ exp_starcount += memb_probs[i];
    //~ }
        
    //~ if (exp_starcount < minimum_exp_starcount) {
        //~ memb_probs = np.copy(memb_probs)
        //~ memb_probs *= minimum_exp_starcount / exp_starcount
    //~ }


    //TODO Python version filters out stars with negligible memberships at this point. However, this should be done before the optimisation is called.

    double lnols[nstars];
    get_lnoverlaps(gr_cov, gr_dim1, gr_dim2, 
        gr_mn, gr_mn_dim, 
        st_covs, st_dim1, st_dim2, st_dim3, 
        st_mns, st_mn_dim1, st_mn_dim2, 
        lnols, nstars);    
    

    // Weight each stars contribution by their membership probability
    double result = 0.0;
    for (int i=0; i<nstars; i++) {
        result += lnols[i] * memb_probs[i];
    }
    
    return result;
    
}


double lnprob_func_gradient_descent(double* pars, int pars_dim, 
    double* data, int data_dim1, int data_dim2) {
    /*
    Returns - (minus) lnprob because gradient descent is minimizing,
    not optimizing!!!!
    
    Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        e.g. for SphereComponent:
            0,1,2,3,4,5,   6,   7,  8
            X,Y,Z,U,V,W,lndX,lndV,age
    data:  np.hstack((data['means'][i], data['covs'][i].flatten(), data['bg_lnols'][i], memb_probs[i]))
    *   This is the only way I know to parse such data from python to C

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    */
    
    int i, j;
    
    // UNZIP DATA
    // np.hstack((data['means'][i], data['covs'][i].flatten(), memb_probs[i]))
    //~ int data_dim2=44;
    int nstars = data_dim1; //sizeof(data)/sizeof(data[0])/data_dim2;
    //~ printf("nstars %d \n", nstars);
    int means_dim = 6;
    int covs_dim1 = 6;
    int covs_dim2 = 6;
    const int covs_dim = covs_dim1*covs_dim2;
    
    double st_mns[nstars*means_dim];
    double st_covs[nstars*covs_dim1*covs_dim2];
    double memb_probs[nstars];
    
    for (i=0; i<nstars; i++) {
        for (j=0; j<means_dim; j++) {
            st_mns[i*means_dim+j] = data[i*data_dim2+j];
        }

        for (j=0; j<covs_dim; j++) {
            st_covs[i*covs_dim+j] = data[i*data_dim2+means_dim+j];
        }
        
        memb_probs[i] = data[(i+1)*data_dim2-1];
    }

    double age = pars[8];
    double dx = exp(pars[6]); // pars[6] is sampled in log space (emcee space in python chronostar)
    double dv = exp(pars[7]); // pars[7] is sampled in log space


    //~ # THIS IS WHERE TRACEFORWARD SHOULD HAPPEN! !!!!!!!!!!!!!!!!!!!
    //~ # Are pars comp_means at time=0? Then I only need to create a 
    //~ # covmatrix at time 0.
    

    // Trace component's mean forward in time
    double mean_start[means_dim];
    for (i=0; i<means_dim; i++) {
        mean_start[i]=pars[i];
    }

    //~ printf("mean\n");
    //~ for (i=0; i<means_dim; i++) {
        //~ printf("%f ", mean_start[i]);
    //~ }
    //~ printf("\n");

    double mean_now[means_dim];
    trace_epicyclic_orbit(mean_start, means_dim, age, mean_now, 
        means_dim);

    // TODO after Chronostar operates in pc/Myr
    // mean_start is changed by trace_epicyclic_orbit (transform to pc/Myr), so init it here again
    // Could this be omitted if Chronostar operated in pc/Myr everywhere?
    for (i=0; i<means_dim; i++) {
        mean_start[i]=pars[i];
    }
    
    //~ for (i=0; i<means_dim; i++) {
        //~ printf("before %f after %f\n", mean_start[i], mean_now[i]);
    //~ }
    
    // Trace component's covmatrix mean forward in time
    //~ double covmatrix[covs_dim] = {};
    double covmatrix[covs_dim];
    for (i=0; i<covs_dim; i++) covmatrix[i]=0.0; // TODO: init in a faster way? with {0.0}?
    double dx2=dx*dx;
    double dv2=dv*dv;
    covmatrix[0] = dx2;
    covmatrix[7] = dx2;
    covmatrix[14] = dx2;
    covmatrix[21] = dv2;
    covmatrix[28] = dv2;
    covmatrix[35] = dv2;
    int cov_dim1=6; //TODO hardcoded...
    int cov_dim2=6;
    
    
    
    double covmatrix_now[covs_dim];
    trace_epicyclic_covmatrix(covmatrix, cov_dim1, cov_dim2,
        mean_start, means_dim, age, H, covmatrix_now, covs_dim);
        
    //~ printf("mean\n");
    //~ for (i=0; i<means_dim; i++) {
        //~ printf("%f ", mean_start[i]);
    //~ }
    //~ printf("\n");
    //~ printf("mean_now\n");
    //~ for (i=0; i<means_dim; i++) {
        //~ printf("%f ", mean_now[i]);
    //~ }
    //~ printf("\n");

    //~ printf("covmatrix C\n");
    //~ for (i=0; i<covs_dim1; i++) {
        //~ for (j=0; j<covs_dim2; j++) {
            //~ printf("%f ", covmatrix[i*covs_dim1+j]);
        //~ }
        //~ printf("\n");
    //~ }

    //~ printf("covmatrix_now C\n");
    //~ for (i=0; i<covs_dim1; i++) {
        //~ for (j=0; j<covs_dim2; j++) {
            //~ printf("%f ", covmatrix_now[i*covs_dim1+j]);
        //~ }
        //~ printf("\n");
    //~ }

  
    // Prior
    double lp = lnprior(mean_start, means_dim, covmatrix, dx, dv, age, 
        memb_probs, nstars);
    
    //~ printf("lp C %g \n", lp);

    
    if isinf(lp) {
        return INFINITY; // This avoids computation of lnlike that is expensive
    }


    // Likelihood
    double lnlk = lnlike(mean_now, means_dim, 
        covmatrix_now, cov_dim1, cov_dim2,
        st_mns, nstars, means_dim, 
        st_covs, nstars, covs_dim1, covs_dim2, 
        memb_probs, nstars);
    
    
    
   // Ln probability
    double lnprob = - (lp + lnlk);
    
    //~ printf("lnprob %g\n", lnprob);
    //~ printf("C lnlik %g\n", lnlk);
    
    return lnprob;
}
