#include <stdio.h>
#include <math.h>

#include "expectation.h" // Sort out these paths
#include "temporal_propagation.h" // Sort out these paths

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
    return log(x*sig*sqrt(2*M_PI)) - pow(log(x)-mu, 2)/(2*sig*sig);
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


// TODO
double lnprior(double* mean, double* covmatrix, double dx, double dv, 
    double age, double* memb_probs, int nstars) {
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
    
    // set maximum allowed age
    //~ double MAX_AGE = 500;
    
    ////#~ covmatrix = comp.get_covmatrix()
    //~ stds = np.linalg.eigvalsh(covmatrix)
    //~ if np.min(mean) < -100000 or np.max(mean) > 100000:
        //~ return -INFINITY
    //~ // Components can be quite large. Lets let them be as large as they like.
    //~ #~ if np.min(stds) <= 0.0: # or np.max(stds) > 10000.0:
    //~ if np.min(stds) <= 0.0 or np.max(stds) > 1e+6:
        //~ return -INFINITY
    //~ if age < 0.0 or age > MAX_AGE:
        //~ return -INFINITY

    //~ # Check covariance matrix is transform of itself
    //~ if not np.allclose(covmatrix, covmatrix.T):
        //~ return -INFINITY
    //~ # Check correlations are valid
    //~ if not np.all(np.linalg.eigvals(covmatrix) > 0):
        //~ return -INFINITY

    double sig=1.0;
    return ln_alpha_prior(dx, dv, memb_probs, sig, nstars);

}



double lnlike(double* mean_now, double* cov_now, 
    double* st_mns, double* st_covs, double* bg_lnols,
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
    
    // Boost expect star count to some minimum threshold
    // This is a bit of a hack to prevent component amplitudes dwindling
    // to nothing
    // TODO: Check if this effect is ever actually triggered...
    //~ int i;
    
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

    //~ // As a potentially negligible optimisation:
    //~ // only consider contributions of stars with larger than provided
    //~ // threshold membership prob.
    //~ nearby_star_mask = np.where(memb_probs > memb_threshold)

    //~ // Calculate log overlaps of relevant stars
    //~ lnols = np.zeros(len(memb_probs))
    //~ lnols[nearby_star_mask] = expectation.get_lnoverlaps(mean_now, 
        //~ cov_now, data, star_mask=nearby_star_mask)
        
        
    //~ double lnols[nstars];
    //~ expectation.get_lnoverlaps(mean_now, 
        //~ cov_now, data, star_mask=nearby_star_mask);
    

    //~ // Weight each stars contribution by their membership probability
    double result = 0.0;
    //~ for (i=0; i<nstars; i++) {
        //~ result += lnols[i] * memb_probs[i];
    //~ }
    
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
    // np.hstack((data['means'][i], data['covs'][i].flatten(), data['bg_lnols'][i], memb_probs[i]))
    //~ int data_dim2=44;
    int nstars = data_dim1; //sizeof(data)/sizeof(data[0])/data_dim2;
    printf("nstars %d \n", nstars);
    int means_dim = 6;
    int covs_dim1 = 6;
    int covs_dim2 = 6;
    const int covs_dim = covs_dim1*covs_dim2;
    
    double st_mns[nstars*means_dim];
    double st_covs[nstars*covs_dim1*covs_dim2];
    double bg_lnols[nstars];
    double memb_probs[nstars];
    
    for (i=0; i<nstars; i++) {
        for (j=0; j<means_dim; j++) {
            st_mns[i*means_dim+j] = data[i*data_dim2+j];
        }
        
        for (j=0; j<covs_dim; j++) {
            st_covs[i*covs_dim+j] = data[i*data_dim2+means_dim+j];
        }
        
        bg_lnols[i] = data[i*data_dim2-2];
        memb_probs[i] = data[i*data_dim2-1];
    }

    double age = pars[8];
    double dx = pars[6];
    double dv = pars[7];
    

    //~ # THIS IS WHERE TRACEFORWARD SHOULD HAPPEN! !!!!!!!!!!!!!!!!!!!
    //~ # Are pars comp_means at time=0? Then I only need to create a 
    //~ # covmatrix at time 0.
    
    //~ #~ comp = Component(emcee_pars=pars, trace_orbit_func=trace_orbit_func) # Tim's chronostar
    //~ #~ comp = Component(pars=pars, trace_orbit_func=trace_orbit_func)
    
    //~ ### IMPORTANT: TODO: Originally, 'pars' were parsed to comp as emcee_pars!!!
    
    
    // Trace component's mean forward in time
    double mean_start[means_dim];
    for (i=0; i<means_dim; i++) {
        mean_start[i]=pars[i];
    }
    double mean_now[means_dim];
    trace_epicyclic_orbit(mean_start, means_dim, age, mean_now, 
        means_dim);
    
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    printf("test\n");
    
    // Trace component's covmatrix mean forward in time
    //~ double covmatrix[covs_dim] = {};
    double covmatrix[covs_dim];
    for (i=0;  i<covs_dim; i++) covmatrix[i]=0.0;
    double dx2=dx*dx;
    double dv2=dv*dv;
    covmatrix[0] = dx2;
    covmatrix[7] = dx2;
    covmatrix[14] = dx2;
    covmatrix[21] = dv2;
    covmatrix[28] = dv2;
    covmatrix[35] = dv2;
    int cov_dim1=6;
    int cov_dim2=6;
    
    double covmatrix_now[covs_dim];
    trace_epicyclic_covmatrix(covmatrix, cov_dim1, cov_dim2,
        mean_now, means_dim, age, H, covmatrix_now, covs_dim);
    
    
  
    // Prior
    double lp = lnprior(mean_now, covmatrix_now, dx, dv, age, 
        memb_probs, nstars);
    
    if isinf(lp) {
        return INFINITY; // This avoids computation of lnlike that is expensive
    }
    
    // Ln probability
    //~ double lnprob = - (lp + lnlike(mean_now, covmatrix_now, data, memb_probs))    
    double lnprob = - (lp + lnlike(mean_now, covmatrix_now, 
        st_mns, st_covs, bg_lnols, memb_probs, nstars));
    
    return lnprob;
}
