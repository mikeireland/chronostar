#include <stdio.h>
#include <math.h>

#include "expectation.h" // Sort out these paths

#define G_const 0.004300917270069976  // pc (km/s)^2 / Msun

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
    
    return (dv**2 * dx) / (G_const * (nstars+1));
}


double lnlognormal(double x, double mu=2.1, double sig=1.0) {
    return log(x*sig*sqrt(2*M_PI)) - (log(x)-mu)**2/(2*sig**2);
}

    
double ln_alpha_prior(double dx, double dv, double* memb_probs, 
    double sig=1.0, int nstars):
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
    double sig=1.0;
    
    return lnlognormal(alpha, mu, sig)

// TODO
double lnprior(double* mean, double* covmatrix, double dx, double dv, 
    double age, double* memb_probs) {
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
    double MAX_AGE = 500;
    
    //~ #~ covmatrix = comp.get_covmatrix()
    stds = np.linalg.eigvalsh(covmatrix)
    if np.min(mean) < -100000 or np.max(mean) > 100000:
        return -np.inf
    // Components can be quite large. Lets let them be as large as they like.
    #~ if np.min(stds) <= 0.0: # or np.max(stds) > 10000.0:
    if np.min(stds) <= 0.0 or np.max(stds) > 1e+6:
        return -np.inf
    if age < 0.0 or age > MAX_AGE:
        return -np.inf

    # Check covariance matrix is transform of itself
    if not np.allclose(covmatrix, covmatrix.T):
        return -np.inf
    # Check correlations are valid
    if not np.all(np.linalg.eigvals(covmatrix) > 0):
        return -np.inf

    return ln_alpha_prior(dx, dv, memb_probs, sig=1.0)

}



double lnlike(double* mean_now, double* cov_now, double* data, 
    double* memb_probs, int nstars, double memb_threshold=1e-5,
    double minimum_exp_starcount=10.) {
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
    int i;
    
    int exp_starcount = 0;
    for (i=0; i<nstars; i++) {
        exp_starcount += memb_probs[i];
    }
        
    if (exp_starcount < minimum_exp_starcount) {
        memb_probs = np.copy(memb_probs)
        memb_probs *= minimum_exp_starcount / exp_starcount
    }

    // As a potentially negligible optimisation:
    // only consider contributions of stars with larger than provided
    // threshold membership prob.
    nearby_star_mask = np.where(memb_probs > memb_threshold)

    // Calculate log overlaps of relevant stars
    lnols = np.zeros(len(memb_probs))
    lnols[nearby_star_mask] = expectation.get_lnoverlaps(mean_now, 
        cov_now, data, star_mask=nearby_star_mask)
        
        
    double lnols[nstars];
    expectation.get_lnoverlaps(mean_now, 
        cov_now, data, star_mask=nearby_star_mask);
    

    // Weight each stars contribution by their membership probability
    double result = 0.0;
    for (i=0; i<nstars; i++) {
        result += lnols[i] * memb_probs[i];
    }
    
    return result;
}


double lnprob_func_gradient_descent(pars, data, memb_probs=None, 
    trace_orbit_func=None, Component=SphereComponent, **kwargs) {
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
    data
    data: dict
        'means': [nstars,6] float array_like
            the central estimates of star phase-space properties
        'covs': [nstars,6,6] float array_like
            the phase-space covariance matrices of stars
        'bg_lnols': [nstars] float array_like (opt.)
            the log overlaps of stars with whatever pdf describes
            the background distribution of stars.
    memb_probs
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.
    Component: Class implmentation of component.AbstractComponent
        A class that can read in `pars`, and generate the three key
        attributes for the modelled origin point:
        mean, covariance matrix, age
        As well as get_current_day_projection()
        See AbstractComponent to see which methods must be implemented
        for a new model.
    trace_orbit_func: function {None}
        A function that, given a starting phase-space position, and an
        age, returns a new phase-space position. Leave as None to use
        default (traceorbit.trace_cartesian_orbit) set in the abstract
        component initializer.
    kwargs:
        Any extra parameters will be carried over to lnlike.
        As of 2019-12-04 this feature has never been utilised.

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    */
        
    
    # TODO: THis is not OK but it works.
    # scipy optimizer works differently to emcee and it packs all
    # arguments in the data keyword. emcee is different.
    #~ if type(data)==list:
        #~ args=data
        #~ optimisation_method=args[3]
        #~ if optimisation_method=='Nelder-Mead':
            #~ # args = [data, memb_probs, trace_orbit_func]
            #~ memb_probs = args[1]
            #~ trace_orbit_func = args[2]
            #~ data=args[0]

    args=data
    optimisation_method=args[3]

    # args = [data, memb_probs, trace_orbit_func]
    memb_probs = args[1]
    trace_orbit_func = args[2]
    data=args[0]

    age = pars[8]
    dx = pars[6]
    dv = pars[7]
    
    mean = pars[:6]

    
    #~ if memb_probs is None:
        #~ memb_probs = np.ones(len(data['means']))
    #~ print('likelihood:', pars, trace_orbit_func)
    
    # THIS IS WHERE TRACEFORWARD SHOULD HAPPEN! !!!!!!!!!!!!!!!!!!!
    # Are pars comp_means at time=0? Then I only need to create a 
    # covmatrix at time 0.
    
    #~ comp = Component(emcee_pars=pars, trace_orbit_func=trace_orbit_func) # Tim's chronostar
    #~ comp = Component(pars=pars, trace_orbit_func=trace_orbit_func)
    
    ### IMPORTANT: TODO: Originally, 'pars' were parsed to comp as emcee_pars!!!
    
    # NEED TO DETERMINE COVMATRIX FOR pars
    #~ covmatrix = component2.compute_covmatrix_spherical(mean)
    covmatrix = np.identity(6)
    covmatrix[:3, :3] *= dx ** 2
    covmatrix[3:, 3:] *= dv ** 2
    
    # Trace component forward in time
    mean_now = tp.trace_epicyclic_orbit(mean, times=age)
    covmatrix_now = tp.trace_epicyclic_covmatrix(covmatrix, loc=mean, 
        age=age)
    

    #~ dx = np.sqrt(covmatrix[0,0])
    #~ dv = np.sqrt(covmatrix[-1,-1])
    
    #~ dx = comp.get_sphere_dx()
    #~ dv = comp.get_sphere_dv()
    #~ dx2 = component2.get_dx_from_covmatrix(covmatrix)
    #~ dv2 = component2.get_dv_from_covmatrix(covmatrix)
    
    
    lp = lnprior(mean, covmatrix, dx, dv, age, memb_probs)
    if not np.isfinite(lp):
        return np.inf# Should we include this? What is the efficiency of this?
    
    lnprob = - (lp + lnlike(mean_now, covmatrix_now, data, memb_probs, 
        **kwargs))
    
    return lnprob;
}
