"""
A module containing all the required functions to evaluate the Bayesian
posterior of a Component model given the data.

The entry point function lnprob_func yields a score (higher is better)
for a given model for the data. This function can be given to an
emcee Sampler object for the model's parameter space to be explored.

Bayes' Theorem states that the posterior of a model given the data
(goodness of the fit) is proportional to the product of the prior
belief on model parameters with the likelihood of the seeing the data
given the model. For convenience this equation is calculated in
log form. That is:
ln P(M|D) \propto ln P(M) + ln P(D|M)

In this module, lnprob_func calculates P(M|D)
lnlike calculates ln P(D|M)
lnprior calculates ln P(M)

A simple example to consider is finding the posterior probabilty
of a proposed normal distribution given some N data points
D distributed over X. The chance we see one individual data point x
given the model M is P(d|M), which we can find by evaluating the
normal distribution at x.

To find the combined probability of seeing every data point in D,
given the model of M, we take the product of the model evaluated at each
data point:
P(D|M) = P(x_1|M) * P(x_2|M) * .. * P(x_N|M) = \prod_i^N P(x_i|M)
"""
import numpy as np

from chronostar.component import SphereComponent
#~ from chronostar import component
#~ SphereComponent = component.SphereComponent
#~ from . import component
USE_C_IMPLEMENTATION = True
try:
    from ._overlap import get_lnoverlaps as c_get_lnoverlaps
except ImportError:
    print("C IMPLEMENTATION OF GET_OVERLAP NOT IMPORTED")
    USE_C_IMPLEMENTATION = False

def slow_get_lnoverlaps(g_cov, g_mn, st_covs, st_mns, dummy=None):
    """
    A pythonic implementation of overlap integral calculation.
    Left here in case swigged _overlap doesn't work.

    Parameters
    ---------
    g_cov: ([6,6] float array)
        Covariance matrix of the group
    g_mn: ([6] float array)
        mean of the group
    st_covs: ([nstars, 6, 6] float array)
        covariance matrices of the stars
    st_mns: ([nstars, 6], float array)
        means of the stars
    dummy: {None}
        a place holder parameter such that this function's signature
        matches that of the c implementation, which requires an
        explicit size of `nstars`.

    Returns
    -------
    ln_ols: ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    lnols = []
    for st_cov, st_mn in zip(st_covs, st_mns):
        res = 0
        res -= 6 * np.log(2*np.pi)
        res -= np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        res -= np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        res *= 0.5
        lnols.append(res)
    return np.array(lnols)


def calc_alpha(dx, dv, nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.
    
    alpha>1: gravitationally unbound, it is expanding.
    alpha<1: gravitationally bound, it is collapsing.

    Calculated alpha is unitless
    """
    # G_const taken from astropy
    # G_const = 4.30211e-3 #pc (km/s)^2 / M_sol
    G_const = 0.004302113488372941  # pc (km/s)^2 / Msun
    G_const = 0.004300917270069976  # pc (km/s)^2 / Msun
    M_sol = 1. # Msun
    return (dv**2 * dx) / (G_const * nstars * M_sol)


def lnlognormal(x, mu=2.1, sig=1.0):
    return -np.log(x*sig*np.sqrt(2*np.pi)) - (np.log(x)-mu)**2/(2*sig**2)


def ln_alpha_prior(comp, memb_probs, sig=1.0):
    """
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
    """
    dx = comp.get_sphere_dx()
    dv = comp.get_sphere_dv()
    nstars = np.sum(memb_probs)
    alpha = calc_alpha(dx, dv, nstars)
    return lnlognormal(alpha, mu=2.1, sig=sig)


def lnprior(comp, memb_probs):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    comp: Component object
        Component object encapsulating the component model
    memb_probs: [nstars] float array
        array of weights [0.0 - 1.0] for each star, describing probabilty
        of each star being a member of component beign fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters
    """
    # set maximum allowed age
    MAX_AGE = 500
    covmatrix = comp.get_covmatrix()
    stds = np.linalg.eigvalsh(covmatrix)
    if np.min(comp.get_mean()) < -100000 or np.max(comp.get_mean()) > 100000:
        return -np.inf
    # Components can be quite large. Lets let them be as large as they like.
    #~ if np.min(stds) <= 0.0: # or np.max(stds) > 10000.0:
    if np.min(stds) <= 0.0 or np.max(stds) > 1e+6:
        return -np.inf
    if comp.get_age() < 0.0 or comp.get_age() > MAX_AGE:
        return -np.inf

    # Check covariance matrix is transform of itself
    if not np.allclose(covmatrix, covmatrix.T):
        return -np.inf
    # Check correlations are valid
    if not np.all(np.linalg.eigvals(covmatrix) > 0):
        return -np.inf

    return ln_alpha_prior(comp, memb_probs, sig=1.0)


def get_lnoverlaps(comp, data, star_mask=None):
    """
    Given the parametric description of an origin, calculate star overlaps

    Utilises Overlap, a c module wrapped with swig to be callable by python.
    This allows a 100x speed up in our 6x6 matrix operations when compared
    to numpy.

    Parameters
    ----------
    pars: [npars] list
        Parameters describing the origin of group
        typically [X,Y,Z,U,V,W,np.log(dX),np.log(dV),age]
    data: dict
        stellar cartesian data being fitted to, stored as a dict:
        'means': [nstars,6] float array
            the central estimates of each star in XYZUVW space
        'covs': [nstars,6,6] float array
            the covariance of each star in XYZUVW space
    star_mask: [len(data)] indices
        A mask that excludes stars that have negliglbe membership probablities
        (and thus have their log overlaps scaled to tiny numbers).
    """
    # Prepare star arrays
    if star_mask is not None:
        star_means = data['means'][star_mask]
        star_covs = data['covs'][star_mask]
    else:
        star_means = data['means']
        star_covs = data['covs']

    star_count = len(star_means)

    # Get current day projection of component
    mean_now, cov_now = comp.get_currentday_projection()

    # Calculate overlap integral of each star
    if USE_C_IMPLEMENTATION:
        #~ print(cov_now, mean_now, star_count)
        lnols = c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means,
                                 star_count)
    else:
        lnols = slow_get_lnoverlaps(cov_now, mean_now, star_covs, star_means)
    return lnols


def lnlike(comp, data, memb_probs, memb_threshold=1e-5,
           minimum_exp_starcount=10.):
    """Computes the log-likelihood for a fit to a group.

    The emcee parameters encode the modelled origin point of the stars.
    Using the parameters, a mean and covariance in 6D space are constructed
    as well as an age. The kinematics are then projected forward to the
    current age and compared with the current stars' XYZUVW values (and
    uncertainties)

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
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit

    """
    # Boost expect star count to some minimum threshold
    # This is a bit of a hack to prevent component amplitudes dwindling
    # to nothing
    # TODO: Check if this effect is ever actually triggered...
    exp_starcount = np.sum(memb_probs)
    if exp_starcount < minimum_exp_starcount:
        memb_probs = np.copy(memb_probs)
        memb_probs *= minimum_exp_starcount / exp_starcount

    # As a potentially negligible optimisation:
    # only consider contributions of stars with larger than provided
    # threshold membership prob.
    nearby_star_mask = np.where(memb_probs > memb_threshold)

    # Calculate log overlaps of relevant stars
    lnols = np.zeros(len(memb_probs))
    lnols[nearby_star_mask] = get_lnoverlaps(comp, data,
                                             star_mask=nearby_star_mask)

    # Weight each stars contribution by their membership probability
    result = np.sum(lnols * memb_probs)
    return result


def lnprob_func(pars, data, memb_probs=None,
                trace_orbit_func=None, optimisation_method=None,
                Component=SphereComponent, **kwargs):
    """Computes the log-probability for a fit to a group.

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
    """
    
    # TODO: THis is not OK but it works.
    # scipy optimizer works differently to emcee and it packs all
    # arguments in the data keyword. emcee is different.
    if type(data)==list:
        args=data
        optimisation_method=args[3]
        if optimisation_method=='Nelder-Mead':
            # args = [data, memb_probs, trace_orbit_func]
            memb_probs = args[1]
            trace_orbit_func = args[2]
            data=args[0]

    
    if memb_probs is None:
        memb_probs = np.ones(len(data['means']))
    comp = Component(emcee_pars=pars, trace_orbit_func=trace_orbit_func)
    lp = lnprior(comp, memb_probs)
    
    if optimisation_method=='emcee':
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(comp, data, memb_probs, **kwargs)
    
    elif optimisation_method=='Nelder-Mead':
        if not np.isfinite(lp):
            return np.inf
        return - (lp + lnlike(comp, data, memb_probs, **kwargs))
