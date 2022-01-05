"""
Expectation module. Overlaps are computed here.
"""

import numpy as np

try:
    from ._overlap import get_lnoverlaps as c_get_lnoverlaps
except ImportError:
    print("C IMPLEMENTATION OF GET_OVERLAP NOT IMPORTED")
    USE_C_IMPLEMENTATION = False


from chronostar.component import SphereComponent
from chronostar import temporal_propagation # This shouldn't be here but should be done at a higher level

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


# This is also used in likelihood
def get_lnoverlaps(mean_now, cov_now, data, star_mask=None):
    """
    Calculate overlaps between stars and the component.
    There is NO TEMPORAL PROPAGATION of the component here. It must
    already be provided in the frame you want the overlap in.
    THIS IS ON TODO LIST

    Utilises Overlap, a c module wrapped with swig to be callable by 
    python. This allows a 100x speed up in our 6x6 matrix operations 
    when compared to numpy.

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
    #~ mean_now, cov_now = comp.get_currentday_projection()
    # TODO: This should already be propagated when passed as an argument
    # to the function!!! SO there should be no need to import Component
    # here, or temporal_propagation.py.
        
    
    # Calculate overlap integral of each star
    try:
        lnols = c_get_lnoverlaps(cov_now, mean_now, star_covs, 
            star_means, star_count)
    except:
        lnols = slow_get_lnoverlaps(cov_now, mean_now, star_covs, 
            star_means)
    
    return lnols

# No iterations - old version
def get_all_lnoverlaps(data, comps_list, old_memb_probs=None,
    inc_posterior=False, amp_prior=None, use_box_background=False):
    """
    Get the log overlap integrals of each star with each component
    
    Parameters
    ----------
    data: dict -or- astropy.table.Table -or- path to astrop.table.Table
        if dict, should have following structure:
            'means': [nstars,6] float array_like
                the central estimates of star phase-space properties
            'covs': [nstars,6,6] float array_like
                the phase-space covariance matrices of stars
            'bg_lnols': [nstars] float array_like (opt.)
                the log overlaps of stars with whatever pdf describes
                the background distribution of stars.
        if table, see tabletool.build_data_dict_from_table to see
        table requirements.
    comps_list: [ncomps]
        List of components. Every element of this list has mean and cov
        for a component, e.g. [mean, cov]. 
        E.g. comps_list = [[mean, cov], [mean, cov], ...]
        This replaces 'comps' from the old chronostar
    # comps: [ncomps] syn.Group object list
    #    a fit for each comp (in internal form)
    old_memb_probs: [nstars, ncomps] float array {None}
        Only used to get weights (amplitudes) for each fitted component.
        Tracks membership probabilities of each star to each comp. Each
        element is between 0.0 and 1.0 such that each row sums to 1.0
        exactly.
        If bg_hists are also being used, there is an extra column for the
        background (but note that it is not used in this function)
    inc_posterior: bool {False}
        If true, includes prior on groups into their relative weightings
    amp_prior: int {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`
        
    Returns
    -------
    lnols: [nstars, ncomps (+1)] float array
        The log overlaps of each star with each component, optionally
        with the log background overlaps appended as the final column
    """
    
    # THIS IS REPEATED IN 'expectation'. Why?????
    # Tidy input, infer some values
    if not isinstance(data, dict):
        data = tabletool.build_data_dict_from_table(data)
    nstars = len(data['means'])
    ncomps = len(comps_list)
    using_bg = 'bg_lnols' in data.keys()

    lnols = np.zeros((nstars, ncomps + using_bg))

    # Set up old membership probabilities
    if old_memb_probs is None:
        old_memb_probs = np.ones((nstars, ncomps)) / ncomps
    weights = old_memb_probs[:, :ncomps].sum(axis=0)

    # Optionally scale each weight by the component prior, then rebalance
    # such that total expected stars across all components is unchanged
    if inc_posterior:
        comp_lnpriors = np.zeros(ncomps)
        for i, comp in enumerate(comps):
            comp_lnpriors[i] = likelihood2.ln_alpha_prior(
                    comp, memb_probs=old_memb_probs
            )
        assoc_starcount = weights.sum()
        weights *= np.exp(comp_lnpriors)
        weights = weights / weights.sum() * assoc_starcount

    # Optionally scale each weight such that the total expected stars
    # is equal to or greater than `amp_prior`
    if amp_prior:
        if weights.sum() < amp_prior:
            weights *= amp_prior / weights.sum()

    # For each component, get log overlap with each star, scaled by
    # amplitude (weight) of each component's pdf
    for i, comp in enumerate(comps_list):
        mean_now = comp[0]
        cov_now = comp[1]
        lnols[:, i] = \
            np.log(weights[i]) + \
            get_lnoverlaps(mean_now, cov_now, data)
            #~ likelihood.get_lnoverlaps(comp, data)

    # insert one time calculated background overlaps
    if using_bg:
        lnols[:,-1] = data['bg_lnols']
    return lnols


# TODO: change comps to comps_list
def get_overall_lnlikelihood(data, comps_list, return_memb_probs=False,
                             old_memb_probs=None,
                             inc_posterior=False,
                             use_box_background=False):
    """
    Get overall likelihood for a proposed model.

    Evaluates each star's overlap with every component and background
    If only fitting one group, inc_posterior does nothing

    Parameters
    ----------
    data: (dict)
        See fit_many_comps
    comps: [ncomps] list of Component objects
        See fit_many_comps
    return_memb_probs: bool {False}
        Along with log likelihood, return membership probabilites

    Returns
    -------
    overall_lnlikelihood: float
    """
    memb_probs = expectation(data, comps_list,
                             old_memb_probs=old_memb_probs,
                             inc_posterior=inc_posterior,
                             use_box_background=use_box_background)

    all_ln_ols = get_all_lnoverlaps(data, comps_list,
                                    old_memb_probs=memb_probs,
                                    inc_posterior=inc_posterior,
                                    use_box_background=use_box_background)

    # multiplies each log overlap by the star's membership probability
    # (In linear space, takes the star's overlap to the power of its
    # membership probability)

    #einsum is an Einstein summation convention. Not suer why it is used here???
    #weighted_lnols = np.einsum('ij,ij->ij', all_ln_ols, memb_probs)

    weighted_lnols = all_ln_ols * memb_probs

    #if np.sum(weighted_lnols) != np.sum(weighted_lnols):
    #    import pdb; pdb.set_trace() #!!!!

    if return_memb_probs:
        return np.sum(weighted_lnols), memb_probs
    else:
        return np.sum(weighted_lnols)

def calc_membership_probs(star_lnols):
    """
    Calculate probabilities of membership for a single star from 
    overlaps

    Parameters
    ----------
    star_lnols : [ncomps] array
        The log of the overlap of a star with each group

    Returns
    -------
    star_memb_probs : [ncomps] array
        The probability of membership to each group, normalised to sum 
        to 1
    """
    ncomps = star_lnols.shape[0]
    star_memb_probs = np.zeros(ncomps)

    print(star_lnols)
    print(star_lnols.shape)

    # TODO: expsum? logsumexp - but not relevant here as no log
    # Avoid the loop?
    for i in range(ncomps):
        star_memb_probs[i] = 1. / np.sum(np.exp(star_lnols - star_lnols[i]))

    return star_memb_probs


def expectation(data, comps_list, old_memb_probs=None, 
    inc_posterior=False, amp_prior=None, use_box_background=False):
    """
    
    use_box_background=False does not do anything
                   
                    
    Calculate membership probabilities given fits to each group
    
    Parameters
    ----------
    data: dict
        See fit_many_comps
    comps: [ncomps] Component list
        The best fit for each component from previous runs
    old_memb_probs: [nstars, ncomps (+1)] float array
        Memberhsip probability of each star to each fromponent. 
        Only used here to set amplitudes of each component.
        Must be provided!
    inc_posterior: bool {False}
        Whether to rebalance the weighting of each component by their
        relative priors
    amp_prior: float {None}
        If set, forces the combined ampltude of Gaussian components to be
        at least equal to `amp_prior`
    Returns
    -------
    memb_probs: [nstars, ncomps] float array
        An array designating each star's probability of being a member to
        each component. It is populated by floats in the range (0.0, 1.0) such
        that each row sums to 1.0, each column sums to the expected size of
        each component, and the entire array sums to the number of stars.
    """
    
    # Tidy input and infer some values
    #~ if not isinstance(data, dict):
        #~ data = tabletool.build_data_dict_from_table(data)
    ncomps = len(comps_list)
    nstars = len(data['means'])
    using_bg = 'bg_lnols' in data.keys() #TODO: move this up in the algorithm

    # if no memb_probs provided, assume perfectly equal membership
    # MZ: THIS MUST BE PROVIDED IN THIS FUNCTION! Deal with the missing
    # memb_probs earlier in the algoritm
    #~ if old_memb_probs is None:
        #~ old_memb_probs = np.ones((nstars, ncomps+using_bg)) / (ncomps+using_bg)

    # Calculate all log overlaps
    lnols = get_all_lnoverlaps(data, comps_list, old_memb_probs,
        inc_posterior=inc_posterior, amp_prior=amp_prior)

    # Calculate membership probabilities, tidying up 'nan's as required
    memb_probs = np.zeros((nstars, ncomps + using_bg))
    for i in range(nstars):
        memb_probs[i] = calc_membership_probs(lnols[i])
    # TODO: LIST COMPREHENSION!!!    
    
    if np.isnan(memb_probs).any():
        print('memb_probs')
        print(memb_probs)
        #~ for comp in comps_list:
            #~ print(comp)
        #~ log_message('AT LEAST ONE MEMBERSHIP IS "NAN"', symbol='!')
        print('AT LEAST ONE MEMBERSHIP IS "NAN"')
        memb_probs[np.where(np.isnan(memb_probs))] = 0. # TODO: remove the if sentence and do this in any case???
        
    return memb_probs

if __name__=='__main__':
    import pickle
    
    with open('../fastfit/data_for_testing/input_data_to_expectation.pkl', 'rb') as f:
        input_data = pickle.load(f)

    data_dict, comps_new, memb_probs_old, inc_posterior, use_box_background = input_data
    
    comps_new = [[c.get_mean_now(), c.get_covmatrix_now()] for c in comps_new]
    
    c = comps_new[0]
    print(c[0])
    print(c[1])
    
    
    e = expectation(data_dict, comps_new, memb_probs_old, 
        inc_posterior=inc_posterior, 
        use_box_background=use_box_background)
    
    with open('../fastfit/data_for_testing/output_data_from_expectation.pkl', 'rb') as f:
        e_original = pickle.load(f)

    diff = e - e_original
    mask = diff>1e-10
    print("Number of stars with membership probability different to the one from Tim's Chronostar:", np.sum(mask))
