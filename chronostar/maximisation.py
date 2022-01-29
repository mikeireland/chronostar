"""
Maximisation step: Fit a component to the data: Find parameters of the
component at birth time that fit the data today.

This requires the component propagation in time.

"""

import numpy as np
#~ import multiprocessing
import multiprocess as multiprocessing
import scipy.optimize

from chronostar import likelihood2
from chronostar.component import SphereComponent

def fit_comp_gradient_descent_multiprocessing(data, memb_probs=None, 
    init_pos=None, init_pars=None, Component=SphereComponent, 
    convergence_tol=1, max_iter=None, trace_orbit_func=None,
    optimisation_method='Nelder-Mead'):
    """
    MZ: xhanged the code but not the docs...
    
    Fits a single 6D gaussian to a weighted set (by membership
    probabilities) of stellar phase-space positions.

    Stores the final sampling chain and lnprob in `save_dir`, but also
    returns the best fit (walker step corresponding to maximum lnprob),
    sampling chain and lnprob.

    If neither init_pos nor init_pars are provided, then the weighted
    mean and covariance of the provided data set are calculated, then
    used to generate a sample parameter list (using Component). Walkers
    are then initialised around this parameter list.

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
    memb_probs: [nstars] float array_like
        Membership probability (from 0.0 to 1.0) for each star to the
        component being fitted.
    init_pos: [ngroups, npars] array
        The precise locations at which to initiate the walkers. Generally
        the saved locations from a previous, yet similar run.
    init_pars: [npars] array
        the position in parameter space about which walkers should be
        initialised. The standard deviation about each parameter is
        hardcoded as INIT_SDEV
    burnin_steps: int {1000}
        Number of steps per each burnin iteration
    Component: Class implementation of component.AbstractComponent {Sphere Component}
        A class that can read in `pars`, and generate the three key
        attributes for the modelled origin point:
        mean, covariance matrix, age
        As well as get_current_day_projection()
        See AbstractComponent to see which methods must be implemented
        for a new model.
    plot_it: bool {False}
        Whether to generate plots of the lnprob in 'plot_dir'
    pool: MPIPool object {None}
        pool of threads to execute walker steps concurrently
    convergence_tol: float {0.25}
        How many standard deviations an lnprob chain is allowed to vary
        from its mean over the course of a burnin stage and still be
        considered "converged". Default value allows the median of the
        final 20 steps to differ by 0.25 of its standard deviations from
        the median of the first 20 steps.
    plot_dir: str {''}
        The directory in which to store plots
    save_dir: str {''}
        The directory in which to store results and/or byproducts of fit
    sampling_steps: int {None}
        If this is set, after convergence, a sampling stage will be
        entered. Only do this if a very fine map of the parameter
        distributions is required, since the burnin stage already
        characterises a converged solution for "burnin_steps".
    max_iter: int {None}
        The maximum iterations permitted to run. (Useful for expectation
        maximisation implementation triggering an abandonment of rubbish
        components). If left as None, then run will continue until
        convergence.
    trace_orbit_func: function {None}
        A function to trace cartesian oribts through the Galactic potential.
        If left as None, will use traceorbit.trace_cartesian_orbit (base
        signature of any alternate function on this ones)
    optimisation_method: str {'emcee'}
        Optimisation method to be used in the maximisation step to fit
        the model. Default: emcee. Available: scipy.optimise.minimize with
        the Nelder-Mead method. Note that in case of the gradient descent,
        no chain is returned and meds and spans cannot be determined.
    nprocess_ncomp: bool {False}
        Compute maximisation in parallel? This is relevant only in case
        Nelder-Mead method is used: This method computes optimisation
        many times with different initial positions. The result is the 
        one with the best likelihood. These optimisations are computed
        in parallel if nprocess_ncomp equals True.
        
    Returns
    -------
    best_component
        The component model which yielded the highest posterior probability
    chain
        [nwalkers, nsteps, npars] array of all samples
    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """

    """
    Run optimisation multiple times and select result with the
    best lnprob value as the best. Reason is that Nelder-Mead method
    turned out not to be robust enough so we need to run optimisation
    with a few different starting points. 
    
    scipy.optimize.minimize is using -likelihood.lnprob_func because
    it is minimizing rather than optimizing.
    """


    ######## PRINT FOR TESTING PURPOSES ################################
    #~ import pickle
    #~ with open('fit_comp_data.pkl', 'wb') as f:
        #~ pickle.dump([data, memb_probs, init_pos, init_pars], f)
    #~ print('PICKLE DUMPED')

    ####################################################################







    print('IN fit_comp_gradient_descent_multiprocessing')





    # Initialise initial positions to represent components of slightly differing ages
    #~ if init_pars is None:
        #~ init_pars = get_init_emcee_pars(data=data, memb_probs=memb_probs,
                                        #~ Component=Component)
    
    
    # TODO: move this to a much higher level to avoid too many if sentences in iterations...!!!!!!!!!!!!!!!!
    #~ if init_pars is None:
        #~ init_pars = get_init_emcee_pars(data, memb_probs=memb_probs,
                                        #~ Component=Component)

    init_age = init_pars[-1]
    age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [0., 10.] # for testing
    init_ages = np.abs([init_age + age_offset for age_offset in age_offsets])
    init_guess_comp = Component(emcee_pars=init_pars)
    init_guess_comps = init_guess_comp.split_group_ages(init_ages)
    init_pos = [c.get_emcee_pars() for c in init_guess_comps]
    

    # Comment out all these 3 lines
    #~ npars = len(Component.PARAMETER_FORMAT)
    #~ nwalkers = 2*npars
    #~ print('len(init_pos), nwalkers', len(init_pos), nwalkers)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def worker(i, pos, return_dict):
        result = scipy.optimize.minimize(
            likelihood2.lnprob_func_gradient_descent, pos, 
            args=[data, memb_probs, trace_orbit_func, 
            optimisation_method], tol=convergence_tol, 
            method=optimisation_method)
        #~ print('RESULT i', i, result)
        return_dict[i] = result
        #~ logging.info('         res: %5.2f | %5.3f'%(result.x[-1], -result.fun))

    jobs = []
    for i in range(len(init_pos)):
        process = multiprocessing.Process(target=worker, 
            args=(i, init_pos[i], return_dict))
        jobs.append(process)

    # Start the processes
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()

    
    #~ print('return_dict', return_dict)
    
    keys = list(return_dict.keys()) # Keep the keys so you always have the same order
    result_fun = [[k, return_dict[k].fun] for k in keys]
    result_fun_sorted = sorted(result_fun, key=lambda x: x[1])
    best_key = result_fun_sorted[0][0]
    best_result = return_dict[best_key]

    # Identify and create the best component (with best lnprob)
    best_component = Component(emcee_pars=best_result.x)
    
    #~ best_component_mean = best_result.x
    #~ best_component_cov = component2.compute_covmatrix_spherical(dx, dv)
    #~ best_component_cov = component2.compute_covmatrix_spherical(dx, dv)

    #~ print('best_component')
    #~ print(best_component)
    
    print('END fit_comp_gradient_descent_multiprocessing')

    return best_component, best_result.x, -best_result.fun # Check if really minus. Minus is already in the likelihood...



def fit_comp_gradient_descent_serial(data, memb_probs=None, 
    init_pos=None, init_pars=None, Component=SphereComponent, 
    convergence_tol=1, max_iter=None, trace_orbit_func=None,
    optimisation_method='Nelder-Mead'):
    """
    MZ: xhanged the code but not the docs...
    
    Fits a single 6D gaussian to a weighted set (by membership
    probabilities) of stellar phase-space positions.

    Stores the final sampling chain and lnprob in `save_dir`, but also
    returns the best fit (walker step corresponding to maximum lnprob),
    sampling chain and lnprob.

    If neither init_pos nor init_pars are provided, then the weighted
    mean and covariance of the provided data set are calculated, then
    used to generate a sample parameter list (using Component). Walkers
    are then initialised around this parameter list.

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
    memb_probs: [nstars] float array_like
        Membership probability (from 0.0 to 1.0) for each star to the
        component being fitted.
    init_pos: [ngroups, npars] array
        The precise locations at which to initiate the walkers. Generally
        the saved locations from a previous, yet similar run.
    init_pars: [npars] array
        the position in parameter space about which walkers should be
        initialised. The standard deviation about each parameter is
        hardcoded as INIT_SDEV
    burnin_steps: int {1000}
        Number of steps per each burnin iteration
    Component: Class implementation of component.AbstractComponent {Sphere Component}
        A class that can read in `pars`, and generate the three key
        attributes for the modelled origin point:
        mean, covariance matrix, age
        As well as get_current_day_projection()
        See AbstractComponent to see which methods must be implemented
        for a new model.
    plot_it: bool {False}
        Whether to generate plots of the lnprob in 'plot_dir'
    pool: MPIPool object {None}
        pool of threads to execute walker steps concurrently
    convergence_tol: float {0.25}
        How many standard deviations an lnprob chain is allowed to vary
        from its mean over the course of a burnin stage and still be
        considered "converged". Default value allows the median of the
        final 20 steps to differ by 0.25 of its standard deviations from
        the median of the first 20 steps.
    plot_dir: str {''}
        The directory in which to store plots
    save_dir: str {''}
        The directory in which to store results and/or byproducts of fit
    sampling_steps: int {None}
        If this is set, after convergence, a sampling stage will be
        entered. Only do this if a very fine map of the parameter
        distributions is required, since the burnin stage already
        characterises a converged solution for "burnin_steps".
    max_iter: int {None}
        The maximum iterations permitted to run. (Useful for expectation
        maximisation implementation triggering an abandonment of rubbish
        components). If left as None, then run will continue until
        convergence.
    trace_orbit_func: function {None}
        A function to trace cartesian oribts through the Galactic potential.
        If left as None, will use traceorbit.trace_cartesian_orbit (base
        signature of any alternate function on this ones)
    optimisation_method: str {'emcee'}
        Optimisation method to be used in the maximisation step to fit
        the model. Default: emcee. Available: scipy.optimise.minimize with
        the Nelder-Mead method. Note that in case of the gradient descent,
        no chain is returned and meds and spans cannot be determined.
    nprocess_ncomp: bool {False}
        Compute maximisation in parallel? This is relevant only in case
        Nelder-Mead method is used: This method computes optimisation
        many times with different initial positions. The result is the 
        one with the best likelihood. These optimisations are computed
        in parallel if nprocess_ncomp equals True.
        
    Returns
    -------
    best_component
        The component model which yielded the highest posterior probability
    chain
        [nwalkers, nsteps, npars] array of all samples
    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """

    """
    Run optimisation multiple times and select result with the
    best lnprob value as the best. Reason is that Nelder-Mead method
    turned out not to be robust enough so we need to run optimisation
    with a few different starting points. 
    
    scipy.optimize.minimize is using -likelihood.lnprob_func because
    it is minimizing rather than optimizing.
    """


    ######## PRINT FOR TESTING PURPOSES ################################
    #~ import pickle
    #~ with open('fit_comp_data.pkl', 'wb') as f:
        #~ pickle.dump([data, memb_probs, init_pos, init_pars], f)
    #~ print('PICKLE DUMPED')

    ####################################################################







    print('IN fit_comp_gradient_descent_serial')





    # Initialise initial positions to represent components of slightly differing ages
    #~ if init_pars is None:
        #~ init_pars = get_init_emcee_pars(data=data, memb_probs=memb_probs,
                                        #~ Component=Component)
    
    
    # TODO: move this to a much higher level to avoid too many if sentences in iterations...!!!!!!!!!!!!!!!!
    #~ if init_pars is None:
        #~ init_pars = get_init_emcee_pars(data, memb_probs=memb_probs,
                                        #~ Component=Component)

    init_age = init_pars[-1]
    age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [0., 10.] # for testing
    init_ages = np.abs([init_age + age_offset for age_offset in age_offsets])
    init_guess_comp = Component(emcee_pars=init_pars)
    init_guess_comps = init_guess_comp.split_group_ages(init_ages)
    init_pos = [c.get_emcee_pars() for c in init_guess_comps]
    

    # Comment out all these 3 lines
    #~ npars = len(Component.PARAMETER_FORMAT)
    #~ nwalkers = 2*npars
    #~ print('len(init_pos), nwalkers', len(init_pos), nwalkers)

    return_dict={}
    for i in range(len(init_pos)):
        
        # print for C
        import pickle
        with open('lnprob_func_gradient_descent_%d.pkl'%i, 'wb') as h:
            pickle.dump([data, memb_probs, init_pos[i]], h)
        print('lnprob_func_gradient_descent_ pickled', i)
        
        # TODO: filter out stars with small memb_prob. This is not done anymore in likelihood. nearby_star_mask = np.where(memb_probs > memb_threshold)
        result = scipy.optimize.minimize(
            likelihood2.lnprob_func_gradient_descent, init_pos[i], 
            args=[data, memb_probs, trace_orbit_func, 
            optimisation_method], tol=convergence_tol, 
            method=optimisation_method)
        return_dict[i] = result


    
    #~ print('return_dict', return_dict)
    
    keys = list(return_dict.keys()) # Keep the keys so you always have the same order
    result_fun = [[k, return_dict[k].fun] for k in keys]
    result_fun_sorted = sorted(result_fun, key=lambda x: x[1])
    best_key = result_fun_sorted[0][0]
    best_result = return_dict[best_key]

    # Identify and create the best component (with best lnprob)
    best_component = Component(emcee_pars=best_result.x)
    
    #~ best_component_mean = best_result.x
    #~ best_component_cov = component2.compute_covmatrix_spherical(dx, dv)
    #~ best_component_cov = component2.compute_covmatrix_spherical(dx, dv)

    #~ print('best_component')
    #~ print(best_component)
    
    print('END fit_comp_gradient_descent_serial')

    return best_component, best_result.x, -best_result.fun # Check if really minus. Minus is already in the likelihood...



def maximisation_gradient_descent_multiprocessing(data, ncomps=None, 
    memb_probs=None, all_init_pars=None, all_init_pos=None,
    convergence_tol=1, Component=SphereComponent,
    trace_orbit_func=None, optimisation_method='Nelder-Mead', 
    idir=None):
    """
    What is idir?
    """
    
    """
    I get lots pf pickling errors.
    
    From stackoverflow:
    
    The multiprocessing module has a major limitation when it comes to IPython use:

    Functionality within this package requires that the __main__ module be importable by the children. [...] This means that some examples, such as the multiprocessing.pool.Pool examples will not work in the interactive interpreter. [from the documentation]

    Fortunately, there is a fork of the multiprocessing module called multiprocess which uses dill instead of pickle to serialization and overcomes this issue conveniently.

    Just install multiprocess and replace multiprocessing with multiprocess in your imports
    """

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    #~ global worker # should solve pickle error in ipython, but it doesn't

    print('IN maximisation_gradient_descent_multiprocessing')

    def worker(i, return_dict):
        best_comp, final_pos, lnprob =\
            fit_comp_gradient_descent_multiprocessing(data=data, 
                memb_probs=memb_probs[:, i],
                convergence_tol=convergence_tol,
                init_pos=all_init_pos[i],
                init_pars=all_init_pars[i], Component=Component,
                trace_orbit_func=trace_orbit_func,
                optimisation_method=optimisation_method, # e.g. Nelder-Mead
        )

        return_dict[i] = [best_comp, lnprob, final_pos]


    jobs = []
    for i in range(ncomps):
        process = multiprocessing.Process(target=worker, 
            args=(i, return_dict))
        jobs.append(process)

    # Start the processes
    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()


    print('return_dict', return_dict)
    print('ncomps', ncomps)

    new_comps_list = [return_dict[i][0] for i in range(ncomps)]
    all_lnprob = [return_dict[i][1] for i in range(ncomps)]
    all_final_pos = [return_dict[i][2] for i in range(ncomps)]


    print('END maximisation_gradient_descent_multiprocessing')

    return new_comps_list, all_lnprob, all_final_pos


def maximisation_gradient_descent_serial(data, ncomps=None, 
    memb_probs=None, all_init_pars=None, all_init_pos=None,
    convergence_tol=1, Component=SphereComponent,
    trace_orbit_func=None, optimisation_method='Nelder-Mead', 
    idir=None):
    """
    What is idir?
    """
    
    """
    I get lots pf pickling errors.
    
    From stackoverflow:
    
    The multiprocessing module has a major limitation when it comes to IPython use:

    Functionality within this package requires that the __main__ module be importable by the children. [...] This means that some examples, such as the multiprocessing.pool.Pool examples will not work in the interactive interpreter. [from the documentation]

    Fortunately, there is a fork of the multiprocessing module called multiprocess which uses dill instead of pickle to serialization and overcomes this issue conveniently.

    Just install multiprocess and replace multiprocessing with multiprocess in your imports
    """


    print('IN maximisation_gradient_descent_serial')
    
    return_dict={}
    for i in range(ncomps):
        best_comp, final_pos, lnprob =\
            fit_comp_gradient_descent_serial(data=data, 
                memb_probs=memb_probs[:, i],
                convergence_tol=convergence_tol,
                init_pos=all_init_pos[i],
                init_pars=all_init_pars[i], Component=Component,
                trace_orbit_func=trace_orbit_func,
                optimisation_method=optimisation_method, # e.g. Nelder-Mead
        )
        return_dict[i] = [best_comp, lnprob, final_pos]



    new_comps_list = [return_dict[i][0] for i in range(ncomps)]
    all_lnprob = [return_dict[i][1] for i in range(ncomps)]
    all_final_pos = [return_dict[i][2] for i in range(ncomps)]


    print('END maximisation_gradient_descent_serial')

    return new_comps_list, all_lnprob, all_final_pos

