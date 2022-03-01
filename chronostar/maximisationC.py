"""
Maximisation step: Fit a component to the data: Find parameters of the
component at birth time that fit the data today.

This requires the component propagation in time.

"""

import numpy as np
import scipy.optimize

#~ from chronostar import likelihood2
from chronostar.component import SphereComponent

try:
    from chronostar._likelihood import lnprob_func_gradient_descent
except ImportError:
    print("C IMPLEMENTATION OF lnprob_func_gradient_descent NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?

def fit_single_comp_gradient_descent_serial(data, memb_probs=None, 
    init_pars=None, Component=SphereComponent, 
    convergence_tol=1, optimisation_method='Nelder-Mead'):
    """
    Write docs...
    """

    #~ print('init_pars', init_pars)

    init_age = init_pars[-1]
    age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [-9, -4, -0.4, -0.2, -0.5, 0., 0.1, 0.3, 0.5, 5., 10., 20., 40.]
    #~ age_offsets = [0., 10.] # for testing
    init_ages = np.abs([init_age + age_offset for age_offset in age_offsets])
    init_guess_comp = Component(emcee_pars=init_pars)
    # Age split hardcoded!
    init_guess_comps = init_guess_comp.split_group_ages(init_ages)
    init_pos = [c.get_emcee_pars() for c in init_guess_comps]
    

    # Prepare data: exclude non-members
    # This is the required C format
    a = []
    memb_threshold=1e-5
    nearby_star_mask = np.where(memb_probs > memb_threshold)
    for i in nearby_star_mask[0]:
        tmp = np.hstack((data['means'][i], data['covs'][i].flatten(), memb_probs[i]))
        a.append(tmp)
    a=np.array(a)

    return_dict={}
    for i in range(len(init_pos)):
        #~ print('init_pos[%d]'%i, init_pos)
        #~ print('optimisation_method', optimisation_method)
        result = scipy.optimize.minimize(lnprob_func_gradient_descent, 
            init_pos[i], args=a, 
            tol=convergence_tol, method=optimisation_method)
        return_dict[i] = result


    keys = list(return_dict.keys()) # Keep the keys so you always have the same order
    result_fun = [[k, return_dict[k].fun] for k in keys]
    result_fun_sorted = sorted(result_fun, key=lambda x: x[1])
    best_key = result_fun_sorted[0][0]
    best_result = return_dict[best_key]

    # Identify and create the best component (with best lnprob)
    best_component = Component(emcee_pars=best_result.x)
    
    return best_component, best_result.x, -best_result.fun # Check if really minus. Minus is already in the likelihood...


def maximisation_gradient_descent_serial(data, ncomps=None, 
    memb_probs=None, all_init_pars=None, all_init_pos=None,
    convergence_tol=1, Component=SphereComponent,
    optimisation_method='Nelder-Mead', 
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
    
    return_dict={}
    for i in range(ncomps):
        best_comp, final_pos, lnprob =\
            fit_single_comp_gradient_descent_serial(data=data, 
                memb_probs=memb_probs[:, i],
                convergence_tol=convergence_tol,
                #~ init_pos=all_init_pos[i],
                init_pars=all_init_pars[i], Component=Component,
                #~ trace_orbit_func=trace_orbit_func,
                optimisation_method=optimisation_method, # e.g. Nelder-Mead
        )
        return_dict[i] = [best_comp, lnprob, final_pos]



    new_comps_list = [return_dict[i][0] for i in range(ncomps)]
    all_lnprob = [return_dict[i][1] for i in range(ncomps)]
    all_final_pos = [return_dict[i][2] for i in range(ncomps)]

    return new_comps_list, all_lnprob, all_final_pos


def maximisation_gradient_descent_multiprocessing(data, ncomps=None, 
    memb_probs=None, all_init_pars=None, all_init_pos=None,
    convergence_tol=1, Component=SphereComponent,
    optimisation_method='Nelder-Mead', 
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
    print('IN maximisation_gradient_descent_multiprocessing')
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    #~ global worker # should solve pickle error in ipython, but it doesn't
    
    #~ return_dict={}
    #~ for i in range(ncomps):
        #~ best_comp, final_pos, lnprob =\
            #~ fit_single_comp_gradient_descent_serial(data=data, 
                #~ memb_probs=memb_probs[:, i],
                #~ convergence_tol=convergence_tol,
                #init_pos=all_init_pos[i],
                #~ init_pars=all_init_pars[i], Component=Component,
                #trace_orbit_func=trace_orbit_func,
                #~ optimisation_method=optimisation_method, # e.g. Nelder-Mead
        #~ )
        #~ return_dict[i] = [best_comp, lnprob, final_pos]




    def worker(i, return_dict):
        best_comp, final_pos, lnprob =\
            fit_single_comp_gradient_descent_serial(data=data, 
                memb_probs=memb_probs[:, i],
                convergence_tol=convergence_tol,
                init_pars=all_init_pars[i], Component=Component,
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


    print('END maximisation_gradient_descent_multiprocessing')



    new_comps_list = [return_dict[i][0] for i in range(ncomps)]
    all_lnprob = [return_dict[i][1] for i in range(ncomps)]
    all_final_pos = [return_dict[i][2] for i in range(ncomps)]

    return new_comps_list, all_lnprob, all_final_pos
