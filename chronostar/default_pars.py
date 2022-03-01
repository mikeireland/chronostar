"""
Default parameters in Chronostar

THIS SHOULD GO TO CONFIG
"""

pars = {
    'component': 'sphere',
    'filename_init_comps': None,
    'filename_init_memb_probs': None,
    'use_background': True,
    'max_em_iterations': 100,
    'min_em_iterations': 30,
    'bic_conv_tol': 0.1,
    'lnlike_convergence_slice_size': 10, # This should be at most 'min_em_iterations'/3 or smaller
    'component': 'sphere',
    
    
    # INPUT
    'data_table': None,
    
    # Column name for stellar IDs. This is used at the end when generating
    # final fits table with IDs and membership probabilities.
    # This is optional.
    'stellar_id_colname': None,


    # File name that points to a stored list of components, typically from
    # a previous fit. Some example filenames could be:
    #  - 'some/prev/fit/final_comps.npy
    #  - 'some/prev/fit/2/A/final_comps.npy
    # Alternatively, if you already have the list of components, just
    # provide them to `init_comps`. Don't do both.
    # 'init_comps_file':None, # TODO: Is this redundant with 'init_comps'
    'init_comps': None,

    # One of these two are required if initialising a run with ncomps != 1

    # One can also initialise a Chronostar run with memberships.
    # Array is [nstars, ncomps] float array
    # Each row should sum to 1.
    # Same as in 'final_membership.npy'
    # TODO: implement this in a way that info can be passed in from text file
    #       e.g. a path to a file name
    #       for now, can only be used from within a script, i.e. given a numpy
    #       array object
    'init_memb_probs': None,   


    # Model
    'component': 'sphere',
    'max_comp_count': 20,

    # It turns out that in scipy.optimize.maximize tol=1 is optimal...
    'optimisation_method': 'Nelder-Mead',
    'convergence_tol': 1,
    
    # TODO: organise this together with min_em_iterations, lnlike_convergence_slice_size
    # EM convergence criterion: when the median values of lnprob slices change less than X (fraction)
    'EM_convergence_requirement': 0.03, # fraction of maximal lnprob change


    # How to split group: in age or in space?
    'split_group_method': 'age',
    'split_label': '',


    'trace_orbit_func': 'epicyclic',

    # Convergence criteria for when a fit_many_comps run has converged
    'bic_conv_tol':0.1, # TODO: NOT TESTED!
    'use_background':True,
    'use_box_background':False,
    
    
    # TODO: DELETE THIS!!!
    'historical_colnames': False,


    # OUTPUT
    'overwrite_prev_run': False,
    'folder_destination': 'result',

    # For every component of each iteration
    'filename_best_comp_fit': 'best_comp_fit.npy',
    
    # Each iteration
    'filename_iter_memberships': 'membership.npy',
    'filename_iter_comps': 'best_comps.npy',
    'filename_iter_lnprob_and_bic': 'lnprob_bic.npy',
    'filename_lnprob_convergence': 'lnprob_convergence.png',

    'filename_ABC_all_bics': 'all_bics.npy',
    'filename_ABC_all_bics_figure': 'all_bics.pdf',
    'filename_figure_bics': 'bics.pdf',

    'filename_bics_list': 'bic_list.npy',
    'filename_lihelihood_and_bic': 'likelihood_post_and_bic.npy',

    # Results
    'filename_final_memberships': 'final_memberships.npy',
    'filename_final_components': 'final_comps.npy',
    'filename_final_lnprob_and_bic': 'final_lnprob_and_bic.npy',
    
    
    'par_log_file': 'fit_pars.log',
    'folder_tmp': 'tmp',
    'filename_log': 'log.log',


}

