import os
import numpy as np

run_name = 'dummy_run'

output_folder = '../results'

config = {
    # INPUT DATA
    'data_loadfile':'../data/gaia_cartesian_full_6d_table.fits',
    # 'datafile':'../results/{}/data.fits'.format(run_name),

    # OUTPUT DATA
    'results_dir': os.path.join(output_folder, '{}'.format(run_name)),
    'data_savefile': os.path.join(output_folder, '{}'.format(run_name), '{}_subset.fit'.format(run_name)),

    # SETTINGS

    'plot_it':True,
    # 'background_overlaps_file':'',
    'include_background_distribution':True,
    'kernel_density_input_datafile':'../data/gaia_cartesian_full_6d_table.fits',
                                                    # Cartesian data of all Gaia DR2 stars
                                                    # e.g. ../data/gaia_dr2_mean_xyzuvw.npy
    'run_with_mpi':True,       # not yet inpmlemented
    'convert_to_cartesian':False,        # whehter need to convert data from astrometry to cartesian
    'overwrite_datafile':False,         # whether to store results in same talbe and rewrite to file
    'cartesian_savefile':'../results/{}/{}_subset.fit'.format(run_name, run_name),
    'save_cartesian_data':True,         #
    'ncomps':20,                        # maximum number of components to reach
    'overwrite_prev_run':True,          # explores provided results directorty and sees if results already
                                        # exist, and if so picks up from where left off
    'dummy_trace_orbit_function':False,  # For testing, simple function to skip computation
    'pickup_prev_run':True,             # Pick up where left off if possible
    'banyan_assoc_name':'Lower Centaurus-Crux',
    'init_comps_file':'../data/all_nonbg_scocen_comps_unique.npy',         # file that stored raw comps with which to initialise this run

    # Orbits
    'epicyclic': False,
}

synth = None
# synth = {
#     'pars':np.array([
#         [ 50., 0.,10., 0., 0., 3., 5., 2., 1e-10],
#         [-50., 0.,20., 0., 5., 2., 5., 2., 1e-10],
#         [  0.,50.,30., 0., 0., 1., 5., 2., 1e-10],
#     ]),
#     'starcounts':[100,50,50]
# }

# data_bound = {
#     'upper_bound':np.array([20.93930279, 58.41681567, 61.35019961,
#                              4.02520573, -5.38948337, 5.12689673]),
#     'lower_bound':np.array([-71.49657695, -94.28236532, -64.15451725,
#                              -6.12051672, -12.97631891,  -3.83867341]),
# }
data_bound = None

historical_colnames = True

astro_colnames = {
    # 'main_colnames':None,     # list of names
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

cart_colnames = {
    # 'main_colnames':None,
    # 'error_colnames':None,
    # 'corr_colnames':None,
}

special = {
    'component':'sphere',       # parameterisation for the origin
    'max_em_iterations':200,
}

advanced = {
    'burnin_steps':500,        # emcee parameters, number of steps for each burnin iteraton
    'sampling_steps':500,
    'store_burnin_chains':True, 
    'pos_margin':10,
    'vel_margin':2,
    'ignore_stable_comps': False,
}
