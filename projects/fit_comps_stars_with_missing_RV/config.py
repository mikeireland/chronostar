
import numpy as np

assoc_name = 'beta_pictoris'
rdir = '/home/marusa/chronostar/integration_tests/fit_comps_stars_with_missing_RV/results_with_RV/'
config = {
    # Input data
    'results_dir':rdir+'{}'.format(assoc_name),
    'data_loadfile': '/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits',
    'data_savefile':rdir+'{}/{}_subset.fit'.format(assoc_name, assoc_name),
    # 'datafile':'../results/{}/data.fits'.format(assoc_name),
    'init_comps_file': None,

    # 'background_overlaps_file':'',
    'include_background_distribution':True,
    'kernel_density_input_datafile': '/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits',
                                                    # Cartesian data of all Gaia DR2 stars
                                                    # e.g. ../data/gaia_dr2_mean_xyzuvw.npy
    'plot_it':True,
    'run_with_mpi':True,       # not yet inpmlemented
    'convert_to_cartesian':False,        # whehter need to convert data from astrometry to cartesian
    'overwrite_datafile':False,         # whether to store results in same talbe and rewrite to file
    'cartesian_savefile':rdir+'{}/{}_subset.fit'.format(assoc_name, assoc_name),
    'save_cartesian_data':True,         #
    'ncomps':10,                        # maximum number of components to reach
    'overwrite_prev_run':True,          # explores provided results directorty and sees if results already
                                        # exist, and if so picks up from where left off
    'dummy_trace_orbit_function':False,  # For testing, simple function to skip computation
    'pickup_prev_run':True,             # Pick up where left off if possible
    'banyan_assoc_name':'beta Pictoris',
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
}

advanced = {
    'burnin_steps':500,        # emcee parameters, number of steps for each burnin iteraton
    'sampling_steps':500,
    'scale_margin':1.,
    'pos_margin': 10,
    'vel_margin': 2,
}
