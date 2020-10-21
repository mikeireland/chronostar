import numpy as np
import sys
sys.path.insert(0, "..")

from chronostar.synthdata import SynthData
from chronostar.component import EllipComponent
from chronostar import tabletool as tt
from chronostar import compfitter
from astropy.table import Table
import matplotlib.pyplot as plt

XU_CORR = 0.8
NSTARS = 100

# Initialise pars with everything = 0
my_free_pars = np.zeros(len(EllipComponent.PARAMETER_FORMAT))
print('EllipComponent.PARAMETER_FORMAT', EllipComponent.PARAMETER_FORMAT)
print('my_free_pars', len(my_free_pars))



        #~ -8.50025273e-01,  -3.47011420e+00,   1.04158669e-01,
         #~ 7.65501339e+01,   1.14611034e+02,   2.18017742e+01,
         #~ 6.88541797e+00,   1.03224232e-01,   1.71029792e-02,
        #~ -5.83117935e-01,   2.59619018e+00,   1.77044518e+01]

#     PARAMETER_FORMAT = [‘pos’, ‘pos’, ‘pos’, ‘vel’, ‘vel’, ‘vel’,
#                         ‘scaled_log_std’, ‘scaled_log_vel_std’, ‘scaled_log_vel_std’,
#                         ‘quat’, ‘quat’, ‘quat’, ‘quat’,
#                         ‘corr’, ‘age’]`
my_free_pars[0] = 31.0
my_free_pars[1] = 50.0
my_free_pars[2] = 24.0
my_free_pars[3] = -0.8
my_free_pars[4] = 1.0
my_free_pars[5] = 0.5
my_free_pars[6] = 5     # dX pc
my_free_pars[7] = 2     # dV or dU? km/s
my_free_pars[8] = 2     # dV or dU? km/s
my_free_pars[9] = 0.9     # Set quaternians to 1(?)
my_free_pars[10] = 0.3
my_free_pars[11] = 0.7
my_free_pars[12] = 0.5
my_free_pars[13] = XU_CORR   # Set XU correlation to XU_CORR
my_free_pars[-1] = 19. # Age in Myr

print('my_free_pars', my_free_pars)
# my_free_comp = EllipComponent(pars=my_free_pars)
#
# print('my_free_comp', my_free_comp)

my_synth_data = SynthData(pars=my_free_pars, starcounts=NSTARS, Components=EllipComponent)
#~ my_synth_data.generate_all_init_cartesian()
my_synth_data.synthesise_everything()         # Don't actually need everything

# mean_colnames = [el for el in 'XYZUVW']
# mean_colnames = [el+'0' for el in 'xyzuvw']     # Use this for initial star positions
mean_colnames = [el+'_now' for el in 'xyzuvw']  # Use this for current day star positions,
                                                  # will need to uncomment synthesise_everything() though

means = tt.build_data_dict_from_table(
        my_synth_data.table[:],
        main_colnames=mean_colnames,
        only_means=True,
)

my_table = my_synth_data.table

plt.clf()
# plt.plot(means[:,0], means[:,3], '.')
plt.plot(my_table['x_now'], my_table['u_now'], '.', c='r')
plt.plot(my_table['x0'], my_table['u0'], '.', c='b')
my_synth_data.components[0].plot('X', 'U', comp_then=True, comp_now=True, comp_orbit=True)
plt.xlabel('X')
plt.ylabel('U')
plt.savefig('../scripts/synthData_plot_of_xu.png')
plt.clf()

plt.plot(my_table['x_now'], my_table['y_now'], '.', c='r')
plt.plot(my_table['x0'], my_table['y0'], '.', c='b')
my_synth_data.components[0].plot('X', 'Y', comp_then=True, comp_now=True, comp_orbit=True)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('../scripts/synthData_plot_of_xy.png')
plt.clf()

plt.plot(my_table['z_now'], my_table['w_now'], '.', c='r')
plt.plot(my_table['z0'], my_table['w0'], '.', c='b')
my_synth_data.components[0].plot('Z', 'W', comp_then=True, comp_now=True, comp_orbit=True)
plt.xlabel('Z')
plt.ylabel('W')
plt.savefig('../scripts/synthData_plot_of_zw.png')
plt.clf()

# plt.plot(means[:,0], means[:,4], '.')
plt.plot(my_table['y_now'], my_table['v_now'], '.', c='r')
plt.plot(my_table['y0'], my_table['v0'], '.', c='b')
my_synth_data.components[0].plot('Y', 'V', comp_then=True, comp_now=True, comp_orbit=True)
plt.xlabel('Y')
plt.ylabel('V')
plt.savefig('../scripts/synthData_plot_of_yv.png')
plt.clf()

xu_pos = means[:,np.array([0,3])]

print('means')
print(means)

data_filename = '../scripts/synthData_ellip.fits'
tt.convert_table_astro2cart(my_synth_data.table, write_table=True,
                                       filename=data_filename)
#  res = compfitter.fit_comp(
#          data=my_synth_data.table,
#          plot_it=True,
#          burnin_steps=burnin_step,
#          plot_dir=plot_dir,
#          save_dir=save_dir,
#          trace_orbit_func=trace_orbit_func,
#  )

# my_table = my_synth_data.table
# print(len(my_table))
# print(len(means))
# print(my_table.colnames)
# my_table['X'] = means[:,0]
# my_table['Y'] = means[:,1]
# my_table['Z'] = means[:,2]
# my_table['U'] = means[:,3]
# my_table['V'] = means[:,4]
# my_table['W'] = means[:,5]
# my_table['X_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['Y_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['Z_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['U_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['V_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['W_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# background_log_overlap = np.genfromtxt('../scripts/bgols_multiprocessing.dat', dtype=None)
# my_table['background_log_overlap'] = background_log_overlap
# print(my_table)
# my_table.write('../scripts/data_table.fits', overwrite=True, format='fits')
#
# calculated_corr = np.cov(xu_pos.T)[0,1]
# print('calculated corr:    ', calculated_corr)
# print('Actual correlation: ', XU_CORR)




# import numpy as np
# from chronostar.synthdata import SynthData
# from chronostar.component import EllipComponent
# from chronostar import tabletool as tt
# from astropy.table import Table
# import matplotlib.pyplot as plt
#
# XU_CORR = 0.5
# NSTARS = 1000
#
# # Initialise pars with everything = 0
# my_free_pars = np.zeros(len(EllipComponent.PARAMETER_FORMAT))
#
# my_free_pars[6:10] = 1.      # Set standard deviations to 1.
# my_free_pars[13] = XU_CORR   # Set XU correlation to XU_CORR
#
# my_free_comp = EllipComponent(pars=my_free_pars)
#
# my_synth_data = SynthData(pars=my_free_pars, starcounts=NSTARS, Components=EllipComponent)
# my_synth_data.generate_all_init_cartesian()
# # my_synth_data.synthesise_everything()         # Don't actually need everything
#
# # mean_colnames = [el for el in 'XYZUVW']
# mean_colnames = [el+'0' for el in 'xyzuvw']     # Use this for initial star positions
# # mean_colnames = [el+'_now' for el in 'xyzuvw']  # Use this for current day star positions,
#                                                   # will need to uncomment synthesise_everything() though
#
# means = tt.build_data_dict_from_table(
#         my_synth_data.table[:],
#         main_colnames=mean_colnames,
#         only_means=True,
# )
#
# plt.plot(means[:,0], means[:,3], '.')
# my_synth_data.components[0].plot('X', 'U', comp_then=True)
# plt.xlabel('X')
# plt.ylabel('U')
# plt.savefig('xu.png')
# plt.clf()
#
# plt.plot(means[:,0], means[:,4], '.')
# my_synth_data.components[0].plot('X', 'V', comp_then=True)
# plt.xlabel('X')
# plt.ylabel('V')
# plt.savefig('xv.png')
# plt.clf()
#
# xu_pos = means[:,np.array([0,3])]
#
# my_table = my_synth_data.table
# print(len(my_table))
# print(len(means))
# print(my_table.colnames)
# my_table['X'] = means[:,0]
# my_table['Y'] = means[:,1]
# my_table['Z'] = means[:,2]
# my_table['U'] = means[:,3]
# my_table['V'] = means[:,4]
# my_table['W'] = means[:,5]
# my_table['X_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['Y_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['Z_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['U_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['V_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# my_table['W_error'] = np.concatenate(0.05*np.random.rand(1000,1))
# background_log_overlap = np.genfromtxt('../scripts/bgols_multiprocessing.dat', dtype=None)
# my_table['background_log_overlap'] = background_log_overlap
# print(my_table)
# my_table.write('../scripts/data_table.fits', overwrite=True, format='fits')
#
# calculated_corr = np.cov(xu_pos.T)[0,1]
# print('calculated corr:    ', calculated_corr)
# print('Actual correlation: ', XU_CORR)
