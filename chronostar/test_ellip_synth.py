import numpy as np
from chronostar.synthdata import SynthData
from chronostar.component import EllipComponent
from chronostar import tabletool as tt
import matplotlib.pyplot as plt

XU_CORR = 0.5
NSTARS = 1000

# Initialise pars with everything = 0
my_free_pars = np.zeros(len(EllipComponent.PARAMETER_FORMAT))

my_free_pars[6:10] = 1.      # Set standard deviations to 1.
my_free_pars[14] = XU_CORR   # Set XU correlation to XU_CORR

my_free_comp = EllipComponent(pars=my_free_pars)

my_synth_data = SynthData(pars=my_free_pars, starcounts=NSTARS, Components=EllipComponent)
my_synth_data.generate_all_init_cartesian()
# my_synth_data.synthesise_everything()         # Don't actually need everything

mean_colnames = [el for el in 'XYZUVW']
# mean_colnames = [el+'0' for el in 'xyzuvw']     # Use this for initial star positions
# mean_colnames = [el+'_now' for el in 'xyzuvw']  # Use this for current day star positions,
                                                  # will need to uncomment synthesise_everything() though

means = tt.build_data_dict_from_table(
        my_synth_data.table[2:],
        main_colnames=mean_colnames,
        only_means=True,
)

plt.plot(means[:,0], means[:,3], '.')
my_synth_data.components[0].plot('X', 'U', comp_then=True)
plt.xlabel('X')
plt.ylabel('U')
plt.savefig('xu.png')
plt.clf()

print("here", my_synth_data.table.colnames)

# my_synth_data.store_table(filename='../scripts/data_table.fits', overwrite=True)
my_synth_data.table.write('../scripts/data_table.fits', overwrite=True, format='fits')
plt.plot(means[:,0], means[:,4], '.')
my_synth_data.components[0].plot('X', 'V', comp_then=True)
plt.xlabel('X')
plt.ylabel('V')
plt.savefig('xv.png')
plt.clf()

xu_pos = means[:,np.array([0,3])]

calculated_corr = np.cov(xu_pos.T)[0,1]
print('calculated corr:    ', calculated_corr)
print('Actual correlation: ', XU_CORR)