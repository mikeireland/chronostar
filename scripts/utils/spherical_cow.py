import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chronostar.component import SphereComponent
from chronostar import tabletool

# Input data
data_table = tabletool.read('beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits')
best_fit_comps = SphereComponent.load_raw_components('final_comps.npy')
memberships = np.load('final_membership.npy')
p = 0.5 # If membership probability >p, a star is a member of this component
##################

# Create data dict
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
)

labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']
plt_dir = 'output/'

labels = 'XYZUVW'
# labels = ['xi', 'eta', 'zeta', 'xi dot', 'eta dot', 'zeta dot']
units = 3*['pc'] + 3*['km/s']
# units = ['units'] * 6

# <--!!! Choose which cartesian dimensions you wish to plot !!!--> #
# <--!!! 0 -> X, 1-> Y, 2 -> Z, 3 -> U etc.                 !!!--> #
dims = [(0,1), (3,4), (0,3), (1,4)]

figsize = 10
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize, figsize))
fig.set_tight_layout(True)

for ax, (dim1, dim2) in zip(axes.flatten(), dims):
    ax.tick_params(direction='in')
    print(dim1, dim2)

    ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
    ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

    means_all = data_dict['means'] # XYZUVW


    for i, best_fit_comp in enumerate(best_fit_comps):
        best_fit_comp.plot(dim1, dim2, comp_now=True, comp_then=False, color='red', ax=ax,
                       comp_orbit=False, orbit_color='red')
        
        comp_members_mask = memberships[:,i]>p
        comp_members = means_all[comp_members_mask]
        ax.scatter(comp_members[:, dim1], comp_members[:, dim2], c='k', s=1)
        

#fig.savefig(plt_dir+'ellip_comps.png')
#print('FIG SAVED', plt_dir+'ellip_comps.png')
plt.show()
