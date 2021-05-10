"""
Take components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

MZ: It fails in python2 (cannot import emcee).

"""

import numpy as np
import sys
sys.path.insert(0, '/Users/marusa/chronostar/')
from chronostar.component import SphereComponent as Component
from chronostar import tabletool
from chronostar import coordinate
from chronostar import expectmax
from chronostar import likelihood
from astropy.table import Table, vstack, join


####### Input data ##############################
banyan_filename = 'banyan_sigma_parameters.fits'

# List of Banyan associations you want to include. Note that these names
# *MUST* match those from the Banyan table.
#~ list_of_banyan_associations = ['BPMG  ', 'HYA   '] # Must include spaces!
list_of_banyan_associations = ['BPMG  '] # Must include spaces!

# Filename with stellar data in XYZUVW space
#~ data_filename = 'beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits' # This is Gaia DR2
#~ data_filename = 'best_rv_cartesian.fits' # This is Gaia DR2

data_filename = 'best_rv_cartesian_with_rvs_renamed_to_radial_velocity.fits' # This is Gaia DR2. Because Chronostar expects radial velocity information to be provided in the 'radial_velocity' and 'radial_velocity_error' columns, I renamed Gaia rvs to 'radial_velocity_DR2' and 'best_rv' to 'radial_velocity' + same for RV uncertainties.

output_filename = 'banyan_component_overlaps_bpic_LSR.fits'
#################################################

banyan = Table.read(banyan_filename)
banyan_dict = dict(zip(banyan['NAME'], banyan))

data_table = tabletool.read(data_filename)

# This table is masked (missing values in the table are masked). Unmask:
data_table=data_table.filled()

############################################################################
############ COMPONENT OVERLAPS ############################################
############################################################################

historical = 'c_XU' in data_table.colnames
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        historical=historical,
)


# This part takes in a component at time TODAY. Everything needs to be in the XYZUVW space
"""
cov_now: component covariance matrix today
mean_now: component coordinates today in XYZUVW space
star_covs: a list of covariance matrices for stars today
star_means: a list of mean positions for stars today
star_count: how many stars are in this list
"""
star_means = data_dict['means']
star_covs = data_dict['covs']
star_count = len(star_means)

tab = Table((data_table['source_id'], ), names=('source_id', ))
#~ tab = Table((data_table['main_designation'],), names=('source_id', )) # FIX THIS. This is for 'beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits'

for assoc in list_of_banyan_associations:
    print(assoc)
    b = banyan_dict[assoc]
    
    cov_now = b['COVARIANCE_MATRIX']
    mean_now = b['CENTER_VEC']
    

    # TODO: Check if BANYAN DATA IS IN LSR... Banyan and Chronostar BPic coordinates do not match, but they match best if I assume Banyan is in heliocentric system and I convert it into LSR. X-coordinate does still not match at all.
    print('Converting helio to LSR')
    mean_now = coordinate.convert_helio2lsr(mean_now)


    #~ lnols = likelihood.c_get_lnoverlaps(cov_now, mean_now, star_covs, star_means, star_count) # This doesn't work...
    
    # log overlaps
    lnols = likelihood.slow_get_lnoverlaps(cov_now, mean_now, star_covs, star_means)

    # Save this somewhere...
    tab[assoc] = lnols



    # PLOTTING

    ##~ attributes_now = {'mean': mean_now, 'covmatrix': cov_now}
    ##~ best_fit_comps = [Component(attributes=attributes_now)]

    # When initialising Component, params should be at time 0. However, in this case, the age is set to 0 and no orbital tracing is performed. The component is thus plotted at time TODAY.
    attributes_now = {'mean': mean_now, 'covmatrix': cov_now}
    comp_now = Component(attributes=attributes_now)
    comp_now._set_age(age=-24)
    print('Age', comp_now.get_age())
    
    mean_then = comp_now.get_mean_now()
    cov_then = comp_now.get_covmatrix_now()
    attributes_then = {'mean': mean_then, 'covmatrix': cov_then}
    comp_then = Component(attributes=attributes_then)
    comp_then._set_age(age=24)
    
    best_fit_comps = [comp_then]


    #~ banyan_xyzuvw = mean_now # np.array([-4.1,-6.7,-15.7,-10.9,-16.0,-9.0])
    #~ chrono_xyzuvw_now = coo.convert_helio2lsr(banyan_xyzuvw)
    #~ chrono_xyzuvw_then = torb.trace_cartesian_orbit(chrono_xyzuvw_now, times=-24)

    
    
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

    # Only plot data with lnols above this limit
    lnols_limit_for_plotting = -33

    for ax, (dim1, dim2) in zip(axes.flatten(), dims):
        ax.tick_params(direction='in')
        #~ print(dim1, dim2)

        ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
        ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

        means_all = mean_now # XYZUVW


        for i, best_fit_comp in enumerate(best_fit_comps):
            best_fit_comp.plot(dim1, dim2, comp_now=True, comp_then=True, color='red', ax=ax)
            
            mask = lnols>lnols_limit_for_plotting
            
            # Stars with and without RV
            ax.scatter(star_means[:, dim1][mask], star_means[:, dim2][mask], c='k', s=1)
            
            # Only stars with RV
            mask_rv = data_table['radial_velocity_error']<1000 # best_rv_cartesian_with_rvs_renamed_to_radial_velocity.fits: best_rv_err has been renamed to radial_velocity_error
            ax.scatter(star_means[:, dim1][mask&mask_rv], star_means[:, dim2][mask&mask_rv], c='r', s=1)
            

plt.show()

# Save results
# lnols
tab.write(output_filename, format='fits', overwrite=True)
print('%s written.'%output_filename)

# Store components
best_fit_comps[0].store_raw('bpic_init_comp_lsr.npy')
print('Init comp saved.')

# Store data with lnols above the limit
mask = lnols>-33
data_table = data_table[mask]
data_table.write('banyan_bpic_LSR.fits')
print('Member stars saved. NOTE: you need to add background overlaps to this table.')
