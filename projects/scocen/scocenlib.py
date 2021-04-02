"""
A config file: what components are well defined, what are background,
what colors are associated with what components etc.
"""

# Filename with all the data (astrometry, Cartesian, photometry, lithium etc.)
# This file should contain overlaps with the components
#~ data_filename = 'data/gaia_200_pc_for_scocen_with_overlaps.fits'
#~ data_filename = 'data/scocen_vac_EDR3_SUBTABLE_with_bgols_need_to_update_bg_ols_with_21_overlaps.fits'

#~ data_filename = 'data/scocen_vac_EDR3_with_21_overlaps_TODO_update_bgols_with_new_rv.fits' # This one should have RVs fixed
data_filename = 'data/scocen_vac_DR2_with_21_overlaps.fits' # This one should have RVs fixed

# Data that was used for the fitting
data_filename_fit = 'data/scocen_5k_candidates_only_members_with_significant_membership_from_tims_3_subsets.fits'

# Fits file (the same content as npy file)
comps_filename = 'data/final_comps_21.fits'

# Healthy components (only PMS sequence)
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U', 'B', 'Q', 'J']

# Components with both MS and PMS sequences. Need further splitting!
comps_multiple_pop = ['J', 'B', 'Q']

# Diffuse components with MS stars and thousands of members
bg_comps = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S']

# When plotting, exclude these components
exclude_components = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S']

# Colors only for good_comps. I don't know so many colors that I could assign them to bad comps as well.
colors = {'A': 'g', 'C': 'orange', 'D': 'yellow', 'E': 'purple', 'F': 'brown', 'G': 'b', 'H': 'magenta', 'I': 'cyan', 'T': 'k', 'U': 'r', 'B': 'grey', 'Q': 'olive', 'J': 'pink'}

# Comparison with the literature
compnames = {'C': r'$\rho$Oph', 'D': 'V1062 Sco', 'B': 'CrA', 'H': 'IC2602', 'T': 'incl. PDS70', 'Q': r'incl. IC2391, $\epsilon$Cha', 'I': 'Platais 8'}


