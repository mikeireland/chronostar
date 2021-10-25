"""
A config file: what components are well defined, what are background,
what colors are associated with what components etc.
"""

# Filename with all the data (astrometry, Cartesian, photometry, lithium etc.)
# This file should contain overlaps with the components
#~ data_filename = 'data/gaia_200_pc_for_scocen_with_overlaps.fits'
#~ data_filename = 'data/scocen_vac_EDR3_SUBTABLE_with_bgols_need_to_update_bg_ols_with_21_overlaps.fits'

#~ data_filename = 'data/scocen_vac_EDR3_with_21_overlaps_TODO_update_bgols_with_new_rv.fits' # This one should have RVs fixed
#~ data_filename = 'data/scocen_vac_DR2_with_21_overlaps.fits' # This one should have RVs fixed
#~ data_filename = 'data/scocen_vac_DR2_with_21_overlaps_may2021_test2.fits' # This one should have RVs fixed. Works fine...

#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components.fits' # This one should have RVs fixed
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_reddening_correction.fits' # This one should have RVs fixed # THIS ONE IS USED IN THE PAPER
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_reddening_correction2.fits' # This one should have RVs fixed. This was reconstructed after I accidentally saved only a table with 10k scocen members. But I want to have all 800k stars
#~ data_filename = 'data/test.fits' # This one should have RVs fixed. This was reconstructed after I accidentally saved only a table with 10k scocen members. But I want to have all 800k stars
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_iterative_color_depedent_reddening_correction.fits' # This one should have RVs fixed. Reddening is computed with a revised Adam's code in an iterative processs! This is the GOOD version.
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_iterative_color_depedent_reddening_correction_Ref.fits' # This one should have RVs fixed. Reddening is computed with a revised Adam's code in an iterative processs! This is the GOOD version. Added references for the paper.
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_iterative_color_depedent_reddening_correction_Ref_RizzutoLithium.fits' # This one should have RVs fixed. Reddening is computed with a revised Adam's code in an iterative processs! This is the GOOD version. Added references for the paper, and Rizzuto lithium


# New radial velocities: merged repeats from RAVE, 2.3m and APOGEE into one single value.
# Lithium added manually, including Rizzuto EW(Li)
# New background overlaps, except: Those beyond 200pc are mostly missing and are set to -99999. Since we don't really have many members beyond 200pc I decided that it's ok and leave it.
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_li_XYZUVW_overlaps_with_21_components.fits'
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_li_XYZUVW_overlaps_with_21_components_and_reddening_correction.fits'
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges_li_XYZUVW_overlaps_with_21_components_and_reddening_correction.fits'
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges_li_XYZUVW_overlaps_with_21_components.fits'
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges2_li_XYZUVW_within200pc_overlaps_with_21_components.fits'
#~ data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges2_li_XYZUVW_within200pc_overlaps_with_21_components_and_reddening_correction.fits'
data_filename = 'data/scocen_gaiadr2_for_the_paper_rv_no_ges3_li_XYZUVW_within200pc_overlaps_with_21_components_and_reddening_correction.fits'

#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_components_with_cra.fits' # This one should have RVs fixed. Added CrA
#~ data_filename = 'data/scocen_vac_DR2_distinct_overlaps_with_21_plus_additional_comps.fits'

# Data that was used for the fitting
data_filename_fit = 'data/scocen_5k_candidates_only_members_with_significant_membership_from_tims_3_subsets.fits'

# Fits file (the same content as npy file)
comps_filename = 'data/final_comps_21.fits'
#~ comps_filename = 'data/final_comps_21_with_cra.fits'
#~ comps_filename = 'data/final_comps_21_additional_comps.fits'

# Healthy components (only PMS sequence)
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U', 'B', 'Q', 'J', 'V', 'W']

# Components with both MS and PMS sequences. Need further splitting!
comps_multiple_pop = ['J', 'B', 'Q']

# Diffuse components with MS stars and thousands of members
bg_comps = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S']

# When plotting, exclude these components
exclude_components = ['K', 'L', 'M', 'N', 'O', 'P', 'R', 'S']

# Colors only for good_comps. I don't know so many colors that I could assign them to bad comps as well.
colors = {'A': 'g', 'C': 'orange', 'D': 'yellow', 'E': 'purple', 'F': 'lime', 'G': 'b', 'H': 'magenta', 'I': 'cyan', 'T': 'k', 'U': 'r', 'B': 'grey', 'Q': 'olive', 'J': 'pink', 'V': 'teal', 'W': 'violet', 'X': 'k', 'Y': 'orange', 'Z': 'green', '1': 'blue', '2': 'red', '3': 'cyan', '4': 'orange', '5': 'k', '6': 'brown'}

# Comparison with the literature
compnames = {'C': r'$\rho$Oph', 'D': 'V1062 Sco', 'F': 'V1062 Sco', 'B': 'CrA', 'H': 'IC2602', 'T': 'UCL, incl. PDS70', 'Q': r'incl. IC2391, $\epsilon$Cha', 'I': 'Platais 8', 'G': 'UCL', 'A': 'LCC', 'U': 'LCC', 'E': 'CrA'}


def plot_3_windows_gx(ax, labels=True, lw=2, ls='-', c='b'):
    """
    Plot lines designating USco, UCL, LCC
    """
    print('WINDOWS plotting...')
    
    def plot_window(ax, x1=None, x2=None, y1=None, y2=None, c=None, ls=None, lw=None):
        ax.plot([x1, x1], [y1, y2], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x2, x2], [y1, y2], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x1, x2], [y1, y1], c=c, linestyle=ls, linewidth=lw)
        ax.plot([x1, x2], [y2, y2], c=c, linestyle=ls, linewidth=lw)
    
    # USco
    plot_window(ax, x1=342, x2=360, y1=10, y2=30, c=c, ls=ls, lw=lw)
    
    # UCL
    ax.plot([350, 350], [0, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 312], [0, 25], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 350], [0, 0], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 312], [25, 25], c=c, linestyle=ls, linewidth=lw)
    
    # LCC
    plot_window(ax, x1=312, x2=285, y1=-10, y2=23, c=c, ls=ls, lw=lw)

    # Lupus complex (Hara et al. 1999)
    # Plot with windows from https://www.eso.org/~fcomeron/lupus_rev.pdf
    ls = '--'
    # Lupus 1
    plot_window(ax, x1=337, x2=340, y1=14, y2=18, c=c, ls=ls, lw=lw)

    # Lupus 2
    plot_window(ax, x1=338, x2=340, y1=11, y2=13, c=c, ls=ls, lw=lw)

    # Lupus 3
    plot_window(ax, x1=338, x2=341, y1=8, y2=10, c=c, ls=ls, lw=lw)

    # Lupus 4
    plot_window(ax, x1=338, x2=335, y1=6, y2=8, c=c, ls=ls, lw=lw)

    # Lupus 5
    plot_window(ax, x1=344, x2=341, y1=7, y2=10, c=c, ls=ls, lw=lw)

    # Lupus 6
    plot_window(ax, x1=340, x2=342, y1=7, y2=5, c=c, ls=ls, lw=lw)

    # Lupus 7
    plot_window(ax, x1=336.5, x2=337.5, y1=4.5, y2=5.5, c=c, ls=ls, lw=lw)

    # Lupus 8
    plot_window(ax, x1=336, x2=334, y1=4.5, y2=3, c=c, ls=ls, lw=lw)

    # Lupus 9
    plot_window(ax, x1=346.5, x2=347.5, y1=6, y2=7, c=c, ls=ls, lw=lw)


    # Damiani et al. 2019. UCL-1 is V1092 Sco
    # UCL-2
    plot_window(ax, x1=316, x2=322, y1=11, y2=15, c=c, ls=ls, lw=lw)
    
    # LCC-1
    plot_window(ax, x1=297, x2=303, y1=-4, y2=0, c=c, ls=ls, lw=lw)



    # IC2602
    ax.scatter([289.6014], [-04.9061], c=c, s=10)
    if labels:
        ax.annotate('IC2602',
                xy=(289.6014, -04.9061), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # V1062 Sco moving group (a newly discovered MG in ScoCen by Roser et al. 2018)
    # (X, Y, Z, U, V, W) = (167.20, -49.14, 13.44, -3.80, -19.96, -4.06). from Roser et al. 2018
    ax.scatter([343.6], [4.3], c=c, s=10)
    if labels:
        ax.annotate('V1062 Sco',
                xy=(343.6, 4.3), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Corona Australis
    CRA = [359.74400822, -17.51551102] # (l, b)
    ax.scatter(CRA[0], CRA[1], c=c, s=10)
    if labels:
        ax.annotate('CrA',
                xy=(359.7, -17.5), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Esplin et al. 2018: rho Oph location (they are separating between rho Oph and USco)
    # Rho Ophiuci
    ROPH = [353.22097900, 16.53342332] # (l, b)
    ax.scatter(ROPH[0], ROPH[1], c=c, s=10)
    if labels:
        ax.annotate(r'$\rho$ Oph',
                xy=(353, 16), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # IC2391
    IC2391 = [270.36829815, -6.83062731] # (l, b)
    ax.scatter(IC2391[0], IC2391[1], c=c, s=10)
    if labels:
        ax.annotate('IC2391',
                xy=(270, -7), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Platais 8
    PL8 = [277.6824, -07.6209] # (l, b)
    ax.scatter(PL8[0], PL8[1], c=c, s=10)
    if labels:
        ax.annotate('Platais 8',
                xy=(277, -7), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Platais 9
    PL9 = [270, 5] # (l, b)
    ax.scatter(PL9[0], PL9[1], c=c, s=10)
    if labels:
        ax.annotate('Platais 9',
                xy=(270, 5), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # eps Chamaeleontis
    EPSC = [300.20873944, -15.62481300] # (l, b)
    ax.scatter(EPSC[0], EPSC[1], c=c, s=10)
    if labels:
        ax.annotate(r'$\epsilon$ Cha',
                xy=(300, -15), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # eta Chamaeleontis
    ETAC = [292.40233238, -21.65095171] # (l, b)
    ax.scatter(ETAC[0], ETAC[1], c=c, s=10)
    if labels:
        ax.annotate(r'$\eta$ Cha',
                xy=(292, -21), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)


    # Unknown thing in Lower Scorpius
    # Equatorial coordinates: to search in known catalogs
    # 242.03367512      -38.32856212      316.855146   
    # 16h08m08.08203s   -38d19m42.8236s
    thing = [340, 10] # (l, b)
    ax.scatter(thing[0], thing[1], c=c, s=10)
    if labels:
        ax.annotate(r'thing',
                xy=(340, 10), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)



