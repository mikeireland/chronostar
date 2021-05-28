"""
Plot ScoCen in (l, b) coordinates, stars for each component with different
colours.
"""

import numpy as np
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
exclude_components = lib.exclude_components
exclude_components.append('Q')
exclude_components.append('B')
exclude_components.append('J')
compnames = lib.compnames
colors = lib.colors
############################################

# Minimal probability required for membership
pmin_membership = 0.5
############################################

# Read data
try:
    tab = tab0
    comps = comps0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0

# Shift data in galactic 'l' to put ScoCen in the middle of the plot
print('Shifting l by 100 to put ScoCen in the middle')
lshift=100
mask = np.where(tab['l']<lshift)
tab['l'][mask] = 360 + tab['l'][mask]

# Sort components by their number of members. Plot the biggest first.
indices = np.argsort([np.sum(tab['membership%s'%c['comp_ID']]>pmin_membership ) for c in comps])
comps = comps[indices]
comps = comps[::-1]



# Take only stars with RVs
#~ mask = tab['radial_velocity_error']<100
#~ tab=tab[mask]
print(len(tab))


#### PLOTTING ###############################
fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax=fig.add_subplot(111)

total=0 # Total number of stars in the plot

# Plot components (stars) one by one with different colours
for c in comps:
    comp_id = c['comp_ID']
    if comp_id in exclude_components:
        continue

    # Take only members of this component
    mask = tab['membership%s'%comp_id] > pmin_membership 
    t=tab[mask]
    
    #~ print(comp_id, colors[comp_id])
    
    # PLOT STARS
    name_literature = '' # Component name from the literature, e.g. rho Oph
    try:
        name_literature = '\n%s'%compnames[comp_id]
    except:
        pass
    
    age=c['Age']
    label = r'%s (%d), %.1f$\pm$%.1f Myr, %s'%(comp_id, len(t), age, c['Crossing_time'], name_literature)
    
    cb=ax.scatter(t['l'], t['b'], s=1, c=colors[comp_id], marker='.', label=label)
    
    total+=len(t)
    
print('Total number of stars in this plot:', total)

total=0
# Plot BJQ components
fig2=plt.figure(figsize=(figsize[1], figsize[0]))
ax2=fig2.add_subplot(111)

for c in comps:
    comp_id = c['comp_ID']
    if comp_id not in ['B', 'J', 'Q']:
        continue

    # Take only members of this component
    mask = tab['membership%s'%comp_id] > pmin_membership 
    t=tab[mask]
    
    #~ print(comp_id, colors[comp_id])
    
    # PLOT STARS
    name_literature = '' # Component name from the literature, e.g. rho Oph
    try:
        name_literature = '\n%s'%compnames[comp_id]
    except:
        pass
    
    age=c['Age']
    label = r'%s (%d), %.1f$\pm$%.1f Myr, %s'%(comp_id, len(t), age, c['Crossing_time'], name_literature)
    
    #~ cb=ax.scatter(t['l'], t['b'], s=1, c=colors[comp_id], marker='.', label=label)
    ax2.scatter(t['l'], t['b'], s=1, c='k', marker='.', label=label)
    
    total+=len(t)
    
print('Total number of stars in this plot:', total)

def plot_3_windows_gx(ax, labels=True, lw=2, ls='-', c='b'):
    """
    Plot lines designating USco, UCL, LCC
    """
    print('WINDOWS plotting...')
    
    # USco
    ax.plot([360, 360], [10, 30], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 342], [10, 30], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 360], [10, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 360], [30, 30], c=c, linestyle=ls, linewidth=lw)
    
    # UCL
    ax.plot([350, 350], [0, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 312], [0, 25], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 350], [0, 0], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 312], [25, 25], c=c, linestyle=ls, linewidth=lw)
    
    # LCC
    ax.plot([312, 312], [-10, 23], c=c, linestyle=ls, linewidth=lw)
    ax.plot([285, 285], [-10, 23], c=c, linestyle=ls, linewidth=lw)
    ax.plot([285, 312], [-10, -10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([285, 312], [23, 23], c=c, linestyle=ls, linewidth=lw)


    # Lupus complex (Hara et al. 1999)
    # Lupus 1
    337-340, 14-18
    ax.plot([337, 340], [14, 14], c=c, linestyle=ls, linewidth=lw)
    ax.plot([337, 340], [18, 18], c=c, linestyle=ls, linewidth=lw)
    ax.plot([337, 337], [14, 18], c=c, linestyle=ls, linewidth=lw)
    ax.plot([340, 340], [14, 18], c=c, linestyle=ls, linewidth=lw)

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
    PL9 = [137.32131607, -41.94779643] # (l, b)
    ax.scatter(PL9[0], PL9[1], c=c, s=10)
    if labels:
        ax.annotate('Platais 8',
                xy=(137, -42), xycoords='data',
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


    return ax

def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    #~ ax.set_ylim(-40, 60)
    ax.set_ylim(-30, 40)
    #~ ax.set_xlim(400, 220)
    ax.set_xlim(380, 260)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    ax.set_aspect('equal')

    return ax

def manage_legend(ax):
    # ADD LEGEND OUTSIDE THE PLOT
    #~ fig.subplots_adjust(bottom=0.3, top=0.95)

    #~ # Put a legend below current axis
    #~ ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False,
              #~ fancybox=False, shadow=False, ncol=3, markerscale=10)
              
    
    fig.subplots_adjust(left=-0.05, bottom=0.15, top=0.9)
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.05), frameon=False,
              fancybox=False, shadow=False, ncol=1, markerscale=10)


# PDS 70
#~ ax.scatter([318.0621220374871], [+19.2084968169996], c='r', s=10)

# Overplot young part of the component T
#~ tabT = Table.read('component_T_younger_subcomponent.fits')
#~ print('Shifting l by 100 to put ScoCen in the middle')
#~ lshift=100
#~ mask = np.where(tabT['l']<lshift)
#~ tabT['l'][mask] = 360 + tabT['l'][mask]
#~ ax.scatter(tabT['l'], tabT['b'], s=1, c='r', marker='.')


lib.plot_3_windows_gx(ax, labels=True, lw=1, ls='-', c='r')
gx_set_labels_and_ticks_over_360deg(ax)
manage_legend(ax)


plot_3_windows_gx(ax2, labels=True, lw=1, ls='-', c='r')
gx_set_labels_and_ticks_over_360deg(ax2)

# SAVE FIGURES
#~ plt.savefig('gx.pdf')
plt.show()
