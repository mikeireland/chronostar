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
compnames = lib.compnames
colors = lib.colors
############################################

# Minimal probability required for membership
pmin_membership = 0.5
############################################
print('pmin membership', pmin_membership)

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
#~ fig=plt.figure(figsize=(figsize[1], figsize[0]))
fig=plt.figure(figsize=(figsize[1]*0.8, figsize[0]))
ax=fig.add_subplot(111)

total=0 # Total number of stars in the plot

# Plot components (stars) one by one with different colours
for c in comps:
    comp_id = c['comp_ID']
    if comp_id not in ['J', 'B', 'Q']:
        continue

    # Take only members of this component
    mask = tab['membership%s'%comp_id] > pmin_membership 
    t=tab[mask]
    
    # PLOT STARS
    label = r'%s (%d)'%(comp_id, len(t))
    
    cb=ax.scatter(t['l'], t['b'], s=1, c='grey', marker='.', alpha=0.2)
    
    total+=len(t)
    
print('Total number of stars in this plot:', total)


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
    if labels:
        ax.annotate('USCO',
                xy=(356, 32), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)
    
    # UCL
    ax.plot([350, 350], [0, 10], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 312], [0, 25], c=c, linestyle=ls, linewidth=lw)
    ax.plot([312, 350], [0, 0], c=c, linestyle=ls, linewidth=lw)
    ax.plot([342, 312], [25, 25], c=c, linestyle=ls, linewidth=lw)
    
    # LCC
    plot_window(ax, x1=312, x2=285, y1=-10, y2=23, c=c, ls=ls, lw=lw)


    # Unknown cluster on the left
    if labels:
        ax.annotate('overdensity',
                xy=(378, 7), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)    

    # Corona Australis
    CRA = [359.74400822, -17.51551102] # (l, b)
    #~ ax.scatter(CRA[0], CRA[1], c=c, s=10)
    if labels:
        ax.annotate('CrA',
                xy=(362, -23), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)


    # IC2391
    IC2391 = [270.36829815, -6.83062731] # (l, b)
    #~ ax.scatter(IC2391[0], IC2391[1], c=c, s=10)
    if labels:
        ax.annotate('IC 2391',
                xy=(275, -15), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # Platais 8
    #~ PL8 = [277.6824, -07.6209] # (l, b)
    #~ ax.scatter(PL8[0], PL8[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('Platais 8',
                #~ xy=(277, -7), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # Platais 9
    #~ PL9 = [270, 5] # (l, b)
    #~ ax.scatter(PL9[0], PL9[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate('Platais 9',
                #~ xy=(270, 5), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    # eps Chamaeleontis
    EPSC = [300.20873944, -15.62481300] # (l, b)
    #~ ax.scatter(EPSC[0], EPSC[1], c=c, s=10)
    if labels:
        ax.annotate(r'$\epsilon$ Cha',
                xy=(300, -21), xycoords='data',
                xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)

    #~ # eta Chamaeleontis
    #~ ETAC = [292.40233238, -21.65095171] # (l, b)
    #~ ax.scatter(ETAC[0], ETAC[1], c=c, s=10)
    #~ if labels:
        #~ ax.annotate(r'$\eta$ Cha',
                #~ xy=(292, -21), xycoords='data',
                #~ xytext=(0, 1), textcoords='offset points', color=c, fontsize=12)


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
    ax.set_xlim(380, 260) #
    #~ ax.set_xlim(400, 260)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    ax.set_aspect('equal')

    return ax


plot_3_windows_gx(ax, labels=True, lw=0.5, ls='-', c='k')
gx_set_labels_and_ticks_over_360deg(ax)

plt.tight_layout()

# SAVE FIGURES
#~ plt.savefig('gx_components_JBQ.pdf')

ax.set_rasterized(True)
plt.savefig('gx_components_JBQ_rasterized.pdf')

plt.show()
