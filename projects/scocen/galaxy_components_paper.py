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
add_labels = True # Add component labels in the plot
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
#~ fig=plt.figure(figsize=(figsize[1], figsize[0]))
fig=plt.figure(figsize=(figsize[1]*0.8, figsize[0]))
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
    #~ label = r'%s (%d), %.1f$\pm$%.1f Myr, %s'%(comp_id, len(t), age, c['Crossing_time'], name_literature)
    label = r'%s (%d)'%(comp_id, len(t))
    
    if comp_id=='D':
        ax.scatter(t['l'], t['b'], s=30, c=colors[comp_id], edgecolor='k', linewidth=0.2, marker='.', label=label)
    else:
        ax.scatter(t['l'], t['b'], s=1, c=colors[comp_id], marker='.', label=label, alpha=0.6)
    
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

def manage_legend(ax):
    # ADD LEGEND OUTSIDE THE PLOT
    #~ fig.subplots_adjust(bottom=0.3, top=0.95)

    #~ # Put a legend below current axis
    #~ ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False,
              #~ fancybox=False, shadow=False, ncol=3, markerscale=10)
              
    
    fig.subplots_adjust(left=0.05, bottom=0.15, top=0.9)
    #~ fig.subplots_adjust(bottom=0.15, top=0.9)
    #~ fig.subplots_adjust(bottom=0.15, top=0.9)
    # Put a legend below current axis
    #~ ax.legend(loc='upper center', bbox_to_anchor=(1.18, 1.05), frameon=False,
    legend = ax.legend(loc='upper center', bbox_to_anchor=(1.09, 1.03), frameon=False,
              fancybox=False, shadow=False, ncol=1, markerscale=10)
    legend.legendHandles[7]._sizes = [100]

def add_labels_to_the_plot():
    """
    Add component labels to the plot
    """
    
    fs=12 # fontsize
    c='k'
    
    ax.annotate('A', xy=(295, 15), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('U', xy=(310, -9), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('C', xy=(350, 26), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    #~ ax.annotate('E', xy=(367.5, -6.8), 
    ax.annotate('E', xy=(376, -16), 
    #~ ax.annotate('E', xy=(370.5, 5.7), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('G', xy=(340, 26), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('H', xy=(290, -15), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('I', xy=(275, 5), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('T', xy=(310, 24), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    #~ ax.annotate('D', xy=(345, 3), 
    ax.annotate('D', xy=(341, -15), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  

    w=0.05
    lw=0.3
    hw=1.5
    hl=2
    ax.arrow(340, -10, 3, 11, width=w, lw=lw, head_width=hw, 
        head_length=hl, color='k')

    #~ ax.annotate('F', xy=(340, -4), 
    ax.annotate('F', xy=(334, -8), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  

    ax.arrow(335, -5, 2.3, 6, width=w, lw=lw, head_width=hw, 
        head_length=hl, color='k')

plot_3_windows_gx(ax, labels=False, lw=0.5, ls='-', c='k')
gx_set_labels_and_ticks_over_360deg(ax)
manage_legend(ax)

if add_labels:
    add_labels_to_the_plot()


# SAVE FIGURES
plt.savefig('gx_components_with_labels.pdf')

plt.show()
