"""
Distances for components.
"""

import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
exclude_components = lib.exclude_components
############################################

# Minimal probability required for membership
pmin_membership = 0.9
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


print('Shifting l by 100 to put ScoCen in the middle')
lshift=100
mask = np.where(tab['l']<lshift)
tab['l'][mask] = 360 + tab['l'][mask]

print('Plotting %d components.'%len(comps))

fig = plt.figure(figsize=(figsize[1], figsize[0]))
ax = fig.add_subplot(121)
axC = fig.add_subplot(122)
axE = axC.twinx()

test = plt.figure()
axt = test.add_subplot(111)

# Number of bins
#~ b = {'A': 20, 'C': 25, 'D': 20, 'E': 40, 'F': 20, 'G': 20, 'I': 20, 'T': 20, 'U': 20}
#~ b = {'A': 20, 'C': 50, 'D': 30, 'E': 100, 'F': 30, 'G': 20, 'T': 50, 'U': 25}
b = {'A': 20, 'C': 100, 'D': 30, 'E': 100, 'F': 30, 'G': 20, 'T': 50, 'U': 25, 'B':500}

#~ usco=Table()

for i, c in enumerate(comps):
    comp_ID = c['comp_ID']
    
    if comp_ID not in ['A', 'C', 'D', 'E', 'F', 'G', 'T', 'U']: # 'B'
    #~ if comp_ID not in ['A', 'D', 'F', 'G', 'T', 'U']:
        continue
    
    mask = tab['membership%s'%comp_ID] > pmin_membership
    t=tab[mask]

    # Sich distances are not meaningful because distributions are complex
    #~ median_distance = np.nanmedian(t['dist'])
    #~ ax.axvline(x=median_distance, c=colors[comp_ID], linewidth=0.5, linestyle='--')
    
    
    #~ bins=np.min([int(float(len(t))/50.0), 25])
    bins = b[comp_ID]
    
    if comp_ID=='B':
        lnC=axC.hist(t['dist'], bins=500, histtype='step', normed=False, color=colors[comp_ID], label=comp_ID)
    
    elif comp_ID=='C':
        mask = tab['membership%s'%comp_ID] > 0.8
        t=tab[mask]

        lnC=axC.hist(t['dist'], bins=np.linspace(120, 180, 50), histtype='step', normed=False, color=colors[comp_ID], label=comp_ID)
        
        #~ cb=axt.scatter(t['l'], t['b'], c=t['dist'], s=1, vmin=125, vmax=150)
        
        #~ m = (t['dist']>134) & (t['dist']<137)
        #~ axt.scatter(t['l'][m], t['b'][m], c='r', s=2, vmin=125, vmax=150)
        
        #~ m = (t['dist']>142) & (t['dist']<145)
        #~ axt.scatter(t['l'][m], t['b'][m], c='cyan', s=2, vmin=125, vmax=150)
        
        
    elif comp_ID=='E':
        mask = tab['membership%s'%comp_ID] > 0.8
        t=tab[mask]
        lnE=axE.hist(t['dist'], bins=np.linspace(120, 180, 50), histtype='step', normed=False, color=colors[comp_ID], label=comp_ID)
        
        cb=axt.scatter(t['l'], t['b'], c=t['dist'], s=1, vmin=125, vmax=150)
        
        m = (t['dist']>139) & (t['dist']<142)
        axt.scatter(t['l'][m], t['b'][m], c='r', s=2, vmin=125, vmax=150)
        
        m = (t['dist']>127) & (t['dist']<130)
        axt.scatter(t['l'][m], t['b'][m], c='cyan', s=2, vmin=125, vmax=150)
        
    else:
        ax.hist(t['dist'], bins=bins, histtype='step', normed=False, color=colors[comp_ID], label=comp_ID)
    #~ ax.hist(t['dist'], bins=bins, histtype='step', normed=False, color=colors[comp_ID])
        
    print(comp_ID, np.nanmedian(t['dist']), colors[comp_ID], bins, len(t))

    #~ if comp_ID in ['C', 'E', 'G']:
        #~ usco = vstack((usco, t))

#~ cbar=test.colorbar(cb)
def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    #~ ax.set_ylim(-40, 60)
    ax.set_ylim(-20, 40)
    #~ ax.set_xlim(400, 220)
    ax.set_xlim(400, 340)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    t = ax.get_xticks()
    xtick_labels = [int(x) if x<360 else int(x-360) for x in t]
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('l [deg]')
    ax.set_ylabel('b [deg]')
    ax.set_aspect('equal')

    return ax

gx_set_labels_and_ticks_over_360deg(axt)

# USCO
#~ mask = (usco['l']>342) & (usco['b']>10) & (usco['b']<30)
#~ axC.hist(usco[mask]['dist'], color='k', histtype='step',  bins=50)

#~ ax.set_yscale('log')
#~ axCE.set_yscale('log')

ax.set_ylim(0, 100)
ax.set_xlabel('Distance [pc]')
ax.set_ylabel('Frequency')
ax.legend(frameon=False, title='p$>$0.9')

axC.set_xlabel('Distance [pc]')
axC.set_ylabel('Frequency')
axC.yaxis.label.set_color('orange')
axC.tick_params(axis='y', colors='orange')
axE.yaxis.label.set_color('purple')
axE.tick_params(axis='y', colors='purple')

# LEGEND
handlesC, labelsC = axC.get_legend_handles_labels()
handlesE, labelsE = axE.get_legend_handles_labels()
labels = [labelsC[0], labelsE[0]]
handles = [handlesC[0], handlesE[0]]
legend=axC.legend(handles, labels, frameon=False, title='p$>$0.8')

plt.tight_layout()

fig.savefig('distance_distributions_components.pdf')

plt.show()
