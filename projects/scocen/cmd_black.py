"""
Plot CMDs for each component.
"""

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import copy
from random import shuffle
plt.ion()

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
exclude_components = copy.copy(lib.exclude_components)
exclude_components.append('J')
exclude_components.append('B')
exclude_components.append('Q')
############################################

# Minimal probability required for membership
#~ pmin_membership = 0.5
pmin_membership = 0.8
print('pmin_membership', pmin_membership)
############################################
# CMD limits
xlim = [-1, 5]
ylim = [17, -3]
############################################

# Read data
try:
    tab = tab0
    comps = comps0
except:
    tab0 = Table.read(data_filename)
    
    Gmag = tab0['phot_g_mean_mag'] - 5 * np.log10(1.0 / (tab0['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
    tab0['Gmag'] = Gmag
    
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0


# Main sequence parametrization
# fitpar for pmag, rpmag
fitpar = [0.17954163, -2.48748376, 12.9279348, -31.35434182, 38.31330583, -12.25864507]
poly = np.poly1d(fitpar)
x = np.linspace(1, 4, 100)
y = poly(x)
m = y > 4
yms = y[m]
xms = x[m]

def plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim):
    ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
    #~ ax.plot(xms, yms - 0.75, c='brown', label='0.75 mag above the median', linewidth=1, linestyle='--')
    #~ ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

    ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
    ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
    ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
    ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
        
    return ax

print('Plotting %d components.'%len(comps))


# Sort components by their number of members. Plot the biggest first.
indices = np.argsort([np.sum(tab['membership%s'%c['comp_ID']]>pmin_membership ) for c in comps])
comps = comps[indices]
comps = comps[::-1]

fig=plt.figure()
ax = fig.add_subplot(111)

# In case I want colors after all
norm=plt.Normalize(vmin=0, vmax=len(comps)-len(exclude_components))
#~ mycmap = plt.get_cmap('gist_rainbow')
mycmap = plt.get_cmap('jet')
colors = [mycmap(norm(i)) for i in range(len(comps))]
shuffle(colors) # So that neighbouring components don't have similar colors

i=0
for c in comps:
    comp_ID = c['comp_ID']

    if comp_ID in exclude_components:
        continue

    # Take only members
    mask = tab['membership%s'%comp_ID] > pmin_membership
    t=tab[mask]

    


    
    

    # There are two overlapping sequences that are older than the rest
    # (even older than T).
    # One is H and the other one is I.
    if comp_ID=='H': # IC2602
        clr='r'
    elif comp_ID=='I': # Platais 8
        clr='cyan'
    elif comp_ID=='C':
        clr='lime'
    #~ else:
        #~ continue


    # D is V1062 Sco, probably bound?
    #~ if comp_ID in ['H', 'I', 'C', 'U', 'T', 'A', 'G', 'D']:
        #~ continue
    #~ if comp_ID not in ['U', 'T', 'C']:
        #~ continue

    #~ elif comp_ID=='E':
        #~ clr='magenta'


    clr='k'
    #~ clr=colors[i]

    print(comp_ID, c['Age'], clr)

    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c=clr, alpha=1)


    i+=1

ax=plot_MS_parametrisation_and_spectral_types(ax, xlim, ylim)

ax.set_xlabel('BP-RP')
ax.set_ylabel('Gmag')

plt.show()
