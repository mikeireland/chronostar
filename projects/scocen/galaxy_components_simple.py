import numpy as np
from astropy.table import Table, join, unique
from astropy import units as u
from astropy.coordinates import SkyCoord # This is slow but used only for components
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.ticker as ticker
plt.ion()
import random

#~ import scocenlib2 as lib
from fig_settings import *

import sys
sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
import coordinate

# Minimal probability required for membership
pmin_membership = 0.5
################################

# Data
#~ d=lib.d

# Read components (WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present!)
#~ comps = lib.comps

tab = Table.read('members_from_5k_fit_and_overlaps.fits')

comps = Table.read('final_comps.fits')
comps.rename_column('age', 'Age')
# Crossing time. Only components with sigma<age have reliable ages.
crossing_time = comps['dX']/comps['dV'] * 0.977813106 # pc/km*s to Myr
comps['Crossing_time'] = crossing_time
mask = crossing_time < comps['Age'] # sigma < age
comps['Age_reliable'] = mask



exclude_components = []
include_components = []

fig=plt.figure()
ax=fig.add_subplot(111)


total=0
for c in comps:
    comp_id = c['Name']
    #~ if comp_id in exclude_components:
        #~ continue
    
    #~ if len(include_components)>0:
        #~ if comp_id not in include_components:
            #~ continue
    age=c['Age']
    mask=tab['membership%s'%comp_id]>pmin_membership 
    t=tab[mask]
    print(c['Name'], len(t), np.std(t['b']))
    total+=len(t)
    
    if c['Age_reliable']:
        color=np.ones(len(t))*age
    else:
        color='k'
    #~ color='k'
    #~ color = (random.random(), random.random(), random.random())
    
    ax.annotate(comp_id,
        xy=(np.nanmedian(t['l']), np.nanmedian(t['b'])), xycoords='data',
        xytext=(0, 1), textcoords='offset points', color='r', fontsize=20)
    
    
    cb=ax.scatter(t['l'], t['b'], s=1, c=color, vmin=0, vmax=15, alpha=1)

try:
    cbar=plt.colorbar(cb)
    cbar.ax.set_ylabel('Age [Myr]')
except:
    print('Colorbar not plotted.')


def gx_set_labels_and_ticks_over_360deg(ax):
    """
    When plotting ScoCen data goes >360deg. I set these numbers to start
    with 0 manually.
    """
    
    #~ plt.gca().invert_xaxis()
    ax.set_ylim(-40, 60)
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


gx_set_labels_and_ticks_over_360deg(ax)


plt.show()
