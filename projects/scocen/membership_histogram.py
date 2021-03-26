"""
Plot CMDs for different kinematic ages and see if there is a sequence
i.e. if older components lie below the younger ones.
"""

import numpy as np
from astropy.table import Table
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()

from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
good_comps = lib.good_comps
colors = lib.colors
############################################
# Minimal probability required for membership
pmin_membership = 0.5
############################################


# Remove JBQ from the list because they mostly contain MS stars
good_comps.remove('J')
good_comps.remove('B')
good_comps.remove('Q')


# Read data
try:
    tab = tab0
    comps = comps0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)

    tab = tab0
    comps = comps0



fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax = fig.add_subplot(111)

for c in comps:
    comp_ID = c['comp_ID']       
    
    if comp_ID not in good_comps:
        continue
     
    membname = 'membership%s'%comp_ID
    
    mask=tab[membname]>0.05
    t=tab[mask]

    ax.hist(t[membname], color=colors[comp_ID], label=r'%s (%d)'%(comp_ID, len(t)), bins=int(len(t)/20.0), histtype='step')




#~ ax.legend(markerscale=5, frameon=False) # bbox_to_anchor=(1, 0.7), loc='center right', 
ax.legend(loc='upper center', frameon=False,
              fancybox=False, shadow=False, ncol=2, markerscale=10) #bbox_to_anchor=(1.18, 1.05), 
#~ fig.subplots_adjust(bottom=0.15, top=0.9)

ax.set_yscale('log')

ax.set_xlabel('Membership probability')
ax.set_ylabel('N')

#~ ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
#~ ax.yaxis.set_major_locator(ticker.MultipleLocator(2))


plt.savefig('memb_hist.pdf')
plt.show()
