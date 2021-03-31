"""
XU for stars in the fit (stars with RV).
"""

import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import copy
plt.ion()

#~ import sys
#~ sys.path.insert(0, '/Users/marusa/chronostar/chronostar/')
from chronostar.component import SphereComponent
from chronostar import tabletool

# Pretty plots
from fig_settings import *

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename_fit # Use only data with RV that was actually used in the fit
comps_filename = lib.comps_filename
good_comps = copy.copy(lib.good_comps)
compnames = lib.compnames
colors = lib.colors
############################################
# Minimal probability required for membership
pmin_membership = 0.9
pmin_membership = 0.5
print('pmin_membership', pmin_membership)
############################################

# Remove the big components that contain a lot of MS stars
#~ try:
    #~ good_comps.remove('J')
    #~ good_comps.remove('B')
    #~ #good_comps.remove('Q')
#~ except:
    #~ pass


#~ good_comps = ['A', 'G']


# Read data
try:
    tab = tab0
    comps = comps0
    comps_raw = comps_raw0
except:
    tab0 = Table.read(data_filename)
    comps0 = Table.read(comps_filename)
    comps_raw0 = SphereComponent.load_raw_components(comps_filename)

    tab = tab0
    comps = comps0
    comps_raw = comps_raw0




fig=plt.figure(figsize=(figsize[1], figsize[0]))
ax = fig.add_subplot(111)
dim1=0
dim2=3


data=Table()
for i, c in enumerate(comps):
    # Component data
    comp_ID = c['comp_ID']
    
    # Plot only good comps
    if comp_ID not in good_comps:
        continue
    
    # Take only members
    membname = 'membership%s'%comp_ID
    col=tab[membname]
    mask=col>pmin_membership
    t=tab[mask]


    # Take only stars with RVs
    mask = t['radial_velocity_error']<100
    t = t[mask]


    if comp_ID!='Q':
        try:
            data = vstack((data, t))
        except:
            data = Table(t)

    
    age=c['Age']
    
    # Plot stars
    #~ ax.scatter(t['X'], t['U'], s=1, c=colors[comp_ID], label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    lw=0.5
    if comp_ID=='T':
        zorder=4
    else:
        zorder=1
    
    if comp_ID!='Q':
        ax.errorbar(t['X'], t['U'], xerr=t['X_error'], yerr=t['U_error'], c=colors[comp_ID], fmt='o', markersize=3, lw=lw, zorder=zorder, label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    else: # Q
        #~ ax.errorbar(t['X'], t['U'], xerr=t['X_error'], yerr=t['U_error'], edgecolors=colors[comp_ID], facecolors='none', fmt='o', markersize=3, lw=lw, zorder=zorder, label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
        ax.scatter(t['X'], t['U'], edgecolors=colors[comp_ID], facecolors='none', s=3, lw=lw, zorder=zorder, label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
        
    
    # Plot components
    if comp_ID!='Q':
        comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=True, color=colors[comp_ID], alpha=0.5, ax=ax,
                       comp_orbit=False, orbit_color='red')



    

    
    # Plot stars
    #~ ax.scatter(t['X'], t['U'], s=1, c=colors[comp_ID], label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    # Plot components
    #~ comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=True, color=colors[comp_ID], alpha=0.5, ax=ax,
                       #~ comp_orbit=False, orbit_color='red')

ax.set_xlim(-75, 200)
ax.set_ylim(-40, 40)

ax.set_xlabel('X [pc]')
ax.set_ylabel('U [km/s]')

# LEGEND
legend=ax.legend(markerscale=1, frameon=True, loc='center right', bbox_to_anchor=(0.56, 0.8), title='Kinematic ages', prop={'size': 8}, facecolor='white', edgecolor='none', framealpha=1, ncol=2)
plt.setp(legend.get_title(),fontsize=10)
#~ ax.legend(ncol=2, markerscale=1)
fig.subplots_adjust(bottom=0.15, top=0.9)




# FIT A LINE: y [km/s] = k * x [pc], so units for k are km/s/pc. 1/k should be age
z = np.polyfit(data['X'], data['U'], 1)
p = np.poly1d(z)
x = [-20, 200]
ax.plot(x, p(x), c='k')
print(z)

k = z[0]
age = 1.0/k * 0.977813106 # pc/km*s to Myr
print('Age', age, 'Myr')

ax.scatter(data['X'], data['U'], c='k', s=1)


#~ plt.savefig('XU_comps.pdf')
    
plt.show()
