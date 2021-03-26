"""
Lithium isochrones vs ScoCen components and their kinematic ages.
"""


import numpy as np
from astropy.table import Table, join, unique, vstack
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
good_comps = lib.good_comps
compnames = lib.compnames
colors = lib.colors
############################################
# Minimal probability required for membership
pmin_membership = 0.8
############################################


# Read data
try:
    tab=tab0
    comps = comps0

except:
    tab0 = Table.read(data_filename)

    # ADD LITHIUM INFO
    galah = Table.read('/Users/marusa/galah/lithium/ewli/GALAH_EW_Lithium_Zerjal_20181025.fits')
    lithium_galah = galah[['source_id', 'EW(Li)']]
    
    results_2m3 = Table.read('/Users/marusa/observing/23m/paper/data/2m3_results.fits')
    lithium_2m3 = results_2m3[['source_id', 'EW(Li)']]
    
    lithium = vstack([lithium_galah, lithium_2m3])
    lithium = unique(lithium, keys=['source_id'], keep='last')  # Some are repeated!
    print(lithium)

    tab0 = join(tab0, lithium, keys='source_id')
    
    bp_w1 = tab0['phot_bp_mean_mag'] - tab0['w1mpro']
    tab0['BP-W1'] = bp_w1


    comps0 = Table.read(comps_filename)
    
    tab=tab0
    comps = comps0


#Check if enough stars have WISE magnitudes. If not, then I will have to plot lithium isochrones in the Bp-Rp space!


#### PLOT DATA ###########################################
fig=plt.figure()
ax=fig.add_subplot(111)

for c in comps:
    comp_ID = c['comp_ID']
    if comp_ID not in good_comps:
        continue

    age=c['age']
    col=tab['membership%s'%comp_ID]
    mask=col>pmin_membership 
    t=tab[mask]
    
    print(comp_ID, age, len(t))
    
    if age<11:
        clr='r'
    else:
        clr='k'
        #~ print('Black: Ages less than 11Myr')

    cb=ax.scatter(t['BP-W1'], t['EW(Li)'], c=np.ones(len(t))*age, vmin=3, vmax=15)
    #~ cb=ax.scatter(t['BP-W1'], t['EW(Li)'], c=age, vmin=3, vmax=15)

cbar = plt.colorbar(cb)
cbar.ax.set_ylabel('Kinematic age [Myr]')


######### Lithium isochrones #############################
# If you want to plot abundances as well
from lithium_isochrones.lithium_abundances import LithiumAbundance

# Assumption: Initial A(Li)=3.26
from lithium_isochrones.lithium_isochrones import LithiumIsochrone

"""
Ages available:
[0.0005, 0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08, 0.1, 0.12, 0.2, 0.3, 0.4, 0.5, 0.625, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
"""

# What ages do you want to plot
ages=[0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 1.0, 3.0, 5.0, 8.0, 10.0]

liso = LithiumIsochrone(ages=ages)
iso = liso.isochrone_bpw1



# Annotation coordinates to display ages in the plot: Need to be modified manually if the ages array is changed
ann=[[3.4, 0.53], [4.9, 0.47], [3.65, 0.32], [3.4, 0.25], [3.15, 0.19], [2.8, 0.135], [2.35, 0.06], [2.1, 0.02]]

# Plot isochrones
for en, i in enumerate(iso):
    # Isochrone: x and y
    x=i[1]
    ax.plot(x[:,0], x[:,1], c='blue', linewidth=0.8)
    
    # Annotate age
    a=int(i[0]*1000.0) # Convert to Myr
    if en==0:
        a='%d Myr'%a
    try:
        ax.annotate(a, xy=tuple(ann[en]), xytext=tuple(ann[en]), color='blue', size=12)
    except:
        pass


# Plot abundances
labu = LithiumAbundance()
abu = labu.abundance_bpw1
for en, i in enumerate(abu):
    a=i[0] # abundance label
    x=i[1]
    ax.plot(x[:,0], x[:,1], c='k', linewidth=0.8)

# Plot abundance uncertainties in the BP-W1 space due to uncertain Teff-colour conversion
abuerr = labu.abundance_bpw1_errors
for en, i in enumerate(abuerr):
    a=i[0] # abundance label
    x=i[1]
    e1=x[0]
    e2=x[1]
    y=x[2]
    ax.plot(e1, y, c='grey', linewidth=0.8, linestyle='--')
    ax.plot(e2, y, c='grey', linewidth=0.8, linestyle='--')


ax.set_xlim(1, 7)
ax.set_ylim(-0.2, 0.8)
#~ ax.set_yscale('log')
ax.set_xlabel('BP-W1')
ax.set_ylabel('EW(Li) [$\mathrm{\AA}$]')

plt.savefig('lithium_isochrones.pdf')


#### PLOT LITHIUM vs BP-RP ###########################################
fig=plt.figure()
ax=fig.add_subplot(111)
colors = ['k', 'r', 'g', 'b', 'orange', 'cyan']
#~ good_comps = ['C', 'A', 'U', 'G', 'F', 'T'] # They are sorted by age
#~ good_comps = ['C', 'U', 'T'] # They are sorted by age
good_comps = ['A', 'G'] # They are sorted by age
i=0
for c in comps:
    comp_ID = c['comp_ID']
    if comp_ID not in good_comps:
        continue

    age=c['age']
    col=tab['membership%s'%comp_ID]
    mask=col>pmin_membership 
    t=tab[mask]
    
    print(comp_ID, age, len(t))
    
    if len(t)<1:
        continue

    ax.scatter(t['bp_rp'], t['EW(Li)'], c=colors[i], label=r'%s (%d) %.1f $\pm$ %.1f Myr'%(comp_ID, len(t), age, c['Crossing_time']))
    
    i+=1

ax.legend()
