"""
Plot XU (to investigate age reliability) for the CUTAG components.
"""

import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
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
#~ good_comps = ['C', 'U', 'T', 'A', 'G', 'E', 'F', 'D']
good_comps = ['C', 'U', 'T', 'A', 'G', 'F', 'D'] # Q only has two points, and the fit is currently meaningless
comps_for_line_fit = ['U', 'T', 'A', 'G', 'F', 'D'] # H and I have very different W velocities and are not associated with ScoCen. C screws up the fit completely!
compnames = lib.compnames
colors = lib.colors
add_labels = True
############################################
# Minimal probability required for membership
pmin_membership = 0.9 # In the paper
#~ pmin_membership = 0.5
############################################


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

data = Table()

labels = {'A': 'A (LCC)', 'C': 'C (USCO)', 'F': 'F (V1062 Sco)', 'G': 'G (UCL)', 'H': 'H (IC\,2602)', 'I': 'I (Platais 8)', 'T': 'T (UCL)', 'U': 'U (LCC)', 'D': 'D (V1062 Sco)'}

for i, c in enumerate(comps):
    # Component data
    comp_ID = c['comp_ID']
    
    # Plot only good comps
    if comp_ID not in good_comps:
        continue
    
    # Take only members
    mask = tab['membership%s'%comp_ID] > pmin_membership
    t=tab[mask]
    
    # Take only stars with RVs
    #~ mask = t['radial_velocity_error']<100
    mask = t['radial_velocity_error']<2
    t = t[mask]


    # Gather data for the line fitting
    if comp_ID in comps_for_line_fit:
        data = vstack((data, t))
    
    age=c['Age']
    
    # Plot stars
    #~ ax.scatter(t['X'], t['U'], s=1, c=colors[comp_ID], label = '%s (%.2f$\pm$%.2f Myr) %d'%(comp_ID, age, c['Age_reliable'], len(t)))
    
    lw=0.5
    if comp_ID=='T':
        zorder=4
    elif comp_ID=='C':
        zorder=0
    else:
        zorder=1
    
    if comp_ID=='D':
        markeredgecolor='k'
        markersize=2
        #~ fmt='o', 
        ecolor='k'
        elinewidth=0.3
    else:
        markeredgecolor='none'
        markersize=2
        ecolor=colors[comp_ID] # errorbar color
        elinewidth=0.5
    
    #~ label = '%s (%.2f$\pm$%.2f Myr) %d'%(labels[comp_ID], age, c['Age_reliable'], len(t))
    #~ label = '%s'%(labels[comp_ID])
    label = '%s'%(labels[comp_ID])
    label = label.replace('(', '(%.0f\,$\pm$\,%.0f\,Myr; '%(age, c['Crossing_time']))
    #~ print(label)
    ax.errorbar(t['X'], t['U'], xerr=t['X_error'], yerr=t['U_error'], c=colors[comp_ID], fmt='o', markersize=markersize, lw=lw, zorder=zorder, label=label, markeredgewidth=0.2, markeredgecolor=markeredgecolor, ecolor=ecolor, elinewidth=elinewidth)    

    
    # Plot components
    #~ comp_then=False
    comp_then=True
    #~ if comp_ID in ['T', 'G', 'A', 'U', 'F', 'E']:
    if comp_ID in ['T', 'G', 'A', 'U', 'C']:
        comp_then=True
    comps_raw[i].plot(dim1, dim2, comp_now=True, comp_then=comp_then, color=colors[comp_ID], alpha=0.2, ax=ax, comp_orbit=False, orbit_color=colors[comp_ID], label=comp_ID)


    mean_now = comps_raw[i].get_mean_now()
    print(comp_ID, mean_now[0], mean_now[3], len(t))


def add_labels_to_the_plot():
    """
    Add component labels to the plot
    """
    
    fs=12 # fontsize
    c='k'
    
    ax.annotate('A', xy=(73, 5), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('U', xy=(55, -1), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('C', xy=(120, 1), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)    
    
    ax.annotate('G', xy=(120, 9), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('T', xy=(87, 1.5), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('D', xy=(158, 4.5), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs)  
    
    ax.annotate('F', xy=(170, 10.5), 
        xycoords='data', xytext=(0, 1), textcoords='offset points', 
        color=c, fontsize=fs) 

if add_labels:
    add_labels_to_the_plot()


# FIT A LINE: y [km/s] = k * x [pc], so units for k are km/s/pc. 1/k should be age
# from python doc page: For gaussian uncertainties, use w=1/sigma, not 1/sigma**2
z, z_cov = np.polyfit(data['X'], data['U'], 1, w=1/data['U_error'], cov=True)
p = np.poly1d(z)


# Plot uncertainties of the fit
# Do the interpolation for plotting:
t = np.linspace(0, 200, 100)
n = 1 # Degree of polynomial
# Matrix with rows 1, t, t**2, ...:
TT = np.vstack([t**(n-i) for i in range(n+1)]).T
yi = np.dot(TT, p)  # matrix multiplication calculates the polynomial values
C_yi = np.dot(TT, np.dot(z_cov, TT.T)) # C_y = TT*C_z*TT.T
sig_yi = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal

 # PLOT FITTING UNCERTAINTY
#~ ax.fill_between(t, yi+sig_yi, yi-sig_yi, alpha=.25, color='grey')

k = z[0]
age = 1.0/k * 0.977813106 # pc/km*s to Myr
print('Age', age, 'Myr')

# Plot a line
x = [0, 200]
ax.plot(x, p(x), c='k', linestyle='-', linewidth=0.5, label='Expansion age = %.0f Myr'%age)
print(z)
print(z_cov)

# What stars were used in the line fit:
#~ ax.scatter(data['X'], data['U'], c='k', s=5, label='Age = %.2f Myr'%age)




ax.set_xlim(0, 200)
ax.set_ylim(-4, 14)

ax.set_xlabel('X [pc]')
ax.set_ylabel(r'U [$\mathrm{km\,s^{-1}}$]')

ax.legend(ncol=2, markerscale=5)
fig.subplots_adjust(bottom=0.15, top=0.9)

# LEGEND
handles, labels = ax.get_legend_handles_labels()
indices = [-6, -3, -2, -7, -1, -5, -4, 0, -10]
labels = [labels[i] for i in indices]
handles = [handles[i] for i in indices]
labels[-1] = 'Components at birth'

legend=ax.legend(handles, labels, markerscale=1, frameon=True, loc=2, prop={'size': 8}, facecolor='white', edgecolor='none', framealpha=1, ncol=2, title='Kinematic ages') # bbox_to_anchor=(0.88, 0.8), title='', 
plt.setp(legend.get_title(),fontsize=10)
#~ legend._legend_box.align = "left" # flush title to the left
#~ legend.legendHandles[3]._sizes = [40]
#~ frame = legend.get_frame()
#~ frame.set_facecolor('white')
#~ frame.set_edgecolor('white')



#~ plt.savefig('XU_comps_scocen.pdf')
plt.savefig('XU_comps_scocen_with_labels.pdf')
    
plt.show()
