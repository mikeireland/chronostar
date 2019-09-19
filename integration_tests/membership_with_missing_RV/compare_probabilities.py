"""
Author: Marusa Zerjal, 2019 - 08 - 20

Compare component memberships determined with and without* radial velocities.

* Radial velocity is set to 0, its error is 1e+5.

"""
import numpy as np
from astropy.table import Table, join
import matplotlib.pyplot as plt

filename = 'results/scocen_data_with_original_radial_velocities_with_membership.fits'
filename_broken = 'results/scocen_data_with_broken_radial_velocities_with_membership.fits'

comps_filename = 'data/all_nonbg_scocen_comps_unique.npy'
c = np.load(comps_filename)
print c.shape

number_of_components = c.shape[0] # the last one is background

d = Table.read(filename)
dbroken = Table.read(filename_broken)

for i in range(1, number_of_components+1):
    dbroken.rename_column('membership_%d'%i, 'membership_%d_broken'%i)

tab = join(d, dbroken, keys='source_id')

# How many false positives do we get?
print('False positives:')
for i in range(1, number_of_components+1):
    mask = (tab['membership_%d'%i]<0.5) & (tab['membership_%d_broken'%i]>0.5)
    print 'Component %d: %d'%(i, len(tab[mask]))
print('')

# How many false negatives do we get?
print('False negatives:')
for i in range(1, number_of_components+1):
    mask_positive = tab['membership_%d'%i]>0.5
    true_positives = tab[mask_positive]

    mask = (tab['membership_%d' % i] > 0.5) & (tab['membership_%d_broken' % i] < 0.5)
    false_negatives = tab[mask]

    ratio = float(len(false_negatives)) / float(len(true_positives))

    print 'Component %d: %d (%.3f); total N of true positives: %d'%(i, len(false_negatives), ratio, len(true_positives))
print('')

#TODO Also check expected number of members. No RV cases should always have lower numbers!

fig=plt.figure()
for i in range(1, number_of_components+1):
    ax = fig.add_subplot(4, 4, i)
    ax.scatter(tab['membership_%d'%i], tab['membership_%d_broken'%i], s=1)
    ax.axhline(y=0.5, linewidth=0.5, color='k', linestyle=':')
    ax.axvline(x=0.5, linewidth=0.5, color='k', linestyle=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.annotate('%d'%i, xy=(0.2, 0.8), xytext=(0.1, 0.8))

    if i==13:
        ax.set_xlabel('p')
        ax.set_ylabel('p (no RV)')
plt.tight_layout()
plt.show()