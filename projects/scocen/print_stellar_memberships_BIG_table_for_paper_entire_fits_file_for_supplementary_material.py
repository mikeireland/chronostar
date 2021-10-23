"""
Print stellar data (gaia ids, memberships etc.) for the paper.
Prepare two versions: One is the first few lines for the paper (stellar_memberships_example.tex), 
and the other one is an online supplementary material.


Print only members of ScoCen. Background stars are not important!

"""

import numpy as np
from astropy.table import Table, unique, join
import astropy.units as u

from chronostar.component import SphereComponent

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U'] # No B and E
print('Revise component LIST!!! there is no B!!!')
############################################

# Read data
try:
    tab = tab0
except:
    tab0 = Table.read(data_filename)
    tab = tab0

print('UNIQUE TABLE')
tab = unique(tab, keys='source_id')
print(len(tab))

"""
Take only ScoCen members
"""
mask = tab['source_id']<1000 # All False
for comp_ID in good_comps:
    m = tab['membership%s'%comp_ID]>0.5
    mask = mask | m

tab = tab[mask]
print('Number of stars in good comps with memb. prob. >0.5: ', len(tab))


from collections import Counter
c_li = Counter(tab['EW(Li)_ref'])
c_rv = Counter(tab['radial_velocity_ref'])


# Find the actual Banyan references for RV
# Banyan references
b = Table.read('data/banyan_sigma_table_of_1406_stars_with_RV_references_and_gaia_ids.fits')
mask_rv_ref = tab['radial_velocity_ref']=='Banyan Sigma'
#~ print(tab[mask_rv_ref])
brv = tab[mask_rv_ref]
mask = np.in1d(b['dr2_source_id'], brv['source_id'])
b=b[mask]
b.rename_column('dr2_source_id', 'source_id')
tab = join(tab, b[['source_id', 'r_RVel']], keys='source_id', join_type='left')
tab.rename_column('r_RVel', 'radial_velocity_reference_banyan_sigma')
tab['radial_velocity_reference_banyan_sigma']=tab['radial_velocity_reference_banyan_sigma'].filled(np.nan)
print(tab)

# What references are actually in our table? I will need to click on every single reference in vizier
# because I cannot find a map between the number and the actual reference to the paper
# except if I click on it in the vizier table.
# So list all the possible numbers here and prepare a dictionary.
bc = Counter(tab['radial_velocity_reference_banyan_sigma'])
#~ print(bc)
#Counter({'nan': 9984, ' 20': 25, ' 30': 21, ' 34': 16, '171': 13, '241': 9, ' 69': 4, '100': 4, '160': 4, ' 80': 3, '164': 2, ' 12,30': 2, '244': 1, '149': 1, '271': 1, '181': 1, '172': 1})

map_banyan_reference = {' 20': '2006AstL...32..759G', ' 30': '2006A&A...460..695T', ' 34': '2007AN....328..889K', '171': 13, '241': '2012ApJ...745...56D', ' 69': '1967IAUS...30...57E', '100': '2009A&A...498..949M', '160': '2013MNRAS.435.1325M', ' 80': '1953GCRV..C......0W', '164': '2007A&A...467.1147G', ' 12,30': '2007AJ....133.2524W,2006A&A...460..695T', '244': '2011AJ....141..187S', '149': '2013A&A...551A..46L', '271': '1988Ap&SS.148..163G', '181': '2011ApJ...738..122C', '172': '1997A&A...328..187C'}
"""
20	2006AstL...32..759G	Gontcharov	Cat. III/252
30	2006A&A...460..695T	Torres et al.	Cat. J/A+A/460/695
34	2007AN....328..889K	Kharchenko et al.	Cat. J/AN/328/889 and III/254
171	 	E. Bubar et al.	2018 in preparation
241	2012ApJ...745...56D	Dahm et al.	Cat. J/ApJ/745/56
69	1967IAUS...30...57E	Evans	 
100	2009A&A...498..949M	Mermilliod et al.	Cat. J/A+A/498/949
160	2013MNRAS.435.1325M	Murphy et al.	Cat. J/MNRAS/435/1325
80	1953GCRV..C......0W	Wilson	 
164	2007A&A...467.1147G	Guenther et al.	 
12	2007AJ....133.2524W	White et al.	Cat. J/AJ/133/2524
244	2011AJ....141..187S	Siebert et al.	Cat. III/265 ; obsoleted by III/279
149	2013A&A...551A..46L	Lopez Marti et al.	Cat. J/A+A/551/A46
271	1988Ap&SS.148..163G	Garcia et al.	 
181	2011ApJ...738..122C	Chen et al.	Cat. J/ApJ/738/122
172	1997A&A...328..187C	Covino et al.	 
"""

# Find Bubar reference
mask = tab['radial_velocity_reference_banyan_sigma']=='171'
print('Bubar')
bubar = tab[mask]
print(tab[mask])

"""
Bubar et al. 2018: Lithium in Sco-Cen

MML7    Gaia DR2 5854812388999707008 MtVel  2.54 \pm 0.14
MML13   Gaia DR2 6053028978057952256 MtVel  2.53 \pm 0.28
MML28   	Gaia DR2 6080177569354209024
MML30   Gaia DR2 6080364825623490688 MtVel  2.18 \pm 0.12

"""



# References
# Merge together references for Lithium and RV as some of them are the same
ewli_ref = set(tab['EW(Li)_ref'])
rv_ref = set(tab['radial_velocity_ref'])
refs = ewli_ref.union(rv_ref)

# Banyan should be last because I should replace it with the actual references there!
refs=list(refs)
refs.append(refs.pop(refs.index('Banyan Sigma')))
refs.append(refs.pop(refs.index('')))
refs.append(refs.pop(refs.index('N/A')))

#~ refs.pop(refs.index(''))
#~ refs.pop(refs.index('N/A'))

ref_dict = dict(zip(refs, range(1, len(refs)+1)))
ref_dict['']=-99
ref_dict['N/A']=-99

print('ref_dict')
print(ref_dict)

tab['Ref'] = [('%d/%d'%(ref_dict[x['radial_velocity_ref']], ref_dict[x['EW(Li)_ref']])).replace('-99/-99', '').replace('-99', '') for x in tab]

ref_set_check = set(tab['Ref'])
print('ref_set_check')
print(ref_set_check)

cr = Counter(tab['Ref'])
print(cr)


# Rename some columns
tab.rename_column('bp_rp_extinction_corrected', '(Bp-Rp)0')
tab.rename_column('phot_g_mean_mag_extinction_corrected', 'G0')
tab.rename_column('best_component', 'comp')
tab.rename_column('best_component_membership', 'p')
tab.rename_column('EW(Li)_err', 'EW(Li)_error')


# Keys in the table
keys = ['source_id', '(Bp-Rp)0', 'G0', 'radial_velocity', 'radial_velocity_error', 'X', 'Y', 'Z', 'U', 'V', 'W', 'X_error', 'Y_error', 'Z_error', 'U_error', 'V_error', 'W_error', 'comp', 'p', 'EW(Li)', 'EW(Li)_error', 'Ref']

tab=tab[keys]

tab['radial_velocity'].unit = u.km/u.s
tab['radial_velocity_error'].unit = u.km/u.s
tab['X'].unit = u.pc
tab['Y'].unit = u.pc
tab['Z'].unit = u.pc
tab['X_error'].unit = u.pc
tab['Y_error'].unit = u.pc
tab['Z_error'].unit = u.pc
tab['U'].unit = u.km/u.s
tab['V'].unit = u.km/u.s
tab['W'].unit = u.km/u.s
tab['U_error'].unit = u.km/u.s
tab['V_error'].unit = u.km/u.s
tab['W_error'].unit = u.km/u.s
tab['EW(Li)'].unit = u.angstrom
tab['EW(Li)_error'].unit = u.angstrom


print(tab)

#~ tab.write('scocen_kinematics_and_membership_probabilities_gaiaDR2.fits')
