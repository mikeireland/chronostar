"""
MZ - 2021 - 03 - 24

Create a new column where the best available radial velocity is chosen.
TODO: Some stars will have variable radial velocities from different catalogs - 
they might be multiple stars. Add flags for such cases where dispersion is
e.g. 2 sigma or bigger.
Add column with the literature reference for RV (which catalog was it
taken from).
"""

import numpy as np
from astropy.table import Table

data_filename = 'data/scocen_vac_EDR3.fits'

# If RV is not available, set to these values:
rv_nan = 0

# RV uncertainty should be a big number, much bigger than the background
# (Galaxy) component, otherwise the overlap between the star and
# the background depends on the star's ellipsoid size!
#~ rv_nan_uncertainty = 500
rv_nan_uncertainty = 1e+4

tab = Table.read(data_filename)

tab['radial_velocity'] = [np.nan]*len(tab)
tab['radial_velocity_error'] = [np.nan]*len(tab)
tab['radial_velocity_ref'] = [' '*20]*len(tab)
tab['radial_velocity_var'] = [False]*len(tab) # True if RV is variable

tab['EW(Li)'] = [np.nan]*len(tab)
tab['EW(Li)_err'] = [np.nan]*len(tab)
tab['EW(Li)_ref'] = [' '*20]*len(tab)


# If radial velocity is changed, you will need to determine new background overlaps
tab_for_new_bg_ols = None

for i, x in enumerate(tab):
    """
    Radial velocities
    """
    tmp=[]
    if np.isfinite(x['dr2_radial_velocity_error']):
        tmp.append(['Gaia eDR3', [x['dr2_radial_velocity'], x['dr2_radial_velocity_error']]])

    if np.isfinite(x['e_rv_obst']):
        tmp.append(['Zwitter et al. 2018', [x['rv_obst'], x['e_rv_obst']]])

    # TODO: if this is AVG velocity, what about binaries? Check this!!
    if np.isfinite(x['verr']):
        tmp.append(['APOGEE16', [x['vhelio_avg'], x['verr']]])

    if np.isfinite(x['hrv_error_sparv']):
        tmp.append(['RAVE DR6', [x['hrv_sparv'], x['hrv_error_sparv']]])

    if np.isfinite(x['rv_err_2m3']):
        tmp.append(['Zerjal et al. 2021', [x['rv_2m3'], x['rv_err_2m3']]])

    if np.isfinite(x['erv_ges']):
        tmp.append(['GES DR4', [x['rv_ges'], x['erv_ges']]])

    # If RV is not available
    if len(tmp)==0:
        tab[i]['radial_velocity'] = rv_nan
        tab[i]['radial_velocity_error'] = rv_nan_uncertainty
        tab[i]['radial_velocity_ref'] = 'N/A'
    
    # Find the best RV
    else:
        tmp = sorted(tmp, key = lambda x: x[1][1])
        best = tmp[0]
        tab[i]['radial_velocity'] = best[1][0]
        tab[i]['radial_velocity_error'] = best[1][1]
        tab[i]['radial_velocity_ref'] = best[0]
        
        # Need to compute new background overlaps
        if 'Gaia' not in best[0]:
            try:
                tab_for_new_bg_ols.add_row(x)
            except:
                tab_for_new_bg_ols = Table(x)
        
        # Maybe rather take the average. Binaries?
        #~ tab[i]['radial_velocity_var'] = 
        

    """
    Lithium
    """
    tmp=[]
    if np.isfinite(x['eew']):
        tmp.append(['Wheeler et al. 2021', [x['eew'], x['eew_err']]])

    if np.isfinite(x['ewli']):
        tmp.append(['Zerjal et al. 2021', [x['ewli'], 0.07]]) # My error estimate

    if np.isfinite(x['ew_li_']):
        tmp.append(['Zerjal et al. 2019', [x['ew_li_'], 0.05]]) # My error estimate. But this value is actually available in the catalog! TODO
    
    if len(tmp)>0:
        tmp = sorted(tmp, key = lambda x: x[1][1])
        best = tmp[0]
        tab[i]['EW(Li)'] = best[1][0]
        tab[i]['EW(Li)_err'] = best[1][1]
        tab[i]['EW(Li)_ref'] = best[0]


    if i%1000==0:
        print(i, len(tab))



# Uncomment this for writing
tab=tab.filled(-10000)
#~ tab.write(data_filename, overwrite=True)

#~ tab_for_new_bg_ols.write('tab_for_new_bg_ols.fits')



# Some stats
mask = np.isfinite(tab['dr2_radial_velocity'])
print('finite dr2_radial_velocity', np.sum(mask))

mask = ~np.isfinite(tab['dr2_radial_velocity']) & (tab['radial_velocity_error']<100)
print('RVs from external catalogs', np.sum(mask))

from collections import Counter
rvref = tab['radial_velocity_ref']
s = Counter(rvref)
for k, v in s.items():
    print(k, v)


mask = np.isfinite(tab['EW(Li)'])
print('Stars with lithium measurement', np.sum(mask))

mask = tab['EW(Li)']>0.1
print('Stars with lithium detection', np.sum(mask))


