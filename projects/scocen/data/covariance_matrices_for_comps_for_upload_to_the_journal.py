"""
A table of covariance matrices for all Sco-Cen components.
For upload to the journal.

Actually, for completeness, unclude all the data from Table 1
and include covariance matrices.

"""

import numpy as np
from astropy.table import Table, Column
from astropy import units as u

from chronostar.component import SphereComponent

############################################
comps_filename = 'final_comps_21.fits'
data_filename_fit = 'scocen_5k_candidates_only_members_with_significant_membership_from_tims_3_subsets.fits'

# Prepare a table with good_comps only
good_comps = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U']
############################################
comps = Table.read(comps_filename)

# Data used for the fit
data_fit = Table.read(data_filename_fit)
############################################



# Create components
comps_raw = SphereComponent.load_raw_components(comps_filename)
print('Number of components: %d'%len(comps_raw))
print('TODO: check if its using EPICYCLIC!!! as opposed to EMCEE')


result = Table()
result['ID'] = [' ']*len(comps)
result['Xt'] = [np.nan]*len(comps)
result['Yt'] = [np.nan]*len(comps)
result['Zt'] = [np.nan]*len(comps)
result['Ut'] = [np.nan]*len(comps)
result['Vt'] = [np.nan]*len(comps)
result['Wt'] = [np.nan]*len(comps)
result['Xt_error'] = [np.nan]*len(comps)
result['Yt_error'] = [np.nan]*len(comps)
result['Zt_error'] = [np.nan]*len(comps)
result['Ut_error'] = [np.nan]*len(comps)
result['Vt_error'] = [np.nan]*len(comps)
result['Wt_error'] = [np.nan]*len(comps)
result.add_column(Column(name='cov_t', dtype=np.float64, length=len(comps), shape=(6,6)))
result['X0'] = [np.nan]*len(comps)
result['Y0'] = [np.nan]*len(comps)
result['Z0'] = [np.nan]*len(comps)
result['U0'] = [np.nan]*len(comps)
result['V0'] = [np.nan]*len(comps)
result['W0'] = [np.nan]*len(comps)
result['X0_error'] = [np.nan]*len(comps)
result['V0_error'] = [np.nan]*len(comps)
result.add_column(Column(name='cov_0', dtype=np.float64, length=len(comps), shape=(6,6)))
result['Age'] = [np.nan]*len(comps)
result['Age_error'] = [np.nan]*len(comps)
#~ result['Nfit'] = [np.nan]*len(comps)
result.add_column(Column(name='Nfit', dtype=np.int, length=len(comps)))


# ADD UNITS
result['Xt'].unit = u.pc
result['Yt'].unit = u.pc
result['Zt'].unit = u.pc
result['Ut'].unit = u.km/u.s
result['Vt'].unit = u.km/u.s
result['Wt'].unit = u.km/u.s
result['Xt_error'].unit = u.pc
result['Yt_error'].unit = u.pc
result['Zt_error'].unit = u.pc
result['Ut_error'].unit = u.km/u.s
result['Vt_error'].unit = u.km/u.s
result['Wt_error'].unit = u.km/u.s
result['X0'].unit = u.pc
result['Y0'].unit = u.pc
result['Z0'].unit = u.pc
result['U0'].unit = u.km/u.s
result['V0'].unit = u.km/u.s
result['W0'].unit = u.km/u.s
result['X0_error'].unit = u.pc
result['V0_error'].unit = u.km/u.s
result['Age'].unit = u.Myr
result['Age_error'].unit = u.Myr


i=0
for c, c_raw in zip(comps, comps_raw):
    comp_id = c['comp_ID']
    
    age=c['Age']
    sigma_age = c['Crossing_time']

    mean_now = c_raw.get_mean_now()
    covmatrix_now = c_raw.get_covmatrix_now()
    
    covmatrix0 = c_raw.get_covmatrix()

    # Component at time NOW (today in the sky)
    Xt = mean_now[0]
    Yt = mean_now[1]
    Zt = mean_now[2]
    Ut = mean_now[3]
    Vt = mean_now[4]
    Wt = mean_now[5]

    sigmaXt = np.sqrt(covmatrix_now[0,0])
    sigmaYt = np.sqrt(covmatrix_now[1,1])
    sigmaZt = np.sqrt(covmatrix_now[2,2])
    sigmaUt = np.sqrt(covmatrix_now[3,3])
    sigmaVt = np.sqrt(covmatrix_now[4,4])
    sigmaWt = np.sqrt(covmatrix_now[5,5])


    # Component at time 0 (at birth)
    X0 = c['X']
    Y0 = c['Y']
    Z0 = c['Z']
    U0 = c['U']
    V0 = c['V']
    W0 = c['W']
    sigmaX0 = c['dX']
    sigmaV0 = c['dV']
    

    # Number of members used in the fit
    pmin_membership = 0.5
    mask = data_fit['membership%s'%comp_id] >= pmin_membership
    Nfit = np.sum(mask)

    result[i]['ID'] = comp_id
    result[i]['Xt'] = Xt
    result[i]['Yt'] = Yt
    result[i]['Zt'] = Zt
    result[i]['Ut'] = Ut
    result[i]['Vt'] = Vt
    result[i]['Wt'] = Wt
    result[i]['Xt_error'] = sigmaXt
    result[i]['Yt_error'] = sigmaYt
    result[i]['Zt_error'] = sigmaZt
    result[i]['Ut_error'] = sigmaUt
    result[i]['Vt_error'] = sigmaVt
    result[i]['Wt_error'] = sigmaWt
    result[i]['cov_t'] = covmatrix_now
    result[i]['X0'] = X0
    result[i]['Y0'] = Y0
    result[i]['Z0'] = Z0
    result[i]['U0'] = U0
    result[i]['V0'] = V0
    result[i]['W0'] = W0
    result[i]['X0_error'] = sigmaX0
    result[i]['V0_error'] = sigmaV0
    result[i]['cov_0'] = covmatrix0
    result[i]['Age'] = age
    result[i]['Age_error'] = sigma_age # Crossing time
    result[i]['Nfit'] = Nfit
    
    i+=1

# Take only good_comps
mask = np.in1d(result['ID'], good_comps)
result = result[mask]

print(result)

result.write('scocen_components_with_covariance_matrices.fits')
