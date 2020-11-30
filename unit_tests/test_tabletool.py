"""
Test a bunch of functions that serve as an interface to standard stellar data
table
"""
import numpy as np
import logging

from astropy.io import fits
from astropy.table import Table
# from astropy.units.core import UnitConversionError
try:
    import exceptions
except ImportError:
    import builtins as exceptions # python 3 consistent

import sys
sys.path.insert(0, '..')
from chronostar.component import SphereComponent
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import coordinate
from chronostar import transform

HIST_FILE_NAME = 'sample_data/small_historical_sample_table.fits'
CURR_FILE_NAME = 'sample_data/small_current_sample_table.fits'

# -----------------------------------------------
# -- To check correctness, have copied over    --
# -- previous implementation into this script  --
# -----------------------------------------------
# retired function, put here for comparison reasons
def transformAstrCovsToCartesian(astr_covs, astr_arr):
    """
    Converts a covariance matrix from astrometric coords to LSR XYZUVW
    Parameters
    ----------
    astr_covs: ([nstars, 6, 6] array)
        values in the diagaonal are the squared errors of
        (ra, dec, plx, pm_ra, pm_dec, rv), with the offdiagonals the product
        of the correlation (valued between -1 and 1) and the two
        intersecting coordinates.
    astr_arr: ([nstars, 6] array)
        the measured (mean) astrometric values
        (ra, dec, plx, pm_ra, pm-dec, rv)
    """
    nstars = astr_arr.shape[0]
    xyzuvw_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        xyzuvw_covs[ix] = transform.transform_covmatrix(
            astr_covs[ix], coordinate.convert_astrometry2lsrxyzuvw, astr_arr[ix],
            dim=6
        )
    return xyzuvw_covs

# retired function, put here for comparison reasons
def convertAstrErrsToCovs(err_arr):
    """
    Converts astrometry errors for each star into covariance matrices

    Note that a negligible error is inserted for ra and dec

    Parameters
    ----------
    err_arr : ([nstars, 6] float array), astrometry errors with placeholder
        0's for ra and dec: (ra, dec, pi, pm_ra, pm_dec, rv)

    Returns
    -------
    astr_covs : ([nstars, 6, 6] float array), covariance matrix made by
        inserting the square of the errors into the diagonal
    """
    err_arr_cp = np.copy(err_arr)
    nstars = err_arr_cp.shape[0]

    err_arr_cp[:, :2] = 1e-1
    logging.info("Angular position error is: {}".format(err_arr_cp[0,0]))
    print("Angular position error is: {}".format(err_arr_cp[0,0]))
    print("Changed!")

    astr_covs = np.zeros((nstars, 6, 6))
    for ix in range(nstars):
        astr_covs[ix] = np.eye(6) * np.tile(err_arr_cp[ix], (6, 1))**2
    return astr_covs

# Retired function, only here for comparison reasons
def convertTableToArray(star_table):
    nstars = star_table['radeg'].shape[0]
    measured_vals = np.vstack((
        star_table['radeg'],
        star_table['dedeg'],
        star_table['plx'],
        star_table['pmra'],
        star_table['pmde'],
        star_table['rv'],
    )).T

    errors = np.vstack((
        np.zeros(nstars),
        np.zeros(nstars),
        star_table['e_plx'],
        star_table['e_pmra'],
        star_table['e_pmde'],
        star_table['e_rv'],
    )).T
    return measured_vals, errors


# Retired funciton, only here for comparison reasons
def buildMeanAndCovMatFromRow(row):
    """
    Build a covariance matrix from a row

    Paramters
    ---------
    row : astropy Table row
        Entries: {X, Y, Z, U, V, W, dX, dY, ..., cXY, cXZ, ...}

    Return
    ------
    cov_mat : [6,6] numpy array
        Diagonal elements are dX^2, dY^2, ...
        Off-diagonal elements are cXY*dX*dY, cXZ*dX*dZ, ...
    """
    dim = 6
    CART_COL_NAMES = ['X', 'Y', 'Z', 'U', 'V', 'W',
                      'dX', 'dY',   'dZ',   'dU',   'dV',   'dW',
                            'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW',
                                    'c_YZ', 'c_YU', 'c_YV', 'c_YW',
                                            'c_ZU', 'c_ZV', 'c_ZW',
                                                    'c_UV', 'c_UW',
                                                            'c_VW']
    mean = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[:6]):
        mean[i] = row[col_name]

    std_vec = np.zeros(dim)
    for i, col_name in enumerate(CART_COL_NAMES[6:12]):
        std_vec[i] = row[col_name]
    corr_tri = np.zeros((dim,dim))
    # Insert upper triangle (top right) correlations
    for i, col_name in enumerate(CART_COL_NAMES[12:]):
        corr_tri[np.triu_indices(dim,1)[0][i],np.triu_indices(dim,1)[1][i]] \
            =row[col_name]
    # Build correlation matrix
    corr_mat = np.eye(6) + corr_tri + corr_tri.T
    # Multiply through by standard deviations
    cov_mat = corr_mat * std_vec * std_vec.reshape(6,1)
    return mean, cov_mat


# Retired funciton, put here for comparison reasons
def loadDictFromTable(table):
    """
    Takes the data in the table, builds dict with array of mean and cov matrices

    Paramters
    ---------
    table : Astropy.Table (or str)

    assoc_name : str
        One of the labels in Moving group column:
         118 Tau, 32 Orionis, AB Doradus, Carina, Carina-Near, Columba,
         Coma Ber, Corona Australis, Hyades, IC 2391, IC 2602,
         Lower Centaurus-Crux, Octans, Platais 8, Pleiades, TW Hya, Taurus,
         Tucana-Horologium, Upper Centaurus Lupus, Upper CrA, Upper Scorpius,
         Ursa Major, beta Pictoris, chi{ 1 For (Alessi 13), epsilon Cha,
         eta Cha, rho Ophiuci

    Returns
    -------
    dict :
        xyzuvw : [nstars,6] array of xyzuvw means
        xyzuvw_cov : [nstars,6,6] array of xyzuvw covariance matrices
        file_name : str
            if table loaded from file, the pathway is stored here
        indices : [nstars] int array
            the table row indices of data converted into arrays
        gaia_ids : [nstars] int array
            the gaia ids of stars successfully converted
    """
    star_pars = {}
    if type(table) is str:
        file_name = table
        star_pars['file_name'] = file_name
        table = Table.read(table)

    gaia_col_name = 'source_id'
    if gaia_col_name not in table.colnames:
        gaia_col_name = 'gaia_dr2'


    xyzuvw = []
    xyzuvw_cov = []
    indices = []
    gaia_ids = []
    nrows = len(table[gaia_col_name])
    for ix, row in enumerate(table):
        if nrows > 10000 and ix % 1000==0:
            print("Done {:7} of {} | {:.2}%".format(ix, nrows,
                                                    float(ix)/nrows*100))
        if np.isfinite(row['U']):
            mean, cov = buildMeanAndCovMatFromRow(row)
            xyzuvw.append(mean)
            xyzuvw_cov.append(cov)
            indices.append(ix)
            gaia_ids.append(row[gaia_col_name])

    star_pars['xyzuvw']     = np.array(xyzuvw).astype(np.float64)
    star_pars['xyzuvw_cov'] = np.array(xyzuvw_cov).astype(np.float64)
    star_pars['indices']    = np.array(indices)
    star_pars['gaia_ids']   = np.array(gaia_ids)
    star_pars['table']      = table
    return star_pars


# Retired funciton, put here for comparison reasons
def convertManyRecToArray(data):
    """
    Convert many Fits Records in astrometry into mean and covs (astro)

    Note: ra_error and dec_error are in 'mas' while ra and dec
    are given in degrees. Everything else is standard:
    plx [mas], pm [mas/yr], rv [km/s]

    Parameters
    ----------
    data: [nstars] array of Recs
        'source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax',
        'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr',
        'ra_pmdec_corr', 'dec_parallax_corr', 'dec_pmra_corr',
        'dec_pmdec_corr', 'parallax_pmra_corr', 'parallax_pmdec_corr',
        'pmra_pmdec_corr', 'astrometric_primary_flag', 'phot_g_mean_mag',
        'radial_velocity', 'radial_velocity_error', 'teff_val'

    Returns
    -------
        means: [nstars, 6] array
        covs: [nstars, 6] array
    """
    nstars = data.shape[0]
    means = np.zeros((nstars,6))

    means[:,0] = data['ra']
    means[:,1] = data['dec']
    means[:,2] = data['parallax']
    means[:,3] = data['pmra']
    means[:,4] = data['pmdec']
    means[:,5] = data['radial_velocity']

    # Array of dictionary keys to aid construction of cov matrices
    cls = np.array([
        ['ra_error',         'ra_dec_corr',           'ra_parallax_corr',
                'ra_pmra_corr',       'ra_pmdec_corr',       None],
        ['ra_dec_corr',      'dec_error',             'dec_parallax_corr',
                'dec_pmra_corr',      'dec_pmdec_corr',      None],
        ['ra_parallax_corr', 'dec_parallax_corr',     'parallax_error',
                'parallax_pmra_corr', 'parallax_pmdec_corr', None],
        ['ra_pmra_corr',     'dec_pmra_corr',         'parallax_pmra_corr',
                 'pmra_error',        'pmra_pmdec_corr',     None],
        ['ra_pmdec_corr',    'dec_pmdec_corr',        'parallax_pmdec_corr',
                 'pmra_pmdec_corr',   'pmdec_error',         None],
        [None,               None,                     None,
                 None,                 None, 'radial_velocity_error']
    ])

    # Construct an [nstars,6,6] array of identity matrices
    covs = np.zeros((nstars,6,6))
    idx = np.arange(6)
    covs[:, idx, idx] = 1.0

    # Insert correlations into off diagonals
    for i in range(0,5):
        for j in range(i+1,5):
            covs[:,i,j] = covs[:,j,i] = data[cls[i,j]]

    # multiply each row and each column by appropriate error
    for i in range(6):
        covs[:,i,:] *= np.tile(data[cls[i,i]], (6,1)).T
        covs[:,:,i] *= np.tile(data[cls[i,i]], (6,1)).T

    # Might need to introduce some artificial uncertainty in
    # ra and dec so as to avoid indefinite matrices (too narrow)

    # RA and DEC errors are actually quoted in mas, so we convert cov
    # entries into degrees
    covs[:,:,:2] /= 3600000.
    covs[:,:2,:] /= 3600000.

    return means, covs

# Retired function put here for comparision reasons
def convertGaiaToXYZUVWDict(astr_file):
    """
    Supposed to generate XYZYVW dictionary for input to GroupFitter

    Doesn't work on whole Gaia catalogue... too much memory I think

    TODO: Sort out a more consistent way to handle file names...
    """
    hdul = fits.open(astr_file)#, memmap=True)
    means, covs = convertManyRecToArray(hdul[1].data)
    astr_dict = {'astr_mns': means, 'astr_covs': covs}
    cart_dict = convertMeasurementsToCartesian(astr_dict=astr_dict)

    return cart_dict

# retired, put here for comparison reasons
def convertMeasurementsToCartesian(t=None, loadfile='', astr_dict=None):
    """
    Parameters
    ----------
    t : astropy Table with the following columns:
        name    : id or name of star
        radeg   : right ascension in degrees
        dedeg   : declination in degrees
        plx     : parallax in mas
        e_plx   : error of parallax in mas
        pmra    : proper motion in right ascension in mas/yr
        e_pmra  : error of pm in right ascension in mas/yr
        pmde    : proper motion in declination in mas/yr
        e_pmde  : error of pm in declination in mas/yr
        rv      : radial velocity in km/s
        e_rv    : error of radial velocity in km/s
    loadfile : (String {''})
        if t is None, try and load table from loadfile
    savefile : (String {''})
        if non-empty, will save a fits file with this filename. Appropriate
        file extension is applied if not there.

    Returns
    -------
    dict :
        t : astropy table
        xyzuvw : [nstars, 6] array
            space positions and velocities of each star
        xyzuvw_cov : [nstars, 6, 6] array
            covariance of positions and velocities of each star
    """
    while True:
        if t:
            # nstars = len(t)
            astr_arr, err_arr = convertTableToArray(t)
            astr_covs = convertAstrErrsToCovs(err_arr)
            break
        if loadfile:
            t = Table.read(loadfile, format='ascii')
            # nstars = len(t)
            astr_arr, err_arr = convertTableToArray(t)
            astr_covs = convertAstrErrsToCovs(err_arr)
            break
        if astr_dict:
            astr_arr = astr_dict['astr_mns']
            astr_covs = astr_dict['astr_covs']
            # nstars = astr_arr.shape[0]
            break
        raise StandardError


    xyzuvw = coordinate.convert_many_astrometry2lsrxyzuvw(astr_arr, mas=True)
    xyzuvw_cov = transformAstrCovsToCartesian(astr_covs, astr_arr)

    xyzuvw_dict = {'table':t, 'xyzuvw':xyzuvw, 'xyzuvw_cov':xyzuvw_cov}

    return xyzuvw_dict


def alternativeBuildCovMatrix(data):
    nstars = len(data)
    cov_labels = np.array([
        ['X', 'c_XY', 'c_XZ', 'c_XU', 'c_XV', 'c_XW'],
        ['c_XY', 'Y', 'c_YZ', 'c_YU', 'c_YV', 'c_YW'],
        ['c_XZ', 'c_YZ', 'Z', 'c_ZU', 'c_ZV', 'c_ZW'],
        ['c_XU', 'c_YU', 'c_ZU', 'U', 'c_UV', 'c_UW'],
        ['c_XV', 'c_YV', 'c_ZV', 'c_UV', 'V', 'c_VW'],
        ['c_XW', 'c_YW', 'c_ZW', 'c_UW', 'c_VW', 'W'],
    ])

    # Construct an [nstars,6,6] array of identity matrices
    covs = np.zeros((nstars,6,6))
    idx = np.arange(6)
    covs[:, idx, idx] = 1.0

    # Insert correlations into off diagonals
    for i in range(0,5):
        for j in range(i+1,5):
            covs[:,i,j] = covs[:,j,i] = data[cov_labels[i,j]]

    # multiply each row and each column by appropriate error
    for i in range(6):
        covs[:,i,:] *= np.tile(data[cov_labels[i,i]], (6,1)).T
        covs[:,:,i] *= np.tile(data[cov_labels[i,i]], (6,1)).T
    return covs


# -----------------------------------------------
# -----------------------------------------------
# --    TESTS BEGIN HERE   ---------------------
# -----------------------------------------------
# -----------------------------------------------

def test_transform_astrocart():
    orig_astr_data = \
        tabletool.build_data_dict_from_table(table=HIST_FILE_NAME, cartesian=False,
                                             historical=False)
    orig_cart_data = \
        tabletool.build_data_dict_from_table(table=HIST_FILE_NAME, cartesian=True,
                                             historical=True)
    astr_mean = orig_astr_data['means'][0]
    astr_cov  = orig_astr_data['covs'][0]

    cart_mean = orig_cart_data['means'][0]  # [67.568, -5.279, 16.888,  -3.871,   1.713,   1.997]
                                    # helio:  [67.568, -5.279, -8.112, -14.971, -10.527,  -5.253]

    cart_cov  = orig_cart_data['covs'][0]



    calc_cart_mean, calc_cart_cov = tabletool.convert_astro2cart(astr_mean, astr_cov)

def test_convertTableXYZUVWToArray():
    """
    Check that generating cartesian means and covariance matrices matches
    previous implementation
    """
    orig_star_pars = loadDictFromTable(HIST_FILE_NAME)
    main_colnames, error_colnames, corr_colnames =\
        tabletool.get_historical_cart_colnames()
    data = tabletool.build_data_dict_from_table(
            orig_star_pars['table'][orig_star_pars['indices']],
            main_colnames=main_colnames,
            error_colnames=error_colnames,
            corr_colnames=corr_colnames
    )

    assert np.allclose(orig_star_pars['xyzuvw'], data['means'])
    assert np.allclose(orig_star_pars['xyzuvw_cov'], data['covs'])

def test_convertSynthTableToCart():
    """
    Checks that current day measured cartesian values (with negligbile
    measurement error) match the true current day cartesian values
    """
    AGE = 40.
    PARS = np.array([
        [0., 0., 0., 0., 0., 0., 10., 5., AGE],
    ])
    STARCOUNTS = [50] #, 30]
    COMPONENTS = SphereComponent
    MEASUREMENT_ERROR = 1e-10

    # Generate synthetic data
    synth_data = SynthData(pars=PARS, starcounts=STARCOUNTS,
                           Components=COMPONENTS,
                           measurement_error=MEASUREMENT_ERROR,
                           )
    synth_data.synthesise_everything()

    # Convert (inplace) astrometry to cartesian
    tabletool.convert_table_astro2cart(synth_data.table)

    # Check consistency between true current-day kinematics and measured
    # current-day kinematics (with negliglbe error)
    for dim in 'XYZUVW':
        dim_now = dim.lower() + '_now'
        assert np.allclose(synth_data.table[dim_now],
                           synth_data.table[dim])


def test_convertAstrTableToCart():
    """
    Using a historical table, confirm that cartesian conversion yields
    same results by comparing the cartesian means and covariance matrices
    are identical.

    Gets historical cartesian data from building data from table cart cols.

    Gets updated cartesian data from building astro data from table cols,
    converting to cartesian (stored back into table) then building data
    from newly inserted table cart cols.
    """
    # hist_filename = '../data/paper1/historical_beta_Pictoris_with_gaia_small_everything_final.fits'
    hist_table = Table.read(HIST_FILE_NAME)

    # curr_filename = '../data/paper1/beta_Pictoris_with_gaia_small_everything_final.fits'
    curr_table = Table.read(CURR_FILE_NAME)
    # Drop stars that have gone through any binary checking
    # hist_table = Table(hist_table[100:300])
    # curr_table = Table(curr_table[100:300])

    # load in original means and covs
    orig_cart_data =\
        tabletool.build_data_dict_from_table(table=hist_table, cartesian=True,
                                             historical=True)

    tabletool.convert_table_astro2cart(table=curr_table, write_table=False)

    cart_data = tabletool.build_data_dict_from_table(curr_table, cartesian=True)

    assert np.allclose(orig_cart_data['means'], cart_data['means'])
    assert np.allclose(hist_table['dX'], curr_table['X_error'])
    assert np.allclose(orig_cart_data['covs'], cart_data['covs'])


def test_badColNames():
    """
    Check that columns have consistent (or absent) units across measurements
    and errors

    First test comparing column with degrees to column with mas/yr raises
    UserWarning
    Then test comparing colum with degrees to column without units raises
    no issue.
    """
    main_colnames, error_colnames, corr_colnames = \
        tabletool.get_colnames(cartesian=False)

    # main_colnames[5] = 'radial_velocity_best'
    # error_colnames[5] = 'radial_velocity_error_best'
    # corrupt ordering of column names
    corrupted_error_colnames = list(error_colnames)
    corrupted_error_colnames[0], corrupted_error_colnames[3] =\
        error_colnames[3], error_colnames[0]

    # filename = '../data/paper1/beta_Pictoris_with_gaia_small_everything_final.fits'
    table = Table.read(CURR_FILE_NAME)

    # Only need a handful of rows
    table = Table(table[:10])

    # Catch when units are inconsistent
    try:
        tabletool.convert_table_astro2cart(
                table,
                astr_main_colnames=main_colnames,
                astr_error_colnames=corrupted_error_colnames,
                astr_corr_colnames=corr_colnames
        )
    except Exception as e:
        assert type(e) == exceptions.UserWarning

    # In the case where units have not been provided, then just leave it be
    try:
        error_colnames[0] = 'ra_dec_corr'
        tabletool.convert_table_astro2cart(table,
                                           astr_main_colnames=main_colnames,
                                           astr_error_colnames=error_colnames,
                                           astr_corr_colnames=corr_colnames)

    except:
        assert False

def test_build_data_from_incomplete_table():
    """
    Sometimes rows will be missing data, e.g. from when binaries
    have been merged. build_data_from_dict should detect the
    presence of nans and skip them
    """
    # build a dummy table of data
    NSTARS = 10
    NDIM = 6
    missing_row_ix = (np.array([0,3,4]),)
    means = np.random.rand(NSTARS,NDIM)
    covs = np.array(NSTARS*[np.eye(NDIM,NDIM)])
    nan_mask = np.array(NSTARS*[False])

    # check bad data are within index range
    assert np.all(missing_row_ix[0] < NSTARS)
    nan_mask[missing_row_ix] = True
    covs[nan_mask] = np.nan

    names = np.arange(NSTARS)

    dummy_table = Table()
    dummy_table['names'] = names

    tabletool.append_cart_cols_to_table(dummy_table)

    for row, mean, cov in zip(dummy_table, means, covs):
        tabletool.insert_data_into_row(row, mean, cov)

    star_pars = tabletool.build_data_dict_from_table(dummy_table)

    assert not np.any(np.isnan(star_pars['means']))
    assert not np.any(np.isnan(star_pars['covs']))

    # check the correct number of rows have been returned
    assert len(star_pars['means']) == np.sum(np.logical_not(nan_mask))
    assert len(star_pars['covs']) == np.sum(np.logical_not(nan_mask))


if __name__ == '__main__':
    test_transform_astrocart()
    test_convertAstrTableToCart()

