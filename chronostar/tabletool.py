"""
A bunch of functions that help handle stellar data stored as
astropy table.
"""

import numpy as np
from astropy.table import Table
from astropy.units.core import UnitConversionError

from . import coordinate
from . import transform


def getHistoricalCartColnames():
    main_colnames = 'XYZUVW'
    error_colnames = ['d'+el for el in main_colnames]
    corr_colnames = []
    for i, colname1 in enumerate(main_colnames):
        for colname2 in main_colnames[i+1:]:
            corr_colnames.append('c_{}{}'.format(colname1, colname2))
    return main_colnames, error_colnames, corr_colnames


def getColnames(main_colnames=None, error_colnames=None, corr_colnames=None,
                cartesian=True):
    """
    Utility function for generating standard column names
    """
    if main_colnames is None:
        if cartesian:
            # main_colnames = [el for el in 'XYZUVW']
            main_colnames = ['X', 'Y', 'Z', 'U', 'V', 'W']
        else:       # provide astrometric column names
            main_colnames = [
                'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
            ]
    if error_colnames is None:
        error_colnames = [el+'_error' for el in main_colnames]
    if corr_colnames is None:
        corr_colnames = []
        for i, colname1 in enumerate(main_colnames):
            for colname2 in main_colnames[i + 1:]:
                corr_colnames.append('{}_{}_corr'.format(
                        colname1, colname2
                ))
    return main_colnames, error_colnames, corr_colnames


def buildDataFromTable(table, main_colnames=None, error_colnames=None,
                       corr_colnames=None, cartesian=True,
                       historical=False, only_means=False):
    """
    Use data in tale columns to construct arrays of means and covariance
    matrices.

    Parameters
    ----------
    table : astropy table -or- string
        The table (or path to table) which holds the required data
    main_colnames : [6] string array_like
        Set of column names of the main measurements
        e.g. ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        would be the input for the default format of Gaia data
    error_colnames : [6] string array_like {None}
        Set of column names for the errors. If left as None will be
        generated by appending '_error' to each of the main_colnames
    corr_colnames : [15] string array_like {None}
        Set of column names for the pairwise correlations between each
        of the six main measurements. If left as None will be generated
        by joining each pair of main_colnames with an underscore then
        appending '_corr'.
        It is assumed that the correlation column names are given in
        a certain order based on input order of main_colnames.
        e.g. ['ra_dec_corr', 'ra_parallax_corr', ... 'ra_radial_velocity_corr',
              'dec_parallax_corr', ... 'dec_radial_velocity_corr',
              'parallax_pmra_corr' ... etc]

    Returns
    -------
    means : [n,6] float array_like
        Array of the mean measurements
    covs : [n,6,6] float array_like
        Array of the covariance matrix for each of the `n` measured objects
    """
    # Tidy up input
    if isinstance(table, str):
        table = Table.read(table)
    if historical:
        main_colnames, error_colnames, corr_colnames =\
            getHistoricalCartColnames()
    else:
        main_colnames, error_colnames, corr_colnames = getColnames(
                main_colnames=main_colnames, error_colnames=error_colnames,
                corr_colnames=corr_colnames, cartesian=cartesian
        )

    # Generate means
    means = np.vstack([table[col] for col in main_colnames]).T
    if only_means:
        return means

    # Generate covariance matrices
    nstars = len(table)
    standard_devs = np.vstack([table[col] for col in error_colnames]).T

    # Detect mismatch in units and scale standard_devs appropriately
    # If units can't be converted
    for ix, (main_colname, error_colname) in\
            enumerate(zip(main_colnames, error_colnames)):
        if table[main_colname].unit != table[error_colname].unit:
            try:
                scale_factor =\
                    table[error_colname].unit.to(table[main_colname].unit)
                standard_devs[:,ix] *= scale_factor
            except UnitConversionError:
                print(main_colname, error_colname)
                raise UserWarning('Units are not convertible between '
                                  'measurments and errors. Are you sure '
                                  'you provided column names in a consistent '
                                  'ordering?')
            except AttributeError:
                # Units haven't been provided. Which is allowed but discouraged
                pass

    # Initialise an array of 6x6 identity matrices
    covs = np.array(nstars * [np.eye(6)])

    # Then turn into correlation matrices by incorporating correlation columns
    indices = np.triu_indices(6,1)      # the indices of the upper right
                                        # triangle, excluding main diagonal
    for ix in range(len(corr_colnames)):
        try:
            fst_ix = indices[0][ix]
            snd_ix = indices[1][ix]
            covs[:, fst_ix, snd_ix] = table[corr_colnames[ix]]
            covs[:, snd_ix, fst_ix] = table[corr_colnames[ix]]
        except KeyError:        # Correlations are allowed to be missing
            pass

    # Now multiply through the standard deviations along both axes
    # First along each column
    # We use einstein notation here such that 'ijk,ij->ijk' means
    # Multiply the 'ijk'th element from covs by the 'ij'th element from
    # standard_devs.
    # More thoroughly: for the i'th covariance matrix, and the i'th 6D
    # standard deviation vector, multiply the j'th row by the j'th std
    covs = np.einsum('ijk,ij->ijk', covs, standard_devs)    # the rows
    covs = np.einsum('ijk,ik->ijk', covs, standard_devs)    # the columsn
    res = {
        'means':means,
        'covs':covs,
    }
    return res


def appendCartColsToTable(table, main_colnames=None, error_colnames=None,
                          corr_colnames=None):
    """
    Insert empty place holder columns for cartesian values

    Parameters
    ----------
    table : astropy.table.Table object
        Modifies table in place by appending empty columns for cartesian
        values. Default values in column are `np.nan`.

    Returns
    -------
    None
    """
    # Tidy input
    if isinstance(table, str):
        table = Table.read(table)
    main_colnames, error_colnames, corr_colnames =\
        getColnames(main_colnames, error_colnames, corr_colnames,
                    cartesian=True)

    # Set up order of column names in table
    cart_colnames = []
    for measure, error in zip(main_colnames, error_colnames):
        cart_colnames.append(measure)
        cart_colnames.append(error)
    for corr in corr_colnames:
        cart_colnames.append(corr)

    # Insert blank rows (default value 'np.nan') with appropriate units
    nrows = len(table)
    empty_col = np.array(nrows * [np.nan])
    units = 6*['pc'] + 6*['km/s'] + 15*[None]
    for col_name, unit in zip(cart_colnames, units):
        table[col_name] = empty_col
        table[col_name].unit = unit


def convertAstroToCart(astr_mean, astr_cov):
    xyzuvw_mean = coordinate.convertAstrometryToLSRXYZUVW(astr_mean)

    xyzuvw_cov = transform.transformCovMat(
            cov=astr_cov, trans_func=coordinate.convertAstrometryToLSRXYZUVW,
            loc=astr_mean
    )

    return xyzuvw_mean, xyzuvw_cov


def insertDataIntoRow(row, mean, cov, main_colnames=None, error_colnames=None,
                      corr_colnames=None, cartesian=True):

    main_colnames, error_colnames, corr_colnames = getColnames(
            main_colnames, error_colnames, corr_colnames, cartesian=cartesian
    )
    # Insert mean data
    for ix, main_colname in enumerate(main_colnames):
        row[main_colname] = mean[ix]

    # Insert errors
    standard_devs = np.sqrt(np.diagonal(cov))
    for ix, error_colname in enumerate(error_colnames):
        row[error_colname] = standard_devs[ix]

    # Build correlation matrix by dividing through by stdevs in both axes

    corr_matrix = cov / standard_devs / standard_devs.reshape(6, 1)

    # corr_matrix = np.copy(cov)
    # inv_standard_devs = 1./standard_devs
    # corr_matrix = np.einsum('jk,j->jk', corr_matrix, inv_standard_devs) # rows
    # corr_matrix = np.einsum('jk,k->jk', corr_matrix, inv_standard_devs) # cols

    # Insert correlations
    indices = np.triu_indices(6,1)      # the indices of the upper right
                                        # triangle, excluding main diagonal
    for ix in range(len(corr_colnames)):
        try:
            fst_ix = indices[0][ix]
            snd_ix = indices[1][ix]
            row[corr_colnames[ix]] = corr_matrix[fst_ix, snd_ix]
        except KeyError:
            pass
            # raise UserWarning, '{} missing from columns'.format(
            #         corr_colnames[ix]
            # )


def convertTableAstroToXYZUVW(table, return_table=False, write_table=False,
                              main_colnames=None, error_colnames=None,
                              corr_colnames=None, filename=''):
    if isinstance(table, str):
        if filename and not write_table:
            raise UserWarning('Specify how to handle result, won\'t overwrite'
                              'without explicit permission.')
        filename = table
        table = Table.read(table)

    main_astr_colnames, error_astr_colnames, corr_astr_colnames =\
        getColnames(main_colnames=main_colnames, error_colnames=error_colnames,
                    corr_colnames=corr_colnames, cartesian=False)

    data = buildDataFromTable(table,
                              main_astr_colnames,
                              error_astr_colnames,
                              corr_astr_colnames)

    # if cartesian columns don't exist, then insert them
    if 'X_V_corr' not in table.keys():
        appendCartColsToTable(table)
    main_cart_colnames, error_cart_colnames, corr_cart_colnames = \
        getColnames(cartesian=True)

    # Iteratively transform data to cartesian coordinates, storing as we go
#     cart_means = np.zeros(astr_means.shape)
#     cart_covs = np.zeros(astr_covs.shape)
#     for ix, (astr_mean, astr_cov) in enumerate(zip(astr_means, astr_covs)):
#         cart_means[ix], cart_covs[ix] = convertAstroToCart(astr_mean, astr_cov)


    for row, astr_mean, astr_cov in zip(table, data['means'], data['covs']):
        cart_mean, cart_cov = convertAstroToCart(astr_mean, astr_cov)
        insertDataIntoRow(row, cart_mean, cart_cov,
                          main_colnames=main_cart_colnames,
                          error_colnames=error_cart_colnames,
                          corr_colnames=corr_cart_colnames
                          )

    if filename and write_table:
        table.write(filename, overwrite=True)

    if return_table:
        return table

