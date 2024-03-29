"""
tabletool.py

A bunch of functions that help handle stellar data stored as
astropy table.
"""

import numpy as np
from astropy.table import Table
from astropy.units.core import UnitConversionError
import string

from . import coordinate
from . import transform

def load(filename, **kwargs):
    """Cause I'm too lazy to import Astropy.table.Table in terminal"""
    return Table.read(filename, **kwargs)

def read(filename, **kwargs):
    """Cause I'm too lazy to import Astropy.table.Table in terminal"""
    return load(filename, **kwargs)


def get_historical_cart_colnames():
    """
    Colnames look like X, Y, Z...
    dX, dY, dZ
    c_XY, c_CU

    (as opposed to modern colnames:
    X, Y, Z...
    X_error, Y_error ...
    X_Y_corr,
    """
    main_colnames = 'XYZUVW'
    error_colnames = ['d'+el for el in main_colnames]
    corr_colnames = []
    for i, colname1 in enumerate(main_colnames):
        for colname2 in main_colnames[i+1:]:
            corr_colnames.append('c_{}{}'.format(colname1, colname2))
    return main_colnames, error_colnames, corr_colnames


def get_colnames(main_colnames=None, error_colnames=None, corr_colnames=None,
                 cartesian=True):
    """
    Utility function for generating standard column names

    Parameters
    ----------
    main_colnames: [6] str array_like {None}
        The column names of the measurements. If left as None then
        if `cartesian` is true:
            ['X', 'Y', 'Z', 'U', 'V', 'W']
        if `cartesian` is false:
            ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
    error_colnames: [6] str array_like {None}
        The column names of the measurements. If left as None then
        we try to infer the names by appending '_error' to the main
        column names.
    corr_colnames: [15] str array_like {None}
        The column names of the correlations between the errors of
        each measurement pair. If left as None we try to infer the
        names by pairing each measurmenet and appending '_corr', e.g.:
        'X_Y_corr'.

    Notes
    -----
    If all column names are provided as argument, this function does
    nothing.

    The default format for column names for errors and correlations is,
    e.g.:
        X_error, Y_error, ...
        X_Y_corr, X_Z_corr, X_U_corr, X_V_corr, X_W_corr, Y_Z_corr, ...
    The correlations are listed in the same way one would read the upper
    triangle of the correlation matrix, where the rows (and columns) of
    the matrix are in the same order as `main_colnames`.
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


def build_data_dict_from_table(table, main_colnames=None, error_colnames=None,
                               corr_colnames=None, cartesian=True,
                               historical=False, only_means=False,
                               get_background_overlaps=True,
                               background_colname=None,
                               return_table_ixs=False):
    """
    Use data in tale columns to construct arrays of means and covariance
    matrices.

    Parameters
    ----------
    table: astropy table -or- string
        The table (or path to table) which holds the required data
    main_colnames: [6] string array_like
        Set of column names of the main measurements
        e.g. ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        would be the input for the default format of Gaia data
    error_colnames: [6] string array_like {None}
        Set of column names for the errors. If left as None will be
        generated by appending '_error' to each of the main_colnames
    corr_colnames: [15] string array_like {None}
        Set of column names for the pairwise correlations between each
        of the six main measurements. If left as None will be generated
        by joining each pair of main_colnames with an underscore then
        appending '_corr'.
        It is assumed that the correlation column names are given in
        a certain order based on input order of main_colnames.
        e.g. ['ra_dec_corr', 'ra_parallax_corr', ... 'ra_radial_velocity_corr',
              'dec_parallax_corr', ... 'dec_radial_velocity_corr',
              'parallax_pmra_corr' ... etc]
    cartesian: bool {True}
        Set to false if trying to build astrometric data
    historical: bool {True}
        Set to True if data set is from historical uses of chronostar. This
        function will then look for different column names by default
    only_means: bool {True}
        Set to True if only after the means of the data. This will save
        time by not building covariance matrices
    get_background_overlaps: bool {True}
        Set to True if after background overlaps too
    background_colname: str {None}
        Set which column name to use for background overlaps. If left as
        None, uses 'background_log_overlap' as default.
    return_table_ixs: boolean {False}
        If set, returns a mapping taking the indices of elements in dictionary
        to rows from original table. This is useful when table rows have
        been skipped due to missing data.
        Convert data to row indices for table assignment e.g. recording of
        membership to `comp_A` thusly:
        >>> my_table['comp_A'][table_ixs] = final_memb[:,0]

        Or to extract gaia ids of comp A members:
        >>> my_table['gaia_dr2'][table_ixs][np.where(final_memb[:,0]>0.5)]

        where `final_memb` is a [nstars, ncomps] array recording membership
        probabilities.

    Returns
    -------
    means: [n,6] float array_like
        Array of the mean measurements
    covs: [n,6,6] float array_like
        Array of the covariance matrix for each of the `n` measured objects
    
    NICH_HONS;
    age_probs: [n,64] float array_like
	Array representing the age pdf generated by honpy_one_setup {TODO rename}
	    on a axis of log(years) 5->11.4
    Comment by Marusa: it is actually a dictionary that is returned.
    """
    # Tidy up input
    if isinstance(table, str):
        table = Table.read(table)
    if historical:
        main_colnames, error_colnames, corr_colnames =\
            get_historical_cart_colnames()
    else:
        main_colnames, error_colnames, corr_colnames = get_colnames(
                main_colnames=main_colnames, error_colnames=error_colnames,
                corr_colnames=corr_colnames, cartesian=cartesian
        )

    # Generate means
    if table.masked:
        raise UserWarning('Table is masked! Replace or remove problem columns')
    means = np.vstack([table[col] for col in main_colnames]).T
    if only_means:
        return means
    results_dict = {'means':means}

    
    #TODO age parameter from .pars file
    age_parameter=True
    if age_parameter:
        from . import honpy_one_setup as hp
        gmag = table['mag_g']
        bprp = table['bprp_col']
        #initializing with the pdf of the first star)
        pdf = hp.g_kernal_den(bprp[0],gmag[0])
        pdfs= pdf
        for i in range(1,len(gmag)):
            pdf = hp.g_kernal_den(bprp[i],gmag[i])
            pdfs= np.vstack([pdfs,pdf])
        results_dict['age_probs']=pdfs
    #except:
	   # print('NICH_HONS; failure at tabletool.py line 195-207') 
    

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
    # multiply the 'ijk'th element from covs by the 'ij'th element from
    # standard_devs. More thoroughly: for the i'th covariance matrix,
    # and the i'th 6D standard deviation vector, multiply the j'th row
    # by the j'th std
    covs = np.einsum('ijk,ij->ijk', covs, standard_devs)    # the rows
    covs = np.einsum('ijk,ik->ijk', covs, standard_devs)    # the columns
    results_dict['covs'] = covs

    # Checks for any nans in the means or covariances
    bad_mean_mask = np.any(np.isnan(means), axis=1)
    bad_cov_mask = np.any(np.isnan(covs), axis=(1,2))

    good_row_mask = np.logical_not(np.logical_or(bad_mean_mask, bad_cov_mask))
    
    # Notify what stars have been excluded!
    nexcluded = np.sum(np.logical_not(good_row_mask))
    if nexcluded>0:
        print('%d stars MASKED OUT!'%nexcluded)
        print(np.where(np.logical_not(good_row_mask)))
        print(table[np.logical_not(good_row_mask)])
    #TODO age parameter from .pars file
    if age_parameter:
        results_dict = {
            'means':means[good_row_mask],
            'covs':covs[good_row_mask],
            #NICH_HONS;
            'age_probs':pdfs[good_row_mask]
        }
    else:
        results_dict = {
            'means':means[good_row_mask],
            'covs':covs[good_row_mask],
        }

    # Insert background overlaps
    if get_background_overlaps:
        if background_colname is None:
            background_colname = 'background_log_overlap'
    if background_colname in table.colnames:
        results_dict['bg_lnols'] = np.array(table[background_colname])[good_row_mask]

    if return_table_ixs:
        return results_dict, np.where(good_row_mask)
    else:
        return results_dict

def construct_an_astropy_table_with_gaia_ids_and_membership_probabilities(table, 
    memb_probs, comps, output_filename, get_background_overlaps=True, stellar_id_colname=None, overwrite_fits=False):
    """
    MZ 2020 - 04 - 16
    Create an astropy table with Gaia DR2 ids and membership probabilities
    for all components, including background.
    
    This shoul NOT append to the original table because the number of
    components is increasing each iteration.
    Parameters
    ----------
    table: astropy table -or- string
        The table (or path to table) which holds the required data
    get_background_overlaps: bool {True}
        Set to True if after background overlaps too
 
    Returns
    -------
    None
    
    """

    # Read table
    if isinstance(table, str):
        table = Table.read(table)
    ids = table[stellar_id_colname]
    tab = Table((ids,), names=(stellar_id_colname,))

    # compnames
    # TODO: This should be generated once in the component class!!
    ncomps = len(comps)
    if ncomps>26:
        print('*** number of components>26, cannot name them properly with letters.')
    abc=string.ascii_uppercase
    compnames = [abc[i] for i in range(ncomps)]

    # Membership
    for i, c in enumerate(compnames):
        tab['membership%s'%c.replace('comp', '')] = memb_probs[:,i]

    #~ todo='background_log_overlap'
    if get_background_overlaps:
        tab['membership_bg'] = memb_probs[:,-1]

    print(tab)
    tab.write(output_filename, format='fits', overwrite=overwrite_fits)

    #add number of components in the file. and a timestamp or random number so nothing gets overwritten.


def append_cart_cols_to_table(table, main_colnames=None, error_colnames=None,
                              corr_colnames=None):
    """
    Insert empty place holder columns for cartesian values

    Parameters
    ----------
    table: astropy.table.Table object
        Modifies table in place by appending empty columns for cartesian
        values. Default values in column are `np.nan`.
    main_colnames: str {None}
        See


    Returns
    -------
    None
    """
    # Tidy input
    if isinstance(table, str):
        table = Table.read(table)
    main_colnames, error_colnames, corr_colnames =\
        get_colnames(main_colnames, error_colnames, corr_colnames,
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


def convert_astro2cart(astr_mean, astr_cov):
    """
    Convert astrometry data (mean and covariance) into cartesian
    coordinates, centred on the local standard of rest (Schoenrich 2010).

    Parameters
    ----------
    astr_mean: [6] float array_like
        The central estimate of a star's astrometry values. Provided in
        the order:
            ra [deg]
            dec [deg]
            parallax [mas]
            pmra*cos(dec) [mas/yr]
            pmdec [mas/yr]
            radial velocity [km/s]
    astr_cov: [6,6] float array_like
        The covariance matrix of the measurments with columns (and rows)
        in same order as `astr_mean`.

    Returns
    -------
    xyzuvw_mean: [6] float array_like
        The cartesian mean (XYZUVW)
    xyzuvw_cov: [6,6] float array_like
        The carteisan covariance matrix
    """
    xyzuvw_mean = coordinate.convert_astrometry2lsrxyzuvw(astr_mean)
    xyzuvw_cov = transform.transform_covmatrix(
            cov=astr_cov, trans_func=coordinate.convert_astrometry2lsrxyzuvw,
            loc=astr_mean
    )

    return xyzuvw_mean, xyzuvw_cov


def insert_data_into_row(row, mean, cov, main_colnames=None, error_colnames=None,
                         corr_colnames=None, cartesian=True):
    """
    Insert data, error and correlations into a single row

    Given the mean and covariance matrix, we derive the standard
    deviations in each dimension as well as each pair-wise correlation,
    which are then inserted into the row (as per the provided column names).

    The columns must already exist!

    Parameters
    row: astropy table row
        The row in which the data will be inserted, with required columns
        already existing
    mean: [6] float array
        The mean of data
    cov: [6,6] float array
        The covariance matrix of data
    """

    main_colnames, error_colnames, corr_colnames = get_colnames(
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

    # Insert correlations
    indices = np.triu_indices(6,1)      # the indices of the upper right
                                        # triangle, excluding main diagonal
    for ix in range(len(corr_colnames)):
        try:
            fst_ix = indices[0][ix]
            snd_ix = indices[1][ix]
            row[corr_colnames[ix]] = corr_matrix[fst_ix, snd_ix]
        except KeyError:
            # It's fine if some correlation columns are missing
            pass


def insert_column(table, col_data, col_name, filename=''):
    """
    Little helper to insert column data

    Parameters
    ----------
    table: astropy table
        the table in which the new column will be inserted
    col_data: array_like
        An array of the column data. Must be same length as table
        (we don't check this)
    col_name: str
        The name of the new column
    filename: str {''}
        If not empty, save the new table to file

    Returns
    -------
    table: astropy table
        The same table, with the modification.
    """
    table[col_name] = col_data
    if filename != '':
        # TODO work out proper way to writ etables that is consistnet across python
        # Table.write(table, filename, overwrite=True, format='ascii') # TC
        table.write(filename, overwrite=True)
    return table


def convert_table_astro2cart(table, return_table=False, write_table=False,
                             astr_main_colnames=None,
                             astr_error_colnames=None,
                             astr_corr_colnames=None,
                             cart_main_colnames=None,
                             cart_error_colnames=None,
                             cart_corr_colnames=None,
                             filename=''):
    """
    Use this function to convert astrometry data to cartesian data.

    Parameters
    ----------
    table: astropy table (or string)
        The table with astrometry data (and radial velocities), either
        with column names consistent with defaults, or provided as input.
        If column names aren't specified we assume the measurements
        have column names:
            ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        With the error column names:
            ['ra_error', 'dec_error', ... ]
        And correlation column names:
            ['ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr' ... ,
             'dec_parallax_corr', 'dec_pmra_corr' ... ,
             'parallax_pmra_corr', ... ,
             ... ]
    return_table: bool {False}
        Whether to return the converted table
    write_table: bool {False}
        Whether to write the converted table to filename. It is not
        sufficient to simply supply a filename to write as we do not
        want to risk overwriting someone's table (even though we simply
        extend with new columns).
    main_colnames: [6] string array_like
        Set of column names of the main measurements
        e.g. ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']
        would be the input for the default format of Gaia data
    error_colnames: [6] string array_like {None}
        Set of column names for the errors. If left as None will be
        generated by appending '_error' to each of the main_colnames
    corr_colnames: [15] string array_like {None}
        Set of column names for the pairwise correlations between each
        of the six main measurements. If left as None will be generated
        by joining each pair of main_colnames with an underscore then
        appending '_corr'.
        It is assumed that the correlation column names are given in
        a certain order based on input order of main_colnames.
        e.g. ['ra_dec_corr', 'ra_parallax_corr', ... 'ra_radial_velocity_corr',
              'dec_parallax_corr', ... 'dec_radial_velocity_corr',
              'parallax_pmra_corr' ... etc]
    filename: str {''}
        Save filename for storing the resulting table

    Returns
    -------
    res: astropy table
        If `return_table` flag is set, will return the resulting
        astropy table
    """
    if isinstance(table, str):
        if filename and not write_table:
            raise UserWarning('Specify how to handle result, I won\'t'
                              'overwrite without explicit permission.')
        filename = table
        table = Table.read(table)

    # Get astrometric column names
    astr_main_colnames, astr_error_colnames, astr_corr_colnames =\
        get_colnames(main_colnames=astr_main_colnames,
                     error_colnames=astr_error_colnames,
                     corr_colnames=astr_corr_colnames,
                     cartesian=False)

    data = build_data_dict_from_table(table,
                                      astr_main_colnames,
                                      astr_error_colnames,
                                      astr_corr_colnames)

    # Establish what column names are used
    cart_main_colnames, cart_error_colnames, cart_corr_colnames = \
        get_colnames(cart_main_colnames,
                     cart_error_colnames,
                     cart_corr_colnames,
                     cartesian=True)

    # if cartesian columns don't exist, then insert them
    if cart_corr_colnames[0] not in table.keys():
        append_cart_cols_to_table(table,
                                  cart_main_colnames,
                                  cart_error_colnames,
                                  cart_corr_colnames)

    # Iteratively transform data to cartesian coordinates, storing as we go
    for row, astr_mean, astr_cov in zip(table, data['means'], data['covs']):
        cart_mean, cart_cov = convert_astro2cart(astr_mean, astr_cov)
        insert_data_into_row(row, cart_mean, cart_cov,
                             main_colnames=cart_main_colnames,
                             error_colnames=cart_error_colnames,
                             corr_colnames=cart_corr_colnames
                             )

    # Save data
    if filename and write_table:
        table.write(filename, overwrite=True)

    if return_table:
        return table

