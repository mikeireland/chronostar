from __future__ import print_function, division, unicode_literals
"""
A collection of functions that aid in data preparation
for input into Chronostar
"""

from astropy.table import Table
from datetime import datetime
import logging
import numpy as np
import os.path

from . import tabletool
from . import readparam
from . import expectmax

def get_region(ref_table, assoc_name=None,
               pos_margin=30., vel_margin=5.,
               scale_margin=None, mg_colname=None):
    """
    Get a 6D box surrounding a set of stars identified in a fits table,
    with the entry `assoc_name` in the column `mg_colname`.

    Reference table should already have data in LSR-centric cartesian
    coordinates with colnames {'X', 'Y', ... 'W'}
    (as assumed by tabletool.build_data_dict), but does not need to
    have covariance matrix elements.

    Parameters
    ----------
    ref_table: str or astropy.table.Table {None}
        filename to fits table, e.g. the BANYAN reference table
    assoc_name: str {None}
        Name of the association as listed in the `mg_colname` column of
        provided table. If left as None, then all entries in table are
        used.
        E.g. if ref_table is the BANYAN, then accepted
        moving group names are:
        {'118 Tau', '32 Orionis', 'AB Doradus', 'Carina', 'Carina-Near',
        'Columba', 'Coma Ber', 'Corona Australis', 'Hyades', 'IC 2391',
        'IC 2602', 'Lower Centaurus-Crux', 'Octans', 'Platais 8',
        'Pleiades', 'TW Hya', 'Taurus', 'Tucana-Horologium',
        'Upper Centaurus Lupus', 'Upper CrA', 'Upper Scorpius',
        'Ursa Major', 'beta Pictoris', 'chi{ 1 For (Alessi 13)',
        'epsilon Cha', 'eta Cha', 'rho Ophiuci'}
    pos_margin: float {30.}
        Margin in position space around known members from which new candidate
        members are included
    vel_margin: float {5.}
        Margin in velocity space around known members from which new candidate
        members are included
    mg_colname: str {'Moving group'}
        The column name of the moving group

    Returns
    -------
    box_lower_bounds: [6] float array
        The lower bounds of the 6D box [X,Y,Z,U,V,W]
    box_upper_bounds: [6] float array
        The upper bounds of the 6D box [X,Y,Z,U,V,W]
    """

    ## Commenting this out for now. Usage by me (Tim) should be the same
    ## as everyone else. i.e. no hardcoded filenames for convenience.
    # if gagne_reference_data is None:
    #     gagne_reference_data =\
    #         '/home/tcrun/chronostar/data/gagne_bonafide_full_kinematics_with_lit_and_best_radial_velocity' \
    #         '_comb_binars_with_banyan_radec.fits'

    if mg_colname is None:
        mg_colname = 'Moving group'

    # If reference table is provided as str, convert to table
    if type(ref_table) is str:
        ref_table = Table.read(ref_table)

    # Extract all stars
    if assoc_name is None:
        subtable = ref_table
    else:
        if assoc_name not in set(ref_table[mg_colname]):
            raise UserWarning(
                    'Association name must be one of:\n{}\nReceived: "{}"'.format(
                            list(set(ref_table[mg_colname])), assoc_name
                    ))
        subtable = ref_table[np.where(ref_table[mg_colname] == assoc_name)]
        logging.info('Initial membership list has {} members'.format(len(subtable)))

    star_means = tabletool.build_data_dict_from_table(subtable, only_means=True,
                                                      cartesian=True)

    data_upper_bound = np.nanmax(star_means, axis=0)
    data_lower_bound = np.nanmin(star_means, axis=0)
    logging.info('Stars span from {} to {}'.format(
        np.round(data_lower_bound),
        np.round(data_upper_bound)
    ))

    # First try and scale box margins by provided scale margin.
    # scale_margin of 1 would double total span (1 + 1)
    if scale_margin is not None:
        data_span = data_upper_bound - data_lower_bound
        box_margin = 0.5 * scale_margin * data_span

        # Set up boundaries of box that span double the association
        box_lower_bound = data_lower_bound - box_margin
        box_upper_bound = data_upper_bound + box_margin

    # Set margin based on provided (or default) constant amounts
    else:
        data_margin = np.array(3*[pos_margin] + 3*[vel_margin])
        box_lower_bound = data_lower_bound - data_margin
        box_upper_bound = data_upper_bound + data_margin

    logging.info('Range extended.\nLower: {}\nUpper: {}'.format(
        np.round(box_lower_bound),
        np.round(box_upper_bound)
    ))

    return box_lower_bound, box_upper_bound


def prepare_data(data_pars):
    """
    Entry point for complete data preparation.

    Set parameters as desired in either a .par file or as a dictionary.
    This function will go through the main data file (as named in
    `data_pars`) and apply all requested data conversions.

    Key functionalities include:
        - convert astrometry measurements, errors and (opt.) correlations
            into cartesian means, standard devs and correlations
        - calculate background overlaps of stars with typical Gaia 6D
            phase-space densities
        - Apply a phase-space data cut based on a cartesian region

    Parameters
    ----------
    data_pars : dict -or- filename
        Parameters that govern behaviour of prepare_data. Exhaustive
        list of options is included in main README.md.
        data_pars can be provided as a dictionary, or as a filename with
        [key] = [value] pairs, to be parsed by `readparam.readParam`

    Returns
    -------
    data_table [opt.]: astropy.Table object

    Notes
    -----
    TODO: update background overlaps to allow for multiprocessing
    TODO: test functionality of overlap calculations
    TODO: Implement initialising synethetic datasets?
    TODO: Implement various input checks
    TODO: Allow for checkpoint saves after each stage
    TODO: Add a logging.log output
    """
    if type(data_pars) is str:
        data_pars = readparam.readParam(data_pars)

    data_pars = readparam.update_data_defaults(data_pars)
    readparam.log_used_pars(data_pars)

    # Input quality checks
    # --------------------------------------------------
    # --  INPUT PARAMETER QUALITY CHECKS  --------------
    # --------------------------------------------------
    # If overwrite is not set, ensure output_file doesn't exist.
    if (not data_pars['overwrite_datafile'] and
        os.path.isfile(data_pars['output_file'])):
        raise UserWarning('Output file exists, yet you have not set'
                          ' `overwrite_data = True` in the input parameters.')

    # If applying cartesian cuts, ensure either cut_on_region or
    # cut_on_bounds has necessary parameters set.
    if data_pars['apply_cart_cuts']:
        if data_pars['cut_on_region']:
            try:
                if type(data_pars['cut_ref_table']) is str:
                    assert os.path.isfile(data_pars['cut_ref_table'])
                elif type(data_pars['cut_ref_table']) == Table:
                    pass
                else:
                    raise TypeError
            except (AssertionError, TypeError):
                raise UserWarning('You have set `cut_on_region`, but there is'
                                  ' an issue with the provided table '
                                  ' cut_ref_table`.')
        elif data_pars['cut_on_bounds']:
            print('here')
            try:
                assert len(data_pars['cut_bound_min']) == 6
                assert len(data_pars['cut_bound_max']) == 6
            except (TypeError, AssertionError):
                raise UserWarning('You have set `cut_on_bounds` yet there'
                                  ' is an issue with your provided cartesian'
                                  ' boundaries.')
        else:
            UserWarning('If setting `apply_cart_cuts` to True, then'
                        ' either `cut_on_region` or `cut_on_bounds` must'
                        ' also be set.')

    # If calculating background, ensure all required info is provided
    if data_pars['calc_overlaps']:
        try:
            assert os.path.isfile(data_pars['bg_ref_table'])
        except (AssertionError, TypeError):
            raise UserWarning('You have set `calc_overlaps`, but there is'
                              ' an issue with the provided table '
                              ' bg_ref_table`.')

    # Establish what column names are
    data_table = Table.read(data_pars['input_file'])

    if data_pars['convert_astrometry']:
        # --------------------------------------------------
        # --  CONVERT ASTROMETRY INTO CARTESIAN  -----------
        # --------------------------------------------------
        data_table = tabletool.convert_table_astro2cart(
                table=data_table,
                astr_main_colnames=data_pars['astr_main_colnames'],
                astr_error_colnames=data_pars['astr_error_colnames'],
                astr_corr_colnames=data_pars['astr_corr_colnames'],
                cart_main_colnames=data_pars['cart_main_colnames'],
                cart_error_colnames=data_pars['cart_error_colnames'],
                cart_corr_colnames=data_pars['cart_corr_colnames'],
                return_table=True,
        )


    if data_pars['apply_cart_cuts']:
        # --------------------------------------------------
        # --  APPLY DATA CUTS IN CARTESIAN SPACE  ----------
        # --------------------------------------------------
        # First try and form region around a subset of reference
        # stars.
        if data_pars['cut_on_region']:
            bounds_min, bounds_max = get_region(
                    ref_table=data_pars['cut_ref_table'],
                    assoc_name=data_pars['cut_assoc_name'],
                    mg_colname=data_pars['cut_colname']
            )
        # Otherwise, use some mins and maxs from the pars file
        elif data_pars['cut_on_bounds']:
            bounds_min = np.array(data_pars['cut_bound_min'])
            bounds_max = np.array(data_pars['cut_bound_max'])
        else:
            UserWarning('If setting `apply_cart_cuts` to True, then'
                        ' either `cut_on_region` or `cut_on_bounds` must'
                        ' also be set.')

        input_means = tabletool.build_data_dict_from_table(
                table=data_table,
                main_colnames=data_pars['cart_main_colnames'],
                only_means=True,
        )
        cart_cut_mask = np.where(
                np.all(input_means > bounds_min, axis=1)
                & np.all(input_means < bounds_max, axis=1)
        )
        data_table = data_table[cart_cut_mask]


    if data_pars['calc_overlaps']:
        # --------------------------------------------------
        # --  CALCULATE BACKGROUND OVERLAPS  ---------------
        # --------------------------------------------------
        # Only accessing the main column names
        bg_star_means = tabletool.build_data_dict_from_table(
                table=data_pars['bg_ref_table'],
                main_colnames=data_pars['bg_main_colnames'],
                only_means=True,
        )
        input_data_dict = tabletool.build_data_dict_from_table(
                table=data_table,
                main_colnames=data_pars['cart_main_colnames'],
                error_colnames=data_pars['cart_error_colnames'],
                corr_colnames=data_pars['cart_corr_colnames'],
        )

        #TODO: A parallelised version of this exists, incorporate it?
        ln_bg_ols = expectmax.get_background_overlaps_with_covariances(
                background_means=bg_star_means,
                star_means=input_data_dict['means'],
                star_covs=input_data_dict['covs'],
        )

        tabletool.insert_column(table=data_table,
                                col_data=ln_bg_ols,
                                col_name=data_pars['bg_col_name'],
                                )

    # Store output. Since this everything above is so computationally
    # expensive, if writing to the prescribed output fails, make sure
    # the result is stored somewhere.
    try:
        data_table.write(data_pars['output_file'],
                         overwrite=data_pars['overwrite_datafile'])
    except:
        emergency_filename = 'emergency_data_save_{:.0f}.fits'.format(
                datetime.timestamp(datetime.now())
        )
        data_table.write(emergency_filename, overwrite=True)
        print("COULDN'T SAVE TO DESIGNATED OUTPUT FILE.\n"
              "Managed an emergency save to {}".format(emergency_filename))

    if data_pars['return_data_table']:
        return data_table

