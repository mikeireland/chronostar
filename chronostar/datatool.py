from __future__ import print_function, division, unicode_literals
"""
A collection of functions that aid in data preparation
for input into Chronostar
"""

from astropy.table import Table
import logging
import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar import tabletool


def get_region(ref_table, assoc_name=None,
               pos_margin=30., vel_margin=5.,
               scale_margin=None, mg_colname='Moving group'):
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

