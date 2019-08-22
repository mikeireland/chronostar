from __future__ import print_function, division, unicode_literals

import logging
import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar import tabletool


def get_region(assoc_name, pos_margin=30., vel_margin=5.,
               scale_margin=None,
               gagne_reference_data=None):
    """
    Get a 6D box surrounding a known association with members from BANYAN

    Parameters
    ----------
    assoc_name: str
        Name of the association as listed in BANYAN table. One of:
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
    gagne_reference_data: str
        filename to BANYAN table

    Returns
    -------
    box_lower_bounds: [6] float array
        The lower bounds of the 6D box [X,Y,Z,U,V,W]
    box_upper_bounds: [6] float array
        The upper bounds of the 6D box [X,Y,Z,U,V,W]
    """

    if gagne_reference_data is None:
        gagne_reference_data =\
            '../data/gagne_bonafide_full_kinematics_with_lit_and_best_radial_velocity' \
            '_comb_binars_with_banyan_radec.fits'

    gagne_table = tabletool.read(gagne_reference_data)

    if assoc_name not in set(gagne_table['Moving group']):
        raise UserWarning(
            'Association name must be one of:\n{}\nReceived: "{}"'.format(
                    list(set(gagne_table['Moving group'])), assoc_name
            ))

    # Extract all stars
    subtable = gagne_table[np.where(gagne_table['Moving group'] == assoc_name)]
    logging.info('Initial membership list has {} members'.format(len(subtable)))

    star_means = tabletool.build_data_dict_from_table(subtable, only_means=True)

    data_upper_bound = np.nanmax(star_means, axis=0)
    data_lower_bound = np.nanmin(star_means, axis=0)
    logging.info('Stars span from {} to {}'.format(
        np.round(data_lower_bound),
        np.round(data_upper_bound)
    ))

    # First try and scale box margins on.
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

if __name__ == '__main__':
    assoc_name = sys.argv[1]
    box_lower_bound, box_upper_bound = get_region(assoc_name)
    print('box lower bound: {}\nbox upper bound: {}'.format(box_lower_bound,
                                                            box_upper_bound,))

