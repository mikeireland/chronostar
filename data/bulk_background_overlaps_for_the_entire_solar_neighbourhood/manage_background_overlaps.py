"""
Author: Marusa Zerjal, 2019 - 08 - 21

Manage background overlaps:
Prepare list of stars that don't have component overlaps yet
Compute bg ols with the external code (multiprocessing)
Insert bg ols back into the file
"""

import numpy as np
from astropy.table import Table
import sys

sys.path.insert(0, '/Users/marusa/chronostar/')
from chronostar import tabletool
from astropy.table import Table, vstack


def prepare_stars_that_need_bg_ols():
    """
    Bg ols for some stars have been computed in the earlier analyses. Use these bg ols and insert them into the table.
    Find stars that don't have bg ols yet.
    """

    wanted = Table.read(
        '/priv/mulga1/marusa/chronostar_projects/solar_neighbourhood/data/ScoCen_box_result_15M_ready_for_bg_ols.fits')

    old = Table.read('../scocen/data/data_table_cartesian_including_tims_stars_with_bg_ols_and_component_overlaps.fits')
    old_scocen = Table.read(
        '/priv/mulga1/marusa/chronostar_projects/scocen/data/scocen_candidates_300k_only_spatial_cut.fits')

    old_solar_neighbourhood_bg_ols = np.loadtxt('bgols_multiprocessing_0.dat')
    wanted0 = wanted[:len(old_solar_neighbourhood_bg_ols)]

    # DELETE THIS!!!
    wanted = wanted[len(old_solar_neighbourhood_bg_ols):]

    mask = np.in1d(wanted['source_id'], old['source_id'])
    mask = np.logical_or(mask, np.in1d(wanted['source_id'], old_scocen['source_id']))
    # ~ mask = np.logical_or(mask, np.in1d(wanted['source_id'], old_solar_neighbourhood['source_id']))

    # Looking for stars that do NOT have bg ols yet
    mask = ~mask

    todo = wanted[mask]
    print
    len(todo)
    print
    len(old), len(wanted), len(wanted) - len(old)

    todo.write('solar_neighbourhood_determine_bg_ols_for_these_stars.fits', format='fits')


def match_bg_ols_from_textfile_and_sobject_id():
    n = [0, 1, 2, 3]  # filenumbers

    datafile = 'solar_neighbourhood_determine_bg_ols_for_these_stars.fits'
    data0 = Table.read(datafile)
    N = 10  # that many chunks. DON'T CHANGE THIS, this number should be the same as in the bg_ols_multiprocessing.py!!
    indices_chunks = np.array_split(range(len(data)), N)

    for NI in n:
        bg_ols_filename = 'bgols_multiprocessing_round2_%d.dat' % NI
        bgols = np.loadtxt(bg_ols_filename)

        data = data0[indices_chunks[NI]]
        ids = data['source_id']

        print
        len(bgols), len(ids)

        if NI == 0:
            tab = Table([ids], names=['source_id'])
            tab['background_log_overlap'] = bgols
        else:
            tab0 = Table([ids], names=['source_id'])
            tab0['background_log_overlap'] = bgols

            tab = vstack([tab, tab0])

    print tab

def insert_bg_ols_into_table():
    bg_ols_filename = 'bgols_multiprocessing_0.dat'
    ln_bg_ols = np.loadtxt(bg_ols_filename)

    #    for

    datafile = 'data/ScoCen_box_result_15M_ready_for_bg_ols.fits'
    data_table = Table.read(datafile)

    bg_lnol_colname = 'background_log_overlap'
    print('Background overlaps: insert column')
    tabletool.insert_column(data_table, ln_bg_ols, bg_lnol_colname, filename=datafile)

    print('Print bg ols to cartesian table')
    data_table.write(datafile, overwrite=True, format='fits')


def merge_both_tables():
    """
    Use bg ols from 'old' and 'additional' to insert into 'wanted'.
    """
    old = Table.read('data/data_table_cartesian_including_tims_stars_with_bg_ols_and_component_overlaps.fits')
    wanted = Table.read('data/scocen_candidates_300k_only_spatial_cut.fits')
    additional = Table.read('data/scocen_candidates_300k_only_spatial_cut_200k_to_determine_bg_ols.fits')

    d_old = dict(zip(old['source_id'], old['background_log_overlap']))
    d_add = dict(zip(additional['source_id'], additional['background_log_overlap']))
    d_old.update(d_add)
    dct = d_old

    ln_bg_ols = [dct[source_id] for source_id in wanted['source_id']]
    print
    len(ln_bg_ols), len(wanted)

    wanted['background_log_overlap'] = ln_bg_ols
    print
    wanted

    wanted.write('data/scocen_candidates_300k_only_spatial_cut.fits', overwrite=True, format='fits')


if __name__ == '__main__':
    prepare_stars_that_need_bg_ols()
    # ~ insert_bg_ols_into_table()
    # merge_both_tables()
