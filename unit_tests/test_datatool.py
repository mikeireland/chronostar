'''
Apply prepare_data on a bunch of parameter files (stored in
sample_par_files) and confirm the resulting table has been modified
in the correct way.
'''
from astropy.table import Table
import numpy as np
import os.path
import pytest
import sys
sys.path.insert(0, '..')

from chronostar import datatool

def test_convert_astro_only():
    data_pars = {
        'input_file':'sample_data/sample_table_astro_only.fits',
        'convert_astrometry':True,
        'output_file':'temp_data/astro_only_output.fits',
        'overwrite_datafile':True,
        'return_data_table':True,
    }
    result = datatool.prepare_data(data_pars)

    orig_table = Table.read(data_pars['input_file'])

    assert len(result) == len(orig_table)
    assert 'X' in result.colnames


def test_cut_on_region():
    '''
    Sets a single star in the mock reference table to be the
    set around which to construct out 6D box.
    This has the effect of enforcing the dimensions of the box to be
    at most double the margins (totalling 60 pc in psoition and 10 km/s in
    velocity)
    '''

    ref_table = Table.read('sample_data/small_current_sample_table.fits')
    mock_assoc_colname = 'moving_groups'
    mock_assoc_label = 'bpmg'

    ref_table[mock_assoc_colname] = 10*' '
    ref_table[mock_assoc_colname][0] = mock_assoc_label

    data_pars = {
        'input_file':'sample_data/sample_table_astro_only.fits',
        'convert_astrometry':True,
        'output_file':'temp_data/astro_only_output.fits',
        'overwrite_datafile':True,
        'apply_cart_cuts':True,
        'cut_on_region':True,
        'cut_ref_table':ref_table,
        'cut_colname':mock_assoc_colname,
        'cut_assoc_name':mock_assoc_label,
        'return_data_table':True,
    }

    result = datatool.prepare_data(data_pars)

    pos_dims = 'XYZ'
    for pos_dim in pos_dims:
        assert np.max(result[pos_dim]) - np.min(result[pos_dim]) <= 60.

    vel_dims = 'UVW'
    for vel_dim in vel_dims:
        assert np.max(result[vel_dim]) - np.min(result[vel_dim]) <= 10.

def test_cut_on_bounds():
    '''
    Applies a data cut based on provided cartesian boundaries.
    '''
    DMIN = -10.
    DMAX = 50.

    data_pars = {
        'input_file':'sample_data/sample_table_astro_only.fits',
        'convert_astrometry':True,
        'output_file':'temp_data/astro_only_output.fits',
        'overwrite_datafile':True,
        'apply_cart_cuts':True,
        'cut_on_bounds':True,
        'cut_bound_min':[DMIN,DMIN,DMIN,DMIN,DMIN,DMIN],
        'cut_bound_max':[DMAX,DMAX,DMAX,DMAX,DMAX,DMAX],
        'return_data_table':True,
    }

    result = datatool.prepare_data(data_pars)

    for col in 'XYZUVW':
        assert np.min(result[col]) >= DMIN
        assert np.max(result[col]) <= DMAX


@pytest.mark.skipif(not os.path.isfile('../data/gaia_cartesian_full_6d_table.fits'),
                    reason='No provided background data reference file. Ask Tim'
                           'for "gaia_cartesian_full_6d_table.fits"')
def test_bg_overlaps():
    data_pars = {
        'input_file':'sample_data/sample_table_astro_only.fits',
        'convert_astrometry':True,
        'output_file':'temp_data/astro_only_output.fits',
        'overwrite_datafile':True,
        'calc_overlaps':True,
        'bg_col_name':'background_log_overlap',
        'bg_ref_table':'../data/gaia_cartesian_full_6d_table.fits',
        'return_data_table':True,
    }

    result = datatool.prepare_data(data_pars)

    assert not np.any(np.isnan(result[data_pars['bg_col_name']]))

def test_attempted_overwrite():
    # TODO: test the case where this fails while everything else is valid
    data_pars = {
        'input_file':'not/a/real/file.fits',
        'apply_cart_cuts':True,
    }
    data_pars['output_file'] = data_pars['input_file']
    try:
        datatool.prepare_data(data_pars)
    except UserWarning:
        pass

def test_param_logfile_usable():
    """
    Confirm that the logfile also serves as a valid input file,
    identical to the file used to initialise it.
    TODO: implement
    """
    pass
