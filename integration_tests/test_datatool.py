"""
datatool is the module used to prepare a dataset for
input into Chronostar. Here we run a few simple tests
on small, contrived datasets.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar import datatool
from chronostar import synthdata
from chronostar import tabletool

PARS =  np.array([
    [  0., 0., 0., 0., 0., 0., 10., 5., 1e-5],
    [100., 0., 0., 0., 0., 0., 10., 5., 1e-5]
])
STARCOUNTS = [50, 30]
DIMS = 'XYZUVW'
def test_convert_astrometry():
    """
    Use a synethetically generated set of astrometry, convert
    to cartesian both manually, and via datatool

    In order to compare results, the datatool cartesian conversion
    will be stored in alternatively named columns
    """
    synth_table_filename = 'temp_data/test_convert_astrometry_data.fits'

    synth_dataset = synthdata.SynthData(pars=PARS, starcounts=STARCOUNTS)

    synth_dataset.synthesise_everything(filename=synth_table_filename,
                                        overwrite=True)

    tabletool.convert_table_astro2cart(synth_table_filename,
                                       write_table=True,
                                       filename=synth_table_filename)

    # Prepare a pars file
    par_file = 'temp_data/test_convert_astrometry.par'

    alt_cart_main_colnames = ['{}_alt'.format(dim) for dim in DIMS]
    alt_cart_error_colnames = ['{}_error_alt'.format(dim) for dim in DIMS]
    alt_cart_corr_colnames = []
    for i, colname1 in enumerate(DIMS):
        for colname2 in DIMS[i + 1:]:
            alt_cart_corr_colnames.append('{}_{}_corr_alt'.format(
                    colname1, colname2
            ))

    with open(par_file, 'w') as fp:
        fp.write('par_log_file = temp_data/test_convert_astrometry_pars.log\n')
        fp.write('input_file = {}\n'.format(synth_table_filename))

        fp.write('convert_astrometry = True\n')

        fp.write('{} = {}\n'.format('cart_main_colnames',
                                    alt_cart_main_colnames).replace("'",''))
        fp.write('{} = {}\n'.format('cart_error_colnames',
                                    alt_cart_error_colnames).replace("'",''))
        fp.write('{} = {}\n'.format('cart_corr_colnames',
                                    alt_cart_corr_colnames).replace("'",''))

        fp.write('overwrite_datafile = True\n')
        fp.write('output_file = {}\n'.format(synth_table_filename))
        fp.write('return_data_table = True\n')

    # Apply datatool to synthetically generated dataset
    data_table = datatool.prepare_data(par_file)

    main_colnames, error_colnames, corr_colnames = tabletool.get_colnames(cartesian=True)

    for orig, alt in zip([main_colnames, error_colnames, corr_colnames],
                         [alt_cart_main_colnames, alt_cart_error_colnames,
                          alt_cart_corr_colnames]):

        for orig_colname, alt_colname in zip(orig, alt):
            assert np.allclose(data_table[orig_colname],
                               data_table[alt_colname], rtol=1e-5)
            print(np.max(np.abs(data_table[alt_colname] - data_table[orig_colname])))

def test_get_region():
    """
    Test whether get_region applies data cut successfully.

    Synthesise two data sets, one which made up of Component A, and the other
    made up of Component A & B. Then, check if applying a get_region cut on
    the combined data set, with the Component A set as reference, only returns
    the Component A stars.
    """
    data_a_filename = 'temp_data/test_get_region_A.fits'
    synth_dataset_a = synthdata.SynthData(pars=PARS[0],
                                          starcounts=STARCOUNTS[0])
    np.random.seed(0)
    synth_dataset_a.synthesise_everything(filename=data_a_filename,
                                          overwrite=True)
    tabletool.convert_table_astro2cart(synth_dataset_a.table, write_table=True,
                                       filename=data_a_filename)

    data_both_filename = 'temp_data/test_get_region_both.fits'
    synth_dataset_both = synthdata.SynthData(pars=PARS,
                                             starcounts=STARCOUNTS)
    np.random.seed(0)
    synth_dataset_both.synthesise_everything(filename=data_both_filename,
                                             overwrite=True)

    # Prepare .par file
    par_file = 'temp_data/test_get_region.par'

    with open(par_file, 'w') as fp:
        fp.write('par_log_file = temp_data/test_get_region_pars.log\n')
        fp.write('input_file = {}\n'.format(data_both_filename))

        fp.write('convert_astrometry = True\n')

        fp.write('apply_cart_cuts = True\n')
        fp.write('cut_on_region = True\n')
        fp.write('cut_ref_table = {}\n'.format(data_a_filename))

        # fp.write('output_file = {}\n'.format())
        fp.write('return_data_table = True\n')

    # Apply datatool to synthetically generated dataset
    data_table = datatool.prepare_data(par_file)

    assert len(data_table) == len(synth_dataset_a.table)

if __name__=='__main__':
    pass


