"""
Profile groupfitter

See where the bulk of computation occurs.

Examples on how to profile with python
https://docs.python.org/2/library/profile.html
"""

import cProfile
import logging
import numpy as np
import pstats
import sys

sys.path.insert(0, '..')
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import compfitter

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename='compfitter.log')
    save_dir = ''
    group_savefile = save_dir + 'origins_stat.npy'
    xyzuvw_init_savefile = save_dir + 'xyzuvw_init_stat.npy'
    astro_savefile = save_dir + 'astro_table_stat.txt'
    xyzuvw_conv_savefile = save_dir + 'xyzuvw_conv_stat.fits'

    pars = np.array([0., 0., 0., 0., 0., 0., 5., 2., 1e-8])
    starcount = 100
    error_frac = 1.
    synth_data = SynthData(pars=pars, starcounts=starcount)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)
    data = tabletool.build_data_dict_from_table(synth_data.table)

    stat_file = 'compfitter.stat'
    # best_fit, chain, lnprob = \
    cProfile.run(
        "compfitter.fit_comp(data=data, plot_it=True,"
        "convergence_tol=2., burnin_steps=400, plot_dir='',"
        "save_dir='')",
        stat_file,
    )

    stat = pstats.Stats(stat_file)
    stat.sort_stats('cumtime')
    stat.print_stats(0.3)




