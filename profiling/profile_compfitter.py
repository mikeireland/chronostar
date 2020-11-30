import numpy as np
import cProfile
import logging
import pstats
import sys

from chronostar import compfitter

from chronostar import datatool
from chronostar import tabletool
from chronostar import traceorbit

import sys

if __name__ == '__main__':
    data_file = sys.argv[1]
    logging.basicConfig(level=logging.INFO, filename='compfitter.log')

    data_dict = tabletool.build_data_dict_from_table(data_file, historical=True)
    # res = compfitter.fit_comp(data_dict)

    stat_file = 'compfitter.stat'
    cProfile.run(
            "from chronostar import traceorbit; "
            "print('Imported!');"
            "compfitter.fit_comp(data=data_dict,"
            "optimisation_method='Nelder-Mead', "
            "trace_orbit_func=traceorbit.trace_epicyclic_orbit,"
            ")",
            stat_file,
    )

    stat = pstats.Stats(stat_file)
    stat.sort_stats('cumtime')
    stat.print_stats(0.3)
