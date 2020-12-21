"""
README
Before you run this, uncomment @profile in likelihood just before the 
def get_lnoverlaps(comp, data, star_mask=None) line.

Then run with
kernprof -l kern_prof_compfitter.py ../scripts/beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits


Read output with 
python -m line_profiler kern_prof_compfitter.py.lprof
"""

import numpy as np

from chronostar import compfitter

from chronostar import datatool
from chronostar import tabletool
from chronostar import traceorbit

import sys

if __name__ == '__main__':
    data_file = sys.argv[1]
    
    trace_orbit_func = traceorbit.trace_epicyclic_orbit

    data_dict = tabletool.build_data_dict_from_table(data_file)
    res = compfitter.fit_comp(data_dict, optimisation_method='Nelder-Mead', trace_orbit_func=trace_orbit_func)

    # nstars = len(data_dict['means'])
    # memb_prob = np.ones(nstars)


