"""
Print data for table `componentsfit.tex` in the paper
"""

import numpy as np
from astropy.table import Table


############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
comps_filename = lib.comps_filename
############################################
comps = Table.read(comps_filename)

for c in comps:
    comp_id = c['comp_ID']
    age=c['age']

    print('%s & %.1f & %.1f & %.1f & %.2f & %.2f & %.2f & %.1f & %.2f & %.1f \\\\'%(c['comp_ID'], c['X'], c['Y'], c['Z'], c['U'], c['V'], c['W'], c['dX'], c['dV'], c['Age']))
    
 
