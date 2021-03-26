"""
Print data for table `componentsoverlaps.tex` in the paper
"""

import numpy as np
from astropy.table import Table

############################################
# Some things are the same for all the plotting scripts and we put
# this into a single library to avoid confusion.
import scocenlib as lib
data_filename = lib.data_filename
comps_filename = lib.comps_filename
compnames = lib.compnames
############################################
# Minimal probability required for membership
pmin_membership = 0.5
############################################

tab = Table.read(data_filename)
comps = Table.read(comps_filename)

for c in comps:
    comp_id = c['comp_ID']

    mask=tab['membership%s'%comp_id]>pmin_membership 
    t=tab[mask]


    comment = ''
    try:
        comment = compnames[comp_id]
    except:
        pass
    
    if comp_id in bg_comps:
        comment = 'Background'
    

    print('%s & %d & %s \\\\'%(c['comp_ID'], len(t), comment))

