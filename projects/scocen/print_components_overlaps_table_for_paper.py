"""
Print data for table `componentsoverlaps.tex` in the paper

Numbers of members
Ages, crossing time

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
good_comps = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'T', 'U']
############################################
# Minimal probability required for membership
pmin_membership = 0.5
pmin_memberships = [0.5, 0.8, 0.9]
############################################

tab = Table.read(data_filename)
comps = Table.read(comps_filename)

total50=0
total80=0
total90=0

for c in comps:
    comp_ID = c['comp_ID']
    
    if comp_ID not in good_comps:
        continue
    
    line = '%s & '%c['comp_ID']
    
    for pmin_membership in pmin_memberships:
        mask=tab['membership%s'%comp_ID]>pmin_membership 
        t=tab[mask]

        line += '%d & '%len(t)
        
        if comp_ID not in ['H', 'I', 'B']: # Not ScoCen
            if pmin_membership==0.5:
                total50+=len(t)
            elif pmin_membership==0.8:
                total80+=len(t)
            elif pmin_membership==0.9:
                total90+=len(t)
        
        

    age = c['Age']
    crossing_time = c['Crossing_time']
    
    line += '%d & %d '%(age, crossing_time)

    line += ' \\\\'


    print(line)

line = 'Total & %d & %d & %d & & \\\\'% (total50, total80, total90)

print(line)

