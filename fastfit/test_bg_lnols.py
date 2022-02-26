import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os.path
import sys
sys.path.insert(0, '..')

from chronostar import tabletool

try:
    from chronostar._expectation import print_bg_lnols
except ImportError:
    print("C IMPLEMENTATION OF expectation NOT IMPORTED")
    USE_C_IMPLEMENTATION = False
    TODO = True # NOW WHAT?

# Read data
data_dict = tabletool.build_data_dict_from_table('data_for_testing.fits', 
    get_background_overlaps=True)

# Take bg_lnols from data_dict['bg_lnols']
bg_lnols = data_dict['bg_lnols']
bg_lnols = np.array(bg_lnols)    
print('type(bg_lnols)', type(bg_lnols), bg_lnols.shape)
#~ print('python:')
#~ print(bg_lnols)

# Print bg_lnols in a C code
print('bg_lnols in C')
print_bg_lnols(bg_lnols)

print('')
print('bg_lnols in python')
print(bg_lnols[:10])
print('')
 

# Save bg_ols in a dat file and read it from there
filename = 'bg_lnols.dat'
np.savetxt(filename, bg_lnols)
bg_lnols = np.array(np.loadtxt(filename))
print('type(bg_lnols)', type(bg_lnols), bg_lnols.shape)
print('')

# Print bg_lnols in a C code
print('bg_lnols in C')
print_bg_lnols(bg_lnols)

print('bg_lnols in python')
print(bg_lnols[:10])
