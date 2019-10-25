#! /usr/bin/env python
"""
A helper script that prepares data for input into Chronostar.
Accepts as a command line argument a path to a parameter file
(see README.md for how to structure the parameter file and
various available parameters).
"""

import os.path
import sys
sys.path.insert(0, '..')

from chronostar import datatool

if len(sys.argv) != 2:
    raise UserWarning('Incorrect usage. Path to parameter file is required'
                      ' as a single command line argument. e.g.\n'
                      '   > python prepare_data.py path/to/parsfile.par')

data_par_file = sys.argv[1]

if not os.path.isfile(data_par_file):
    raise UserWarning('Provided file does not exist')

datatool.prepare_data(data_par_file)
