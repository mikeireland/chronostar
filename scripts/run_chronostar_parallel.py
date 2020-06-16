#! /usr/bin/env python
"""
A helper script that performs a brute force Chronostar fit to
some pre-prepared data.

Accepts as a command line argument a path to a parameter file
TODO: Update README.md with description for NaiveFit parameters
(see README.md for how to structure the parameter file and
various available parameters).
"""

import os.path
import sys
sys.path.insert(0, '..')

from chronostar.naivefit import NaiveFit

if len(sys.argv) != 2:
    raise UserWarning('Incorrect usage. Path to parameter file is required'
                      ' as a single command line argument. e.g.\n'
                      '   > python new_run_chronostar.py path/to/parsfile.par')

fit_par_file = sys.argv[1]

if not os.path.isfile(fit_par_file):
    raise UserWarning('Provided file does not exist')

naivefit = NaiveFit(fit_pars=fit_par_file)
naivefit.run_fit_parallel()
