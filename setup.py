# coding: utf-8

# run with
# python3 setup.py build_ext -b .

# Notes for MZ: if gsl is not found:
# reinstall python3
# reinstall gsl
# export LIBRARY_PATH=/usr/local/Cellar/gsl/2.7/lib/

""" A stellar orbit traceback code """

import os
import re
import sys

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

try:
    from setuptools import Extension

except ImportError:
    from distutils.core import Extension


major, minor1, minor2, release, serial =  sys.version_info

readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}

def readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents

version_regex = re.compile("__version__ = \"(.*?)\"")
contents = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chronostar",
    "__init__.py"))

version = version_regex.findall(contents)[0]

#Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory. This logic works across numpy versions
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# &TC added extra directory
_overlap = Extension("chronostar/_overlap",
                    ["chronostar/overlap/overlap.i", "chronostar/overlap/overlap.c"],
                    include_dirs = [numpy_include],
                    libraries = ['gsl', 'gslcblas'], 
#https://stackoverflow.com/questions/44380459/is-openmp-available-in-high-sierra-llvm
#... but no obvious compile errors. 
#                   extra_compile_args = ["-Xclang -fopenmp -lomp"],
# MJI Remove these flags that now make the default OSX compiler break.
#                   extra_compile_args = ["-floop-parallelize-all","-ftree-parallelize-loops=4"],
                    )

# MZ
# Note: This setup works with python3, but fails with python2.7
_expectation = Extension("chronostar/_expectation",
                    ["chronostar/c/expectation.i", 
                    "chronostar/c/expectation.c"],
                    include_dirs = [numpy_include, '/usr/local/include/'],
                    libraries = ['gsl', 'gslcblas'],
                    )

_temporal_propagation = Extension("chronostar/_temporal_propagation",
                    ["chronostar/c/temporal_propagation.i", 
                    "chronostar/c/temporal_propagation.c"],
                    include_dirs = [numpy_include, '/usr/local/include/'],
                    libraries = ['gsl', 'gslcblas'],
                    library_dirs = ['/usr/local/include/'],
                    )

_likelihood = Extension("chronostar/_likelihood",
                    ["chronostar/c/likelihood.i", 
                    "chronostar/c/likelihood.c",
                    "chronostar/c/expectation.c",                   
                    "chronostar/c/temporal_propagation.c"],
                    include_dirs = [numpy_include, '/usr/local/include/'],
                    libraries = ['gsl', 'gslcblas'],
                    )

setup(name="chronostar",
      version=version,
      author="Michael J. Ireland",
      author_email="michael.ireland@anu.edu.au",
      packages=["chronostar"],
      url="http://www.github.com/mikeireland/chronostar/",
      license="MIT",
      description="A stellar orbit traceback code.",
      long_description=readfile(os.path.join(os.path.dirname(__file__), "README.md")),
      install_requires=[
        "requests",
        "requests_futures"
      ],
      ext_modules = [_overlap, _expectation, _temporal_propagation,
      _likelihood],
     )
