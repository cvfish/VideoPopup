#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# inplace extension module
_myfilter = Extension("_myfilter",
                      ["myfilter.i","myfilter.cpp"],
                      include_dirs = [numpy_include],
                      #                   extra_compile_args = ["-std=gnu++11","-Wextra","-O3", "-mfpmath=sse", "-msse4.2" ,"-ffast-math", "-funroll-loops", "-march=native", "-fomit-frame-pointer"],
                      #                    extra_compile_args = ["-Wextra","-O3", "-mfpmath=sse", "-msse4.2" ,"-ffast-math", "-funroll-loops", "-march=native", "-fomit-frame-pointer"],
                   )

# NumyTypemapTests setup
setup(  name        = "Filter algorithms ",
        description = "Provides bilatterial image filter.",
        author      = "Rui Yu",
        version     = "0.01",
        ext_modules = [_myfilter]
        )

import shutil
import os
shutil.copy('build/'+os.listdir('build')[1]+'/_myfilter.so','.')
