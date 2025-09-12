#!/usr/bin/env python3
# PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Jan. 8, 2024 - Sept. 12, 2025
'''PyDislocDyn is a suite of python programs designed to perform various calculations for basic research
   in dislocation dynamics in metals with various crystal symmetries in the continuum limit. In particular,
   one of its main purposes is to calculate dislocation drag from phonon wind. Additional features include
   the averaging of elastic constants for polycrystals, the calculation of the dislocation field including
   its limiting velocities, and the calculation of dislocation self-energy and line tension.'''

from .metal_data import writeinputfile, writeallinputfiles
from .utilities import usefortran, ompthreads, nonumba, Ncores, Ncpus, read_2dresults, roundcoeff, plotuij
from .elasticconstants import Voigt, UnVoigt, CheckVoigt, strain_poly, elasticC2, elasticC3, \
    elasticS2, elasticS3, CheckReflectionSymmetry, convert_SOECiso, convert_TOECiso
from .crystals import IsoAverages, metal_props, Miller_to_Cart
from .dislocations import StrohGeometry, Dislocation, readinputfile
from .phononwind import elasticA3, phonondrag, B_of_sigma

__version__ = '1.3.2+dev'

if usefortran:
    from .utilities import fsub as subroutines
# else:
#     from .utilities import compilefortranmodule
#     user_input = input('Fortran submodule could not be loaded; run pydislocdyn.utilities.compilefortranmodule() now? [Y/N] ')
#     if user_input.lower() in ('y', 'yes'):
#         fortran_error = compilefortranmodule()
#         if fortran_error==0:
#             print("\n\nSUCCESS! Please reload pydislocdyn to use the submodule.")
