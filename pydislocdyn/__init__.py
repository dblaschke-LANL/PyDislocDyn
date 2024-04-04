#!/usr/bin/env python3
# PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Jan. 8, 2024 - Apr. 4, 2024
'''PyDislocDyn is a suite of python programs designed to perform various calculations for basic research
   in dislocation dynamics in metals with various crystal symmetries in the continuum limit. In particular,
   one of its main purposes is to calculate dislocation drag from phonon wind. Additional features include
   the averaging of elastic constants for polycrystals, the calculation of the dislocation field including
   its limiting velocities, and the calculation of dislocation self-energy and line tension.'''

from .metal_data import writeinputfile, writeallinputfiles
from .utilities import usefortran, ompthreads, nonumba, Ncores, Ncpus, read_2dresults
from .elasticconstants import Voigt, UnVoigt, CheckVoigt, strain_poly, elasticC2, elasticC3, \
    elasticS2, elasticS3, CheckReflectionSymmetry
from .crystals import IsoAverages, metal_props
from .dislocations import StrohGeometry, Dislocation, plotuij, readinputfile
from .phononwind import elasticA3, phonondrag, B_of_sigma

__version__ = '1.2.9+dev'

if usefortran:
    from .utilities import fsub as subroutines
