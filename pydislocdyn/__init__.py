#!/usr/bin/env python3
# PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Jan. 8, 2024 - Nov. 22, 2025
'''PyDislocDyn is a suite of python programs designed to perform various calculations for basic research
   in dislocation dynamics in metals with various crystal symmetries in the continuum limit. In particular,
   one of its main purposes is to calculate dislocation drag from phonon wind. Additional features include
   the averaging of elastic constants for polycrystals, the calculation of the dislocation field including
   its limiting velocities, and the calculation of dislocation self-energy and line tension.'''

from . import utilities

if utilities.usefortran:
    from .utilities import fsub as subroutines
else:
    import pathlib
    import sys
    fname  = pathlib.Path(__file__).resolve().parent / f"fmoderror_py{sys.version_info[0]}.{sys.version_info[1]}.txt"
    f2py_error = None
    if fname.is_file():
        with open(fname,"r", encoding="utf8") as f1:
            f2py_error = int(f1.readlines()[0])
        del f1
    if f2py_error is None:
        from .utilities import compilefortranmodule
        user_input = input('Fortran submodule could not be loaded; run pydislocdyn.utilities.compilefortranmodule() now? [Y/n] ')
        if user_input.lower() in ('y', 'yes', ''):
            f2py_error = compilefortranmodule()
            if f2py_error==0:
                print("\n\nSUCCESS!")
                import importlib
                importlib.reload(utilities)
        else:
            print("user declined, will not ask again; call pydislocdyn.utilities.compilefortranmodule() to compile manually")
            with open(fname,"w", encoding="utf8") as f1:
                f2py_error = 0
                f1.write(f"{f2py_error}") # don't ask again if user declined
            del f1
    del fname

from .metal_data import writeinputfile, writeallinputfiles
from .utilities import usefortran, ompthreads, nonumba, Ncores, Ncpus, read_2dresults, roundcoeff, plotuij
from .elasticconstants import Voigt, UnVoigt, CheckVoigt, strain_poly, elasticC2, elasticC3, \
    elasticS2, elasticS3, CheckReflectionSymmetry, convert_SOECiso, convert_TOECiso
from .crystals import IsoAverages, metal_props, Miller_to_Cart
from .dislocations import StrohGeometry, Dislocation, readinputfile
from .phononwind import elasticA3, phonondrag, B_of_sigma

__version__ = '1.3.4+dev'
