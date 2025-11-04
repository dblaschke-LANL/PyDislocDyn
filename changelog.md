# Changelog

## 1.3.4 (wip)

Features and improvements:

 - automatically compile and reload the fortran submodule if it is missing or outdated upon importing pydislocdyn
   (requires user confirmation)
 - new option in pydislocdyn.utilities.compilefortranmodule(): set clean=True to delete all previously compiled module files and exit

Fix:

 - make auto-setting the number of openmp threads more robust: will use threadpoolctl if it is installed (new optional dependency)
 - fix compiler warnings in the fortran subroutines
 - make the test suite platform independent

Other:

 - ompthreads is now a function, not a constant; i.e. it will always show the current number of OpenMP threads (even if the user changed it with e.g. threadpoolctl)
   if the Fortran subroutines are compiled with OpenMP support (and 0 otherwise)

## 1.3.3 (2025-10-06)

Features and improvements:

 - phonon wind: implement `Debye_series` option also for mixed modes, and warn if the temperature is too low to warrant the high temperature series expansion
 - speedup phononwind calculations (if fortran subroutines are used)
 - speedup `find_wavespeed()` method
 - phonon drag: new commandline option 'allplots' to generate additional plots of B(sigma) that are suppressed by default
 - testsuite: add a new test and options for more control over some tests

Fix:

 - fix a `__repr__` string for symbolic `.Vc` and improve related init methods for symbolic lattice constants
 - fix an edge case of `convert_SOECiso()` for symbolic calculations
 - fix a numpy 2 and a numba related bug, prepare for near-future numpy deprecations
 - testsuite: fix and improve behavior of commandline option `--metals=all`
 - `findRayleigh()`: fix for edge cases

Other:

 - changed: use option `maxrec=-1` instead of `None` to bypass adaptive grid in phonon wind calculations
 - the Fortran subroutines have been completed as a (faster) replacement of numba.jit-compiled functions:
   Therefore, numba.jit is no longer used if the Fortran subroutines are compiled (and Numba is only used as a fall-back method)
 - part of the code has been refactored to better separate the two sets of subroutines
 - updated docs

## 1.3.2 (2025-06-30)

Features and improvements:

 - symbolic (sympy) calculations are now supported by a number of functions and class-methods (in addition to purely numeric),
   in this regard, the new `.init_symbols()` method was added to the Dislocation class for convenience
 - in some cases, issue helpful warnings when initializing the Dislocation class
 - more flexibility in averaging elastic constants within the Dislocation class via new option '`scheme`' in `compute_Lame()` method
 - new functions, `convert_SOECiso()` and `convert_TOECiso()`, to convert between isotropic elastic constants added
 - new option in `.computevcrit()` to control output
 - new option in `.readinputfile()` allows adding two character angles beyond the interval (needed for derivatives at the end points)
 - `polycrystal_averaging`: include anisotropy measures in output
 - include additional TOEC data in dictionaries
 - testsuite: more unit tests added
 - improve logic in `compilefortranmodule()`: better handling of different versions/combinations of python, numpy, setuptools, 
   etc., and print helpful error messages when we do fail
 - added a jupyter notebook showing a few examples of how to use PyDislocDyn as a module
 - moved a number of comments in the python code into (sub-)module docstrings as well as `--help` messages of front-end scripts for easy access

Fix:

 - when running in Jupyter, don't choose pgf/latex backend of Matplotlib even if LaTeX is present on the system
 - some small bugfixes

Other:

 - updated docs

## 1.3.1 (2025-01-23)

Features and improvements:

 - implemented the universal log-Euclidean anisotropy index
 - implemented a function to search for the lowest/highest sound speed and (quasi-)shear wave speed in an anisotropic crystal
   (requires scipy 1.9 or higher)
 - speedup `computeuij_acc_edge()` (via `cse=True` option in `lambdify()`: requires sympy>=1.9)
 - `utilities.compilefortranmodule()`: enable passing options to f2py
 - write B(sigma) to files and include in regression testing
 - testsuite: added new unit tests, compress output files for tests acc, misc

Fix:

 - work around a segfault in python 3.13
 - accelerating edge dislocations: work around some numerical edge cases
 - bugfix for edge cases of B(sigma) at high stress

Other:

 - replace legacy `scipy.optimize.fsolve` with `scipy.optimize.root`
 - updated docs

## 1.3.0.1 (2024-07-30)


Fix:

 - add dependency Jinja2 to .toml file
 - correct typos in docs
 - fix for sympy 1.13
 - phononwind calculations: be less noisy with warnings near limiting velocities


## 1.3.0 (2024-07-09)

Features and improvements:

 - major refactoring to separate frontend scripts from the module
 - module and frontend scripts temporarily add themselves to `sys.path` upon importing / running
 - `dragcoeff_iso`: renamed option `use_exp` -> `use_exp_Lame` (for consistency)
 - support reading options from a file (instead of, or in addition to, the commandline by using the new `--fromfile` option)
 - write options to a log file in order to facilitate reproducible runs
 - if available, use LaTeX for the figures (previously commented out)
 - testing script: support setting all options of frontend scripts
 - added some helpful error messages and improved `--help` messages
 - new helper function `compilefortranmodule()` to simplify compilation of the optional Fortran subroutines for users
 - include a pyproject.toml file to support installing PyDislcDyn via '`pip install`'
 - updated the docs

Fix:

 - avoid double imports of submodules
 - fixed some pylint warnings and some small bugs
 - ensure we use the same font in all parts of all plots
 - keep user-provided fractions in Miller indices (instead of converting to float immediately)

Other:

 - drop Python 3.8 support (now require Python 3.9 or higher, this has implications for minimum required versions of other dependencies)

## 1.2.9 (2024-02-29)

Features and improvements:

 - regression testsuite: less clutter with `--verbose=True` option and more tests added
 - refactor some code: `compueuij` separated into `computeuij` (gradient only) and `computeuk` (no gradient) and speedup the latter
 - method `.plotdisloc()` now accepts additional options (including contour levels)
 - more flexibility with some function options by passing `**kwargs`; e.g. simplify passing of phononwind related command line options
 - promote pydislocdyn to a 'regular package':
    * the most commonly used classes, functions and variables are now available from the top level, e.g. upon '`import pydislocdyn`'
    * `pydislocdyn.readinputfile` is an alias to `pydislocdyn.linetension_calcs.readinputfile` (which returns an instance of the `Dislocation` class)
 - new `pydislocdyn.phonondrag()` wrapper function:
    * refactored dragcoeff frontends to make use of it; this also fixed an issue where vcrit was previously not recomputed for higher T 
      prior to determining whether or not to skip a transonic speed.
    * option `--computevcrit_for_speed` was replaced by the more intuitive option `--skiptransonic`
 - include additional data in dictionaries
 - updated the manual

Fix:

 - docs: fixed typos, a dead link, and updated some references
 - bugfix: previously, `computevcrit()` could fail in rare edge cases
 - other small bugfixes

Other:

 - updated docs on how to compile the fortran subroutines to account for changes in build system in python 3.12 and new/upcoming numpy versions
 - future proof for newer numpy versions (i.e. avoid deprecation warning of numpy>=1.25 and use our own `trapz()` implementation since it is unclear what numpy/scipy will do with theirs and what this means for numba support);
   we now require scipy 1.6 or higher

## 1.2.8 (2023-09-12)

Features and improvements:

 - generalized `find_vRF()` to cover more cases
 - accelerating screw dislocations: implemented the supersonic regime
 - generalize `.vcrit_all` attribute of `Dislocation` class
 - `vcrit_*` keywords are no longer supported in input files (was broken)
 - `write_vcrit` option removed from `linetension_calcs`, `.computevcrit()` is fast enough to be used on the fly
 - vcrit.dat and vcrit-plots now show the true limiting velocities (instead of the poles of `det(nn)`) and polar angle phi is therefore no longer included)
 - new function: `writeallinputfiles()` to conveniently convert all dictionary entries into sample input files
 - new function `CheckReflectionSymmetry()` (now used by other parts of the code to check for special cases)
 - `find_vRF`: speed up some edge cases
 - added a testsuite for regression testing (requires pandas 1.1 or higher)
 - nicer LaTeX output of `polycrystal_averaging` script

Fix:

 - avoid deprecation warnings of numba 0.57 by explicitly passing `nopython=True` option
 - `vcrit_all`: correctly handle special cases
 - classes: ensure all attributes that may be set later are initialized to something
 - make autoconfigure ompthreads more robust
 - `findvcrit_smallest()`: don't overwrite `vcrit_all`
 - some bugfixes

Other:

 - clean up some unused code

## 1.2.7 (2023-04-27)

Features and improvements:

 - new commandline option `--help`
 - fortran subroutines: add comments, new subroutine and module for parameters
 - small speedup through code optimization
 - automatically adjust `OMP_NUM_THREADS` for openmp builds of the subroutines module unless set by the user
 - support fractions for Miller indices in input files
 - additional data in dictionaries
 - include metal names in phononwind warnings
 - function `writeinputfile()` can now access all data included in the dictionaries to write sample input files (increased flexibility through new options)
 - refactor: use temporary input files via `writeinputfile()` to avoid duplicate code
 - new method `find_vRF()` implemented in Dislocation class to find 'radiation-free' dislocation velocities (requires sympy 1.6 or higher)
 - calculating accelerating edge dislocation fields implemented
 - `.computesound()`: facilitate optional analytic calculations
 - updated the manual

Fix:

 - avoid 1/0 and don't complain about `arccos(x)=nan` while optimizing
 - print parallelization info after parsing options (not before)
 - fix `plotdisloc()` for accelerating dislocations (regression) and some other small bugs
 - fix reading Boolean options from the commandline
 - minimize rounding errors in `Miller_to_Cart()`
 - improve results of `computevcrit_barnett()` in some edge cases
 - make frontend scripts executable
 - work around a sympy 1.11 bug
 - future-proof for new versions of scipy
 
Other:

 - remove deprecated code: old (slow) subroutine `computevcrit_stroh()` (replaced earlier by much faster `computevcrit_barnett()`)

## 1.2.6 (2022-08-17)

Features and improvements:

 - support additional commandline options
 - new function `plotuij()` and refactored `plotdisloc()` for increased flexibility
 - make the `Dislocation` class easier to use directly (without the `readinputfile()` routine):
    * new defaults for `theta` and `Nphi`
    * allow setting lattice constants and angles when initializing (optional,
       required for some slip systems when entering `b` and `n0` in Miller index notation)
    * make `alignC2()` method work when `C2` contains sympy symbols
    * infer `.burgers` (length) attribute if omitted (from vector `.b`)
 - add `.plotdisloc()` method to the Dislocation class
 - new, much faster implementation of `computevcrit()`
 - new defaults: remove hardcoded vcrit values and compute on the fly instead
 - dictionaries: add more isotropic data
 - tweak plots and skip unnecessary ones
 - automatically compute unit cell volumes Vc in all cases
 - add support for tetragonal II crystals (`sym=tetr2`), allow setting Vc manually for `sym=trig`
 - include a code manual (finally!)

Fix:

 - fix latex for matplotlib >=3.4
 - better handling of NaN in vcrit
 - some (regression) bugfixes and docstring-typos corrected
 - readinputfile(): fix reading angles

## 1.2.5 (2021-12-22)

Features and improvements:

 - refactored: `B_of_sigma()` and related code is now importable and has new options
 - results now written to xz compressed files, use pandas to read
 - support passing options from the command line (no more need to edit the python scripts to run with different
   parameters, resolutions, etc.)

Other:

 - new requirements: pandas, python >=3.8 (implies newer versions are required of some other modules)
 - removed some duplicate / obsolete code
 - changed some defaults: v>vcrit calculations are now skipped by default

## 1.2.4 (2021-09-13)

Features and improvements:

 - new function to compute the Rayleigh wave speed implemented
 - new `strain_poly class` (to determine which type of deformation is sensitive to which elastic constants)
 - support input files also in the isotropic limit (previously only anisotropic)
 - small speedup via more fortran subroutines (optional)
 - updated / added docstrings in various places

Fix:

 - make certain isotropic cases work in `computevcrit_stroh()` function
 - cleaner/more consistent init of `metal_props` class
 - fixed a joblib related regression and a few other small issues

Other:

 - dropped Python 3.5 support (now require Python 3.6 or higher)
 - some code optimizations

## 1.2.3 (2021-07-02)

Features and improvements:

 - automated critical/limiting velocity calculations via new `computevcrit()` function (needs sympy 1.5 or higher)
 - new option to bypass calculations for v>vcrit
 - add bulk/Young's modulus and Poisson's ratio to `Dislocation` class attributes
 - new option '`symmetric`' in `readinputfile()` function
 - speedup calculating the accelerating screw dislocation field
 - some small code optimizations and changed some colors in plots

Fix:

 - avoid some matplotlib warnings

## 1.2.2 (2021-03-20)

Features and improvements:

 - accelerating screw dislocation displacement gradient field was implemented
 - some code refactoring: new `Dislocation` class and new `readinputfile()` function
 - new (optional) plot of dislocation field
 - add OpenMP support in optional fortran subroutines

Fix:

 - some minor bugfixes
 - temporarily add our path to `sys.path` (workaround for Spyder's runfile() command)

Other:

 - renamed some angle variables

## 1.2.1 (2020-10-10)

Features and improvements:

 - additional crystal symmetries implemented ('trig', 'orth', 'mono', and 'tric')
 - we now support Miller indices (previously only Cartesian coordinates)
 - new / nicer plots
 - some new options (including a debug option in `computeuij` that will output various intermediate results of the integral method)

Fix:

 - fix B(sigma) plots for some edge cases
 - some (regression) bugfixes (including previously buggy pyramidal slip in hcp)
 
Other:

 - support for Python 2.7 has been dropped (now require Python 3.5 or higher)
 - support scipy 1.5 (i.e. avoid a deprecation warning)

## 1.2.0 (2020-05-02)

Features and improvements:

 - major code rewrite / refactor to allow input-file driven calculations (without the need to edit the python scripts)
 - output additional and nicer plots
 - support additional (bcc) slip systems
 - automatically normalize `b` and `n0`
 - new function to compute sound speeds in the crystal along direction v
 - improved on-the-fly estimates of limiting velocities (used for plots)
 - support new versions of numba and matplotlib 3.2 (i.e. avoid deprecation warnings of those libraries)

Fix:

 - some bugfixes

## 1.1.0 (2019-12-20)

Features and improvements:

 - improved accuracy of phonon wind calculations via (partially) adaptive grid for numerical integration
 - new plots: B as a function of stress
 - new options for choosing (and computing on the fly) the normalization used for linetension and beta in phonon wind calculations
 - linetension: new output format, decoupled plots from calculations
 - speedup via more fortran subroutines (optional) and rewrite of some python subroutines (elasticC2,3)
 
Fix:

 - some bugfixes

## 1.0.3 (2019-09-20)

Features and improvements:

 - automated the generation of rotation matrices
   which align the coordinates with the dislocation line/slip plane normal
   thereby simplifying the future implementation of additional slip systems
 - new options to return the dislocation displacement field (instead of its gradient)
   and/or to include the radius explicitly in the array
 - new options to bypass plotting and/or calculations (in which case only previous results are loaded)
 - determine Young's modulus in dictionaries
 - refactored / optimized some code: 2 new classes (`Strohgeometry` and `IsoInvariants`) and speedup

Fix:

 - correctly handle the `beta=0` case in the isotropic limit
 - improve stability of plotting functions for edge cases
 - made joblib and numba optional dependencies (code now works even if one or both of these are missing)

## 1.0.2 (2019-03-21)

Features:

 - output additional nice plots
 - determine Zener ratios in dictionaries
 - phonon wind calculations: implement optional dislocation core cutoff
 
Fix:

 - a number of small bugfixes
 - some code optimization
 
Other:

 - update license file

## 1.0.1 (2018-10-09)

Features:

 - speedup phonon wind calculations (new optional Fortran subroutines via f2py)
 - phonon wind calculations: accept commandline arguments
 - add more metals to dictionaries
 - support more slip systems: hcp-prismatic and pyramidal

## 1.0.0 (2018-07-13)

Initial release:

 - calculate the drag coefficient from phonon wind (isotropic and semi-isotropic)
 - determine line tension of straight, steady-state dislocations (arbitrary character angle)
 - average elastic constants
 - supported crystal geometries / slip systems: fcc, (110)-planes in bcc, basal slip in hcp, one example slip system in tetragonal
