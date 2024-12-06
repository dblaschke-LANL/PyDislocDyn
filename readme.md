# PyDislocDyn

PyDislocDyn is a suite of python programs designed to perform various calculations for basic research in dislocation dynamics in metals with various crystal symmetries in the continuum limit.
In particular, one of its main purposes is to calculate dislocation drag from phonon wind.
Additional features include the averaging of elastic constants for polycrystals, the calculation of the dislocation field including its limiting velocities, and the calculation of dislocation self-energy and line tension.
</br>
This code was first used for the computations leading to [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) and [Materials 12 (2019) 948](https://doi.org/10.3390/ma12060948) ([arxiv.org/abs/1902.02451](https://arxiv.org/abs/1902.02451)),
as well as parts of [Int. J. Plast. 131 (2020) 102750](https://doi.org/10.1016/j.ijplas.2020.102750) ([arxiv.org/abs/1912.08851](https://arxiv.org/abs/1912.08851)),
[J. Mech. Phys. Solids 152 (2021) 104448](https://dx.doi.org/10.1016/j.jmps.2021.104448) ([arxiv.org/abs/2009.00167](https://arxiv.org/abs/2009.00167)),
[Int. J. Plast. 144 (2021) 103030](https://doi.org/10.1016/j.ijplas.2021.103030) ([arxiv.org/abs/2101.10497](https://arxiv.org/abs/2101.10497)), [J. Appl. Phys. 130 (2021) 015901](https://doi.org/10.1063/5.0054536) ([arxiv.org/abs/2104.08650](https://arxiv.org/abs/2104.08650)),
 [J. Phys.: Cond. Mat. 33 (2021) 503005](https://doi.org/10.1088/1361-648X/ac2970) ([arxiv.org/abs/2107.01220](https://arxiv.org/abs/2107.01220)),
[Materials 16 (2023) 4019](https://doi.org/10.3390/ma16114019) ([arxiv.org/abs/2303.10461](https://arxiv.org/abs/2303.10461)), and 
[Phys. Rev. B 108 (2023) 224102](https://doi.org/10.1103/PhysRevB.108.224102) ([arxiv.org/abs/2305.06980](https://arxiv.org/abs/2305.06980)).
Additionally, it is able to reproduce the earlier results of LA-UR-16-24559 ([doi.org/10.2172/1434423](https://doi.org/10.2172/1434423)), [J. Appl. Phys. 122 (2017) 145110](https://doi.org/10.1063/1.4993443) ([arxiv.org/abs/1706.07132](https://arxiv.org/abs/1706.07132)), and [Phil. Mag. 98 (2018) 2397&ndash;2424](https://doi.org/10.1080/14786435.2018.1489152) ([arxiv.org/abs/1711.10555](https://arxiv.org/abs/1711.10555)).

## Author

Daniel N. Blaschke

## License
PyDislocDyn is distributed according to the [license.txt](license.txt) file. All contributions made by employees of Los Alamos National Laboratory are governed by that license.

C Number: C18073</br>
doi: [10.11578/dc.20180619.15](https://doi.org/10.11578/dc.20180619.15)</br>

Copyright (c) 2018, Triad National Security, LLC. All rights reserved.

The LANL development team asks that any forks or derivative works include appropriate attribution and citation of the LANL development team's original work.


## Requirements

* Python >=3.9,</br>
* [numpy](https://numpy.org/doc/stable/user/) >=1.19,</br>
* [scipy](https://docs.scipy.org/doc/scipy/reference/) >=1.9,</br>
* [sympy](https://www.sympy.org) >=1.6,</br>
* [matplotlib](https://matplotlib.org/) >=3.3</br>
* [pandas](https://pandas.pydata.org/) >=1.3 (and Jinja2)</br>

### Optional:

* [numba](https://numba.pydata.org/) >=0.53 (for speedup via just-in-time compilation of some subroutines),</br>
* [joblib](https://joblib.readthedocs.io) >=1.1 (for parallelization),</br>
* a Fortran 90 compiler
to employ the alternative faster Fortran implementations of some subroutines via [f2py](https://docs.scipy.org/doc/numpy/f2py/);
run 'python -m numpy.f2py -c subroutines.f90 -m subroutines' to use </br>
(or add appropriate options to build with OpenMP support, e.g. with gfortran and Python <=3.11: 'python -m numpy.f2py --f90flags=-fopenmp -lgomp -c subroutines.f90 -m subroutines'; </br>
with Python >=3.12: 'python -m numpy.f2py --dep=openmp -c subroutines.f90 -m subroutines').</br>
A helper function, pydislocdyn.utilities.compilefortranmodule(), has been included to automate compilation of the Fortran submodule and to ensure it is placed in the correct location.
* a recent version of LaTeX to build the manual (LA-UR-22-28074), pdf available at [doi:10.2172/1880452](https://doi.org/10.2172/1880452)

## Installation

* Download and unpack PyDislocDyn 1.3.0 or higher from https://github.com/dblaschke-LANL/PyDislocDyn/releases (or clone the latest master branch via:</br>
git clone https://github.com/dblaschke-LANL/PyDislocDyn.git ), then from within the PyDislocDyn folder (which contains the pyproject.toml file) simply run</br>
pip install . </br></br>
or (in order to install also the optional dependencies numba and joblib):</br>
pip install .\[fast\] </br></br>
Note: installation is optional, i.e. pydislocdyn can also be run locally (in which case it will temporarily add itself to sys.path upon importing or running a frontend script).

* then compile the Fortran submodule (optional) via</br>
python -c 'import pydislocdyn; pydislocdyn.utilities.compilefortranmodule()' </br></br>
Note: if you have chosen not to install PyDislocDyn, run the above command from within the PyDislocDyn folder, </br>
otherwise if PyDislocDyn is installed, make sure to change your current working directory so that it is <i>not</i> the one you ran 'pip install .' from.</br>
(Otherwise the local version will be imported instead of the installed one.)</br>
If compilation was successful, the following command will print 'True':</br>
python -c 'import pydislocdyn; print(pydislocdyn.usefortran)'

* to uninstall PyDislocDyn, run</br>
pip uninstall pydislocdyn</br></br>
Note: the compiled Fortran module has to be deleted manually upon uninstalling (and pip will let the user know its location)

## PyDislocDyn consists of:

### Python module

* *pydislocdyn*</br>
Defines functions to generate tensors of elastic constants and compliances,
functions and classes to compute dislocation displacement gradient fields, self energy, and line tension,
as well as functions to compute the drag coefficient from phonon wind.
One of its submodules (pydislocdyn.metaldata) contains
dictionaries storing input data taken from the references listed in that file, the most important ones being the ['CRC Handbook of Chemistry and Physics'](http://hbcp.chemnetbase.com); as well as ['Kaye and Laby Online'](https://web.archive.org/web/20190506031327/http://www.kayelaby.npl.co.uk/).
In particular, this submodule contains elastic constants, densities, lattice constants, and thermal expansion coefficients at room temperature for various metals necessary to run the programs listed above.

### Python programs / examples (using the module above)

* *linetension_calcs.py*</br>
Computes the line tension of a moving dislocation for various metals.
See [Phil. Mag. 98 (2018) 2397&ndash;2424](https://doi.org/10.1080/14786435.2018.1489152) ([arxiv.org/abs/1711.10555](https://arxiv.org/abs/1711.10555)) for details on the method.

* *polycrystal_averaging.py*</br>
Computes averages of elastic constants for polycrystals.
See [J. Appl. Phys. 122 (2017) 145110](https://doi.org/10.1063/1.4993443)  ([arxiv.org/abs/1706.07132](https://arxiv.org/abs/1706.07132)) for details on the method.

* *dragcoeff_iso.py*</br>
Computes the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal.
See LA-UR-16-24559 ([doi.org/10.2172/1434423](https://doi.org/10.2172/1434423)), [Phil. Mag. 100 (2020) 571&ndash;600](https://doi.org/10.1080/14786435.2019.1696484) ([arxiv.org/abs/1907.00101](https://arxiv.org/abs/1907.00101)), and [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) for details on the method.

* *dragcoeff_semi_iso.py*</br>
Computes the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation, where only the phonon spectrum is isotropic and everything else (i.e. the dislocation field and the elastic constants) respect the crystal symmetry. See [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) for details on the method.


