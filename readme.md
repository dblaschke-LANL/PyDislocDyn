# PyDislocDyn

PyDislocDyn is a suite of python programs designed to perform various calculations for basic research in dislocation dynamics in metals with simple crystal symmetries in the continuum limit. In particular, one of its main purposes is to calculate dislocation drag from phonon wind. Additional features include the averaging of elastic constants for polycrystals, the calculation of the dislocation field and the calculation of dislocation self-energy and line tension.
This code was first used for the computations leading to [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) and [arxiv.org/abs/1902.02451](https://arxiv.org/abs/1902.02451).
Additionally, it is able to reproduce the earlier results of LA-UR-16-24559 ([doi.org/10.2172/1434423](https://doi.org/10.2172/1434423)), [J. Appl. Phys. 122 (2017) 145110](https://doi.org/10.1063/1.4993443) ([arxiv.org/abs/1706.07132](https://arxiv.org/abs/1706.07132)), and [Phil. Mag. 98 (2018) 2397&ndash;2424](https://doi.org/10.1080/14786435.2018.1489152) ([arxiv.org/abs/1711.10555](https://arxiv.org/abs/1711.10555)).

## Author

Daniel N. Blaschke

## License
PyDislocDyn is distributed according to the [license.txt](license.txt) file. All contributions made by employees of Los Alamos National Laboratory are governed by that license.

C Number: C18073</br>

Copyright (c) 2018, Triad National Security, LLC. All rights reserved.

The LANL development team asks that any forks or derivative works include appropriate attribution and citation of the LANL development team's original work.


## Prerequisites

numpy,</br>
sympy,</br>
numba,</br>
joblib,</br>
matplotlib</br>

### Optional:

f2py + a Fortran compiler (to use the alternative faster Fortran implementations of some subroutines)

## PyDislocDyn consists of:

### Python programs / examples

* linetension_calcs.py</br>
Computes the line tension of a moving dislocation for various metals.
See [Phil. Mag. 98 (2018) 2397&ndash;2424](https://doi.org/10.1080/14786435.2018.1489152) ([arxiv.org/abs/1711.10555](https://arxiv.org/abs/1711.10555)) for details on the method.

* polycrystal_averaging.py</br>
Computes averages of elastic constants for polycrystals.
See [J. Appl. Phys. 122 (2017) 145110](https://doi.org/10.1063/1.4993443)  ([arxiv.org/abs/1706.07132](https://arxiv.org/abs/1706.07132)) for details on the method.

* dragcoeff_iso.py</br>
Computes the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal.
See LA-UR-16-24559 ([doi.org/10.2172/1434423](https://doi.org/10.2172/1434423)) and [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) for details on the method.

* dragcoeff_semi_iso.py</br>
Computes the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation, where only the phonon spectrum is isotropic and everything else (i.e. the dislocation field and the elastic constants) respect the crystal symmetry. See [J. Phys. Chem. Solids 124 (2019) 24&ndash;35](https://doi.org/10.1016/j.jpcs.2018.08.032) ([arxiv.org/abs/1804.01586](https://arxiv.org/abs/1804.01586)) for details on the method.

### Python modules (used by the programs listed above):

* elasticconstants.py</br>
Defines functions to generate tensors of elastic constants and compliances.

* dislocations.py</br>
Defines functions to compute dislocation displacement gradient fields, self energy and line tension.

* phononwind.py</br>
Defines functions to compute the drag coefficient from phonon wind.

* metal_data.py</br>
Defines dictionaries storing input data taken from the references listed in that file, the most important ones being the ['CRC Handbook of Chemistry and Physics'](http://hbcponline.com); as well as ['Kaye and Laby Online'](http://www.kayelaby.npl.co.uk/).
In particular, this module contains elastic constants, densities, lattice constants, and thermal expansion coefficients at room temperature for various metals necessary to run the programs listed above.



