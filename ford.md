
`dislocdynlib`, distributed as part of PyDislocDyn, is a Fortran library and program designed to perform various calculations for basic research in dislocation dynamics in metals with various crystal symmetries in the continuum limit.
In particular, one of its main purposes is to calculate dislocation drag from phonon wind as well as limiting gliding velocities of dislocations.
Additional features include the averaging of elastic constants for polycrystals and the calculation of the dislocation field.
As such, the Fortran library implements a subset of features included in the `dydislocdyn` Python module.
Note that the Fortran version prioritizes computation speed over accuracy for dislocation drag.

## Author

Daniel N. Blaschke

## License
This software is open source software available under the BSD-3-Clause license (see LICENSE.md file). All contributions made by employees of Los Alamos National Laboratory are governed by that license.

C Number: C18073</br>

Copyright (c) 2018, Triad National Security, LLC. All rights reserved.

## Requirements

* a Fortran 2018 capable compiler (such as gfortran 10 or higher)
* [Fortran package manager (fpm)](https://fpm.fortran-lang.org/) >=0.13 or GNU Make
* [Ford](https://forddocs.readthedocs.io/en/stable/) >=7 (or Doxygen) to build this documentation

## Compiling
* either run</br>
`make shared`</br>
(Linux and MacOS only),
* or use the Fortran package manager:</br>
`fpm build --profile release`

* to use the standalone Fortran library `dislocdynlib` within your fpm project, add the following to your `fpm.toml` file:</br>
```toml
[dependencies]
dislocdynlib.git = "https://github.com/dblaschke-lanl/pydislocdyn"
```

