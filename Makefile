## required compiler versions: gfortran>=15, flang>=22, lfortran>=0.68, ifx>=2025, aof>=6.0, crayftn>=17
## (do concurrent with locality and reduce statement implemented in gfortran 15)
PPFLAGS = 
ifeq ($(FC),lfortran)
  FC = lfortran
  PPFLAGS = --cpp
  FFLAGS = --fast --std=f23 --realloc-lhs-arrays#--openmp
  FFLAGS_dbg = -v --realloc-lhs-arrays --detect-leaks -g --fpe-trap invalid, zero, overflow, underflow, inexact, denormal
  LDFLAGS = 
  LD_SH = --shared $(LDFLAGS)
else ifeq ($(FC),flang)
  FC = flang# -v -flto -ffree-line-length-225
  FFLAGS = -O3 -pedantic -fopenmp -march=native# -mmlir -fdynamic-heap-array -ffast-math
  FFLAGS_dbg = -v -O0 -pedantic -g -fcheck=all
  LDFLAGS = -lomp
  LD_SH = -shared $(LDFLAGS)
else ifeq ($(FC),aof)
  FC = aof
  FFLAGS = -O3 -pedantic -fopenmp -march=native
  FFLAGS_dbg = -v -O0 -pedantic -g -fcheck=all
  LDFLAGS = -Wl,-z,execstack -lomp
  LD_SH = -shared $(LDFLAGS)
else ifeq ($(FC),crayftn)
  FC = crayftn
  FFLAGS = -O3 -fopenmp
  FFLAGS_dbg = -O0
  LDFLAGS = -fopenmp
  LD_SH = -shared $(LDFLAGS)
else ifeq ($(FC),ifx)
  FC = ifx
  FFLAGS = -O3 -stand f23 -qopenmp -heap-arrays 1000# -qno-openmp-simd -fp-model=precise -fno-fast-math
  FFLAGS_dbg = -g -O0 -traceback -check all -warn all -fpe0 -init=snan,arrays -fp-model=strict
  LDFLAGS = -qopenmp 
  LD_SH = -shared $(LDFLAGS)
else # always fall back to gfortran
  FC = gfortran -fimplicit-none# -flto=auto -ffree-line-length-225 -fstack-usage
  FFLAGS = -O3 -Wall -pedantic -Wextra -std=f2023 -march=native -fopenmp# -Warray-temporaries -funroll-loops -finit-real=snan
  FFLAGS_dbg = -O0 -fcheck=all -Wall -pedantic -Wextra -std=f2023 -finit-real=nan -g -fbacktrace -ffpe-trap=invalid,zero,overflow
  LDFLAGS = -lgomp
  LD_SH = -shared $(LDFLAGS)
endif

## name of executable
EXEC = dislocdyn
EXEC_tests = runtests
SHARED = dislocdynlib

ifeq ($(FMIN_PATH),)
	FMIN_PATH = ../fmin/src
endif

## define colors for output via echo below
COL_YELLOW = \033[33m
COL_PURPLE = \033[35m
COL_RESET = \033[0m

all:  build runtests

help:
	@echo "\n$(COL_YELLOW)TARGETS:$(COL_RESET)"
	@echo "make all             build $(EXEC_tests).x and $(EXEC).x"
	@echo "make runtests        only build the testsuite $(EXEC_tests).x"
	@echo "make build           only build $(EXEC).x"
	@echo "make shared          build the shared library lib$(SHARED).so and $(EXEC)_sh.x linked against it"
	@echo "make debug           build $(EXEC_tests).x and $(EXEC).x with debug flags and without openmp support"
	@echo "make clean           delete all object and mod files"
	@echo "make cleanall        delete all object files, mod files, the library, and executables\n"
	@echo "default build is with gfortran, set environment variable FC=lfortran, flang, aof, crayftn, or ifx"
	@echo "to use another compiler (or edit this Makefile as needed)\n"

_buildtest: _builddisl testing/runtests.f90
	$(FC) -c $(FFLAGS) testing/runtests.f90

runtests: _builddisl _buildtest
	$(FC) -o $(EXEC_tests).x subroutines.o elasticconstants.o fmin_module.o optimize.o phononwind.o crystals.o dislocations.o \
  readinputfiles.o runtests.o $(LDFLAGS)

_buildsubrtn: pydislocdyn/subroutines.f90
	$(FC) -c $(FFLAGS) pydislocdyn/subroutines.f90

_buildelastic: _buildsubrtn pydislocdyn/elasticconstants.f90
	$(FC) -c $(FFLAGS) pydislocdyn/elasticconstants.f90

_buildopt: _buildsubrtn $(FMIN_PATH)/fmin_module.F90 pydislocdyn/optimize.f90
	$(FC) -c $(PPFLAGS) $(FFLAGS) $(FMIN_PATH)/fmin_module.F90
	$(FC) -c $(FFLAGS) pydislocdyn/optimize.f90

_buildphonon: _buildsubrtn pydislocdyn/phononwind.f90
	$(FC) -c $(FFLAGS) pydislocdyn/phononwind.f90

_buildcryst: _buildsubrtn _buildelastic pydislocdyn/crystals.f90
	$(FC) -c $(FFLAGS) pydislocdyn/crystals.f90

_builddisl: _buildcryst _buildopt _buildphonon pydislocdyn/dislocations.f90 pydislocdyn/readinputfiles.f90
	$(FC) -c $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c $(FFLAGS) pydislocdyn/readinputfiles.f90

_buildapp: _builddisl app/dislocdyn.f90
	$(FC) -c $(FFLAGS) app/dislocdyn.f90

build: _builddisl _buildapp
	$(FC) -o $(EXEC).x subroutines.o elasticconstants.o fmin_module.o optimize.o phononwind.o crystals.o dislocations.o \
	readinputfiles.o dislocdyn.o $(LDFLAGS)

debug: FFLAGS = $(FFLAGS_dbg)
debug: build runtests

shared: FFLAGS += -fPIC
shared: _buildapp _buildtest
	$(FC) -o lib$(SHARED).so subroutines.o elasticconstants.o fmin_module.o optimize.o phononwind.o crystals.o dislocations.o \
	readinputfiles.o $(LD_SH)

	$(FC) -o $(EXEC_tests)_sh.x runtests.o $(LDFLAGS) -l$(SHARED) -L.
	@echo "\n$(COL_PURPLE)-> On linux run with: LD_LIBRARY_PATH="." ./runtests_sh.x $(COL_RESET)\n"

	$(FC) -o $(EXEC)_sh.x dislocdyn.o $(LDFLAGS) -l$(SHARED) -L.
	@echo "\n$(COL_PURPLE)-> On linux run with: LD_LIBRARY_PATH="." ./dislocdyn_sh.x $(COL_RESET)\n"

static: build
	ar rcs lib$(SHARED).a subroutines.o elasticconstants.o fmin_module.o optimize.o phononwind.o crystals.o dislocations.o \
	readinputfiles.o

clean: 
	rm -f subroutines.o elasticconstants.o fmin_module.o optimize.o phononwind.o crystals.o dislocations.o readinputfiles.o \
	runtests.o dislocdyn.o dislocdyn_*.mod fmin_module.mod

cleanall: clean
	rm -f $(EXEC_tests).x $(EXEC_tests)_sh.x $(EXEC).x $(EXEC)_sh.x lib$(SHARED).so lib$(SHARED).a
