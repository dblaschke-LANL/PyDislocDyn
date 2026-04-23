## required compiler versions: gfortran>=10, flang>=20
ifeq ($(FC),lfortran)
  FC = lfortran -v
  FFLAGS = --fast --std=f23 --realloc-lhs-arrays#--openmp
#~   FFLAGS = --realloc-lhs-arrays --detect-leaks -g --fpe-trap invalid, zero, overflow, underflow, inexact, denormal
  LDFLAGS = 
  LD_SH = --shared $(LDFLAGS)
else ifeq ($(FC),flang)
  FC = flang -v# -flto
  FFLAGS = -O3 -pedantic -std=f2018 -fopenmp -march=native# -ffast-math
  LDFLAGS = -lomp
  LD_SH = -shared $(LDFLAGS)
else # always fall back to gfortran
  FC = gfortran -fimplicit-none# -flto -ffree-line-length-225
  FFLAGS = -O3  -Wall -pedantic -Wextra -std=f2018 -fopenmp -march=native
  LDFLAGS = -lgomp
  LD_SH = -shared $(LDFLAGS)
endif

## name of executable
EXEC = dislocdyn
EXEC_tests = runtests
SHARED = dislocdyn

all:  runtests build clean

help:
	@echo 'targets:'
	@echo 'make all             build $(EXEC_tests), then delete all object files'
	@echo 'make runtests        only build the testsuite $(EXEC_tests)'
	@echo 'make clean           delete all object files'
	@echo 'make cleanall        delete all object files and executables'
	@echo ''

runtests: pydislocdyn/subroutines.f90 pydislocdyn/elasticconstants.f90 \
        pydislocdyn/dislocations.f90 testing/runtests.f90
	$(FC) -c $(FFLAGS) pydislocdyn/subroutines.f90
	$(FC) -c $(FFLAGS) pydislocdyn/elasticconstants.f90
	$(FC) -c $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c $(FFLAGS) testing/runtests.f90
	# Link
	$(FC) -o $(EXEC_tests).x subroutines.o elasticconstants.o dislocations.o runtests.o $(LDFLAGS)

build: pydislocdyn/subroutines.f90 pydislocdyn/elasticconstants.f90 \
       pydislocdyn/dislocations.f90 pydislocdyn/readinputfiles.f90 app/dislocdyn.f90
	$(FC) -c $(FFLAGS) pydislocdyn/subroutines.f90
	$(FC) -c $(FFLAGS) pydislocdyn/elasticconstants.f90
	$(FC) -c $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c $(FFLAGS) pydislocdyn/readinputfiles.f90
	$(FC) -c $(FFLAGS) app/dislocdyn.f90
	# Link
	$(FC) -o $(EXEC).x subroutines.o elasticconstants.o dislocations.o readinputfiles.o dislocdyn.o $(LDFLAGS)
	
shared: pydislocdyn/subroutines.f90 pydislocdyn/elasticconstants.f90 \
        pydislocdyn/dislocations.f90 pydislocdyn/readinputfiles.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/subroutines.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/elasticconstants.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/readinputfiles.f90
	$(FC) -c $(FFLAGS) testing/runtests.f90
	$(FC) -c $(FFLAGS) app/dislocdyn.f90
	# Link for linux
	$(FC) -o lib$(SHARED).so subroutines.o elasticconstants.o dislocations.o readinputfiles.o $(LD_SH)
	$(FC) -o $(EXEC_tests)_sh.x runtests.o $(LDFLAGS) -l$(SHARED) -L.
	## on linux run with: LD_LIBRARY_PATH="." ./runtests_sh.x)
	$(FC) -o $(EXEC)_sh.x dislocdyn.o $(LDFLAGS) -l$(SHARED) -L.
	## on linux run with: LD_LIBRARY_PATH="." ./dislocdyn_sh.x)

clean: 
	rm -f subroutines.o elasticconstants.o dislocations.o readinputfiles.o runtests.o dislocdyn.o \
	parameters.mod utilities.mod various_subroutines.mod phononwind.mod phononwind_subroutines.mod \
	elastic_constants.mod dislocations.mod readinputfiles.mod checks.mod tests.mod

cleanall: clean
	rm -f $(EXEC_tests).x $(EXEC_tests)_sh.x $(EXEC).x $(EXEC)_sh.x lib$(SHARED).so
