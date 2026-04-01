ifeq ($(FC),lfortran)
  FC = lfortran -v
  FFLAGS = -O3 --std=f23 --separate-compilation --implicit-interface #--openmp
  LDFLAGS = 
  LD_SH = --shared $(LDFLAGS)
else ifeq ($(FC),flang)
  FC = flang
  FFLAGS = -O3 -std=f2018 -fopenmp
  LDFLAGS = -lomp
  LD_SH = -shared $(LDFLAGS)
else # always fall back to gfortran
  FC = gfortran -fimplicit-none
  FFLAGS = -O3 -fimplicit-none -Wall -pedantic -Wextra -std=f2018 -fopenmp
  LDFLAGS = -lgomp
  LD_SH = -shared $(LDFLAGS)
endif

## name of executable
EXEC_tests = runtests
SHARED = dislocdyn

all:  runtests clean

help:
	@echo 'targets:'
	@echo 'make all             build $(EXEC_tests), then delete all object files'
	@echo 'make runtests        only build the testsuite $(EXEC_tests)'
	@echo 'make clean           delete all object files'
	@echo 'make cleanall        delete all object files and executables'
	@echo ''

runtests:  $(EXEC_tests)
$(EXEC_tests): pydislocdyn/subroutines.f90 pydislocdyn/elasticconstants.f90 \
        pydislocdyn/dislocations.f90 pydislocdyn/runtests.f90
	$(FC) -c $(FFLAGS) pydislocdyn/subroutines.f90
	$(FC) -c $(FFLAGS) pydislocdyn/elasticconstants.f90
	$(FC) -c $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c $(FFLAGS) pydislocdyn/runtests.f90
	# Link
	$(FC) -o $(EXEC_tests).x subroutines.o elasticconstants.o dislocations.o runtests.o $(LDFLAGS)
	
shared:  $(SHARED)
$(SHARED): pydislocdyn/subroutines.f90 pydislocdyn/elasticconstants.f90 \
        pydislocdyn/dislocations.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/subroutines.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/elasticconstants.f90
	$(FC) -c -fPIC $(FFLAGS) pydislocdyn/dislocations.f90
	$(FC) -c $(FFLAGS) pydislocdyn/runtests.f90
	# Link for linux
	$(FC) -o lib$(SHARED).so subroutines.o elasticconstants.o dislocations.o $(LD_SH)
	$(FC) -o $(EXEC_tests)_sh.x runtests.o $(LDFLAGS) -l$(SHARED) -L.
	## on linux run with: LD_LIBRARY_PATH="." ./runtests_sh.x

clean: 
	rm -f subroutines.o elasticconstants.o dislocations.o runtests.o \
	parameters.mod  elastic_constants.mod  phononwind_subroutines.mod dislocations.mod checks.mod tests.mod

cleanall: clean
	rm -f $(EXEC_tests).x $(EXEC_tests)_sh.x lib$(SHARED).so
