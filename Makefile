ifeq ($(FC),lfortran)
  FC = lfortran -v
  FFLAGS = -O3 --std=f23 --separate-compilation --implicit-interface #--openmp
  LDFLAGS = 
else ifeq ($(FC),flang)
  FC = flang
  FFLAGS = -O3 -std=f2018 -fopenmp
  LDFLAGS = -lomp
else # always fall back to gfortran
  FC = gfortran -fimplicit-none
  FFLAGS = -O3 -fimplicit-none -Wall -pedantic -Wextra -std=f2018 -fopenmp
  LDFLAGS = -lgomp
endif

## name of executable
EXEC_tests = runtests.x

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
	$(FC) -o runtests.x subroutines.o elasticconstants.o dislocations.o runtests.o $(LDFLAGS)

clean: 
	rm -f subroutines.o elasticconstants.o dislocations.o runtests.o \
	parameters.mod  elastic_constants.mod  phononwind_subroutines.mod dislocations.mod checks.mod tests.mod

cleanall: clean
	rm -f $(EXEC_tests)
