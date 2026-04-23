! dislocdyn - a program to calculate the dislocation drag coefficient and other properties
! this Fortran implementation features only a subset of what the Python module can do
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - Apr. 22, 2026
! NOTE: this program uses features of the fortran 2018 standard (such as assumed ranks of arrays); a recent compiler is required!
program dislocdyn
  use parameters, only : sel, prog_version=>version
  use utilities, only : ompinfo, linspace
  use dislocations
  use readinputfiles
  implicit none
  
  integer :: nthreads, i
  character(256) :: materialfile, instructionfile, cmdlinearg, exe_name, proginfo, threadinfo, usageinfo
  type(disloc) :: disl
  real(sel), allocatable :: B(:,:), beta(:)
  
  write(proginfo, '(I0)') prog_version
  proginfo = "DislocDyn version " // trim(proginfo)
  call get_command_argument(1, cmdlinearg)
  call get_command_argument(0, exe_name)
  usageinfo = "USAGE: " // trim(exe_name) // " [--version] [--help] <inputfilename>"

  call ompinfo(nthreads)
  if (nthreads>0) then
    write(threadinfo, '(I0)') nthreads
    threadinfo = trim(exe_name) // " compiled with openmp support, using " // trim(threadinfo) // " threads"
  else
    threadinfo = trim(exe_name) // " compiled without openmp support"
  end if
  
  if (len_trim(cmdlinearg) > 0) then
    if ((cmdlinearg == '--version') .or. (cmdlinearg == '-v')) then
      print*,prog_version
      stop
    else
      print*,proginfo
      print*,threadinfo
      if ((cmdlinearg == '--help') .or. (cmdlinearg == '-h')) then
        print*,usageinfo
        stop
      end if
    end if
  else
    stop "Missing input; " // usageinfo
  end if
  
  call read_materialfile(trim(cmdlinearg),disl)
  print*,"name: ",disl%metal,", sym=", disl%sym," rho= ", disl%rho,"kg/m^3 T= ", disl%Temp,"K"
  print*,"lattice constants: ",disl%lat_a*1.d10," Angstroem"
  print*,"cij: ",disl%cij/1.d9,"GPa"
  print*,"Lame constants: ", disl%lam/1.d9, disl%mu/1.d9, " GPa"
  print*,"cijk: ", disl%cijk/1.d9,"GPa"
  
  if (trim(disl%sym)=='fcc') then 
    disl%b = disl%lat_a(1)*[0.5d0,0.5d0,0.d0]
    disl%n0=[-1.d0,1.d0,-1.d0]
  end if
  call disl%init()
  print*,"Vc=", disl%Vc*1.d27, "nm^3"
  
  allocate(beta(6))
  call linspace(0.1d0,0.6d0,size(beta),beta)
  call phonondrag(B,disl,beta)
  
  print*,"beta // drag coefficient for screw , edge:"
  do i=1,size(beta)
    print*,beta(i), B(:,i)
  end do

end program dislocdyn
