! dislocdyn - a program to calculate the dislocation drag coefficient and other properties
! this Fortran implementation features only a subset of what the Python module can do
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - May 5, 2026
! NOTE: this program uses features of the fortran 2018 standard (such as assumed ranks of arrays); a recent compiler is required!
program dislocdyn
  use parameters, only : sel, pi, prog_version=>version
  use utilities, only : ompinfo, linspace
  use dislocations
  use readinputfiles
  implicit none
  
  integer :: nthreads, i, j, num_args
  character(256) :: materialfile, instructionfile, cmdlinearg, exe_name, proginfo, threadinfo, usageinfo
  character(256), dimension(:), allocatable :: args
  type(disloc), dimension(:), allocatable :: disl
  type(inputdeck) :: sim_plan
  real(sel), allocatable :: B(:,:), theta(:)
  real(kind=sel) :: start_time, finish_time
  
  write(proginfo, '(I0)') prog_version
  proginfo = "DislocDyn version " // trim(proginfo)
  call get_command_argument(1, cmdlinearg)
  call get_command_argument(0, exe_name)
  usageinfo = "USAGE: " // trim(exe_name) // " [--version] [--help] <1 or more materialfiles> <inputdeck>"

  call ompinfo(nthreads)
  if (nthreads>0) then
    write(threadinfo, '(I0)') nthreads
    threadinfo = trim(exe_name) // " compiled with openmp support, using " // trim(threadinfo) // " threads"
  else
    threadinfo = trim(exe_name) // " compiled without openmp support"
  end if
  
  num_args = command_argument_count()
  allocate(args(num_args))
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

  if (num_args>=2) then
    do i=1,num_args
      call get_command_argument(i,args(i))
    end do
  else
    stop "Missing input; " // usageinfo
  end if
  
  instructionfile = trim(args(num_args))
  allocate(disl(num_args-1))
  call read_inputdeck(instructionfile,sim_plan)
  
  open(unit=123, file=sim_plan%logfile, status='replace') ! open log file
  write(123,*) proginfo
  write(123,*) threadinfo
  
  call cpu_time(start_time)
  ! loop over material files, using the same instruction file for each one
  do i=1,num_args-1
    materialfile = trim(args(i))
    call read_materialfile(materialfile,disl(i))
    print*,new_line('a') // "name: ",disl(i)%metal
    write(123,*) new_line('a') // "name: ",disl(i)%metal
    disl(i)%b = sim_plan%b*disl(i)%lat_a(1) ! assume input is Cartesian in units of lattice constant a
    disl(i)%n0=sim_plan%n0
    if (all(disl(i)%b==0) .and. all(disl(i)%n0==0)) then
      if (trim(disl(i)%sym)=='fcc' .or. trim(disl(i)%sym)=='iso') then 
        disl(i)%b = disl(i)%lat_a(1)*[0.5d0,0.5d0,0.d0]
        disl(i)%n0=[-1.d0,1.d0,-1.d0]
        print*,"WARNING: slip plane undefined; using default for fcc"
        write(123,*) "WARNING: slip plane undefined; using default for fcc"
      else
        error stop "slip plane undefined"
      end if
    end if
    if (sim_plan%ntheta>2) then
      disl%ntheta = sim_plan%ntheta
    end if
    call disl(i)%init()
    if (sim_plan%echoinput) then
      print '(a, a, a10, f10.2, a10, f10.2, a)',"sym=", disl(i)%sym,", rho= ", disl(i)%rho,"kg/m^3, T= ", disl(i)%Temp," K"
      print '(a, f10.6, f10.6, f10.6, a)',"lattice constants: ",disl(i)%lat_a*1.d10," Angstroem"
      print '(a, *(f10.2))',"cij [GPa]: ",disl(i)%cij/1.d9
      print '(a, f10.2, f10.2, a)',"Lame constants: ", disl(i)%lam/1.d9, disl(i)%mu/1.d9, " GPa"
      print '(a, *(f10.2))',"cijk [GPa]: ", disl(i)%cijk/1.d9
      print '(a, f10.8, a)',"Vc=", disl(i)%Vc*1.d27, " nm^3"
      print '(a, f10.6, a)',"burgers=", disl(i)%burgers*1.d10, " Angstroem"
      print*,"slip plane: "//new_line('a')//"    b=", disl(i)%b, new_line('a'), "    n0=", disl(i)%n0, new_line('a')
    end if
    write(123,'(a, a, a10, f10.2, a10, f10.2, a)') "sym=", disl(i)%sym,", rho= ", disl(i)%rho,"kg/m^3, T= ", disl(i)%Temp," K"
    write(123,'(a, f10.6, f10.6, f10.6, a)') "lattice constants: ",disl(i)%lat_a*1.d10," Angstroem"
    write(123,'(a, *(f10.2))') "cij [GPa]: ",disl(i)%cij/1.d9
    write(123,'(a, f10.2, f10.2, a)') "Lame constants: ", disl(i)%lam/1.d9, disl(i)%mu/1.d9, " GPa"
    write(123,'(a, *(f10.2))') "cijk [GPa]: ", disl(i)%cijk/1.d9
    write(123,'(a, f10.8, a)') "Vc=", disl(i)%Vc*1.d27, " nm^3"
    write(123,'(a, f10.6, a)') "burgers=", disl(i)%burgers*1.d10, " Angstroem"
    write(123,*) "slip plane: "//new_line('a')//"    b=", disl(i)%b, new_line('a'), "    n0=", disl(i)%n0, new_line('a')
    
    if (sim_plan%sim_type=='drag') then
      call phonondrag(B,disl(i),sim_plan%beta)
      print*,"character angles in units of pi (theta=0 is pure screw, theta=pi/2 is pure edge):"
      write(123,*) "character angles in units of pi (theta=0 is pure screw, theta=pi/2 is pure edge):"
      print '(*(f10.4))',disl(i)%theta/pi
      write(123,'(*(f10.4))') disl(i)%theta/pi
      print*,"beta // drag coefficient B(beta,theta) in units of mPas:"
      write(123,*) "beta // drag coefficient B(beta,theta) in units of mPas:"
      do j=1,size(sim_plan%beta)
        print '(f10.4, *(f10.6))',sim_plan%beta(j), B(:,j)
        write(123,'(f10.4, *(f10.6))') sim_plan%beta(j), B(:,j)
      end do
    else
      write(123,*) new_line('a') // "ERROR: sim_type="//sim_plan%sim_type//" not implemented"
      error stop new_line('a') // "sim_type="//sim_plan%sim_type//" not implemented"
    end if
  end do

  call cpu_time(finish_time)
  print*,new_line('a') // "time: ",(finish_time-start_time)/real(nthreads, kind=sel), "s" // new_line('a')
  close(unit=123) ! close log file

end program dislocdyn
