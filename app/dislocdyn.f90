! dislocdyn - a program to calculate the dislocation drag coefficient and other properties
! this Fortran implementation features only a subset of what the Python module can do
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - May 11, 2026
! NOTE: this program uses features of the fortran 2018 standard (such as assumed ranks of arrays); a recent compiler is required!
program dislocdyn
  use, intrinsic :: iso_fortran_env, only : error_unit, output_unit
  use parameters, only : sel, rzero, pi, prog_version=>version
  use utilities, only : ompinfo, linspace
  use elastic_constants, only : CheckReflectionSymmetry
  use dislocations
  use readinputfiles
  implicit none
  
  integer :: nthreads, i, j, k, p, num_args, un(3), start_time, finish_time, countrate
  character(256) :: materialfile, instructionfile, cmdlinearg, exe_name, proginfo, threadinfo, usageinfo
  character(256), dimension(:), allocatable :: args
  type(disloc), dimension(:), allocatable :: disl
  type(inputdeck) :: sim_plan
  real(sel), allocatable :: B(:,:), vlim(:,:)
  un = [output_unit, 123, error_unit]
  
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
  
  call system_clock(start_time,countrate)
  ! loop over material files, using the same instruction file for each one
  do i=1,num_args-1
    materialfile = trim(args(i))
    call read_materialfile(materialfile,disl(i))
    if (i==1) then
      call read_inputdeck(instructionfile,sim_plan,disl(i)%sym) ! need to know how many Miller indices to expect based on 'sym'
      open(unit=un(2), file=sim_plan%logfile, status='replace') ! open log file
      write(un(2),*) proginfo
      write(un(2),*) threadinfo
    end if
    do k=1,2
      write(un(k),*) new_line('a') // "--- Crystal / Dislocation properties ---"
      write(un(k),*) "name: ",disl(i)%metal
    end do
    if (all(abs(sim_plan%b)<rzero) .and. all(abs(sim_plan%n0)<rzero)) then
      if (trim(disl(i)%sym)=='fcc' .or. trim(disl(i)%sym)=='iso') then 
        sim_plan%b = [0.5d0,0.5d0,0.d0]
        sim_plan%n0=[-1.d0,1.d0,-1.d0]
        do k=1,2
          write(un(k),*) "WARNING: slip plane undefined; using default for fcc"
        end do
      else
        error stop "slip plane undefined"
      end if
    end if
    if (sim_plan%ntheta>2) then
      disl%ntheta = sim_plan%ntheta
    end if
    if (sim_plan%nphi/=500) then
      disl%nphi = sim_plan%nphi
    end if
    sim_plan%b = sim_plan%b/sim_plan%Millernorm
    sim_plan%n0 = sim_plan%n0/sim_plan%Millernorm
    call disl(i)%init(Millerb=sim_plan%b,Millern0=sim_plan%n0)
    do k = 1,2
      if (sim_plan%echoinput .or. k==2) then
        write(un(k),'(a, a, a10, f10.2, a10, f10.2, a)') "sym=", disl(i)%sym,", rho= ", disl(i)%rho,"kg/m^3, T= ", disl(i)%Temp," K"
        write(un(k),'(a, f10.6, f10.6, f10.6, a)') "lattice constants: ",disl(i)%lat_a*1.d10," Angstroem"
        write(un(k),'(a, f10.6, f10.6, f10.6)') "lattice angles [pi]: ",disl(i)%lat_angles/pi
        write(un(k),'(a, *(f10.2))') "cij [GPa]: ",disl(i)%cij/1.d9
        write(un(k),'(a, f10.2, f10.2, a)') "Lame constants: ", disl(i)%lam/1.d9, disl(i)%mu/1.d9, " GPa"
        write(un(k),'(a, *(f10.2))') "cijk [GPa]: ", disl(i)%cijk/1.d9
        write(un(k),'(a, f10.8, a)') "Vc=", disl(i)%Vc*1.d27, " nm^3"
        write(un(k),'(a, f10.6, a)') "burgers=", disl(i)%burgers*1.d10, " Angstroem"
        write(un(k),*) "slip plane: "//new_line('a')//"    b=", disl(i)%b, new_line('a'), "    n0=", disl(i)%n0
      end if
    end do
    
    do p=1,sim_plan%nsims
      if (sim_plan%sim_type(p)%str=='drag') then
        call phonondrag(B,disl(i),sim_plan%beta)
        do k=1,2
          write(un(k),*) new_line('a')//"--- Dislocation drag from phonon wind ---"
          write(un(k),*) "character angles in units of pi (theta=0 is pure screw, theta=pi/2 is pure edge):"
          write(un(k),'(*(f10.4))') disl(i)%theta/pi
          write(un(k),*) "beta // drag coefficient B(beta,theta) in units of mPas:"
          do j=1,size(sim_plan%beta)
            write(un(k),'(f10.4, *(f10.6))') sim_plan%beta(j), B(:,j)
          end do
        end do
      else if (sim_plan%sim_type(p)%str=='vlimit') then
        if (allocated(vlim)) deallocate(vlim)
        allocate(vlim(disl(i)%ntheta,3))
        call disl(i)%computevcrit(vlim)
        do k=1,2
          write(un(k),*) new_line('a')//"--- Limiting dislocation velocities ---"
          write(un(k),*) "theta [pi] / vlimit in m/s; lowest, 2nd, and highest"
          do j=1,disl(i)%ntheta
            write(un(k),'(f10.4,*(f10.2))') disl(i)%theta(j)/pi,vlim(j,:)
          end do
        end do
      else
        write(un(2),*) new_line('a') // "ERROR: sim_type="//sim_plan%sim_type(p)%str//" not implemented"
        error stop new_line('a') // "sim_type="//sim_plan%sim_type(p)%str//" not implemented"
      end if
    end do
  end do

  call system_clock(finish_time)
  print*,new_line('a') // "------------"
  print*,"time: ",real(finish_time-start_time)/real(countrate), "s" // new_line('a')
  close(unit=un(2)) ! close log file

end program dislocdyn
