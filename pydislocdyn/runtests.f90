! standalone test suite for subroutines.f90
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 25, 2026 - Mar. 30, 2026
! compile with:
! gfortran pydislocdyn/subroutines.f90 pydislocdyn/runtests.f90 -o runtests.x -Wall -pedantic -Wextra -fopenmp -lgomp
! or similar
! NOTE: this file uses features of the fortran 2018 standard (such as assumed ranks of arrays); a recent compiler is required!
module checks
  use parameters, only: sel
  implicit none
  contains
  subroutine testtrue(equal,string,count_pass,count_fail)
    logical, intent(in) :: equal
    integer :: count_pass,count_fail
    character(*), intent(in) :: string
    if (equal) then
      count_pass = count_pass+1
      print*,"test " // string//": "//char(9)//"PASSED"
    else
      count_fail = count_fail+1
      print*,"test " // string//": "//char(9)//"FAILED"
    end if
  end subroutine testtrue
  subroutine testequal(A,B,n,m,string,tolerance,count_pass,count_fail)
  !  Check if two NxM matrices are equal (with some tolerance), and print 'test <string> PASSED/FAILED'
    implicit none
    integer, intent(in)  :: n, m
    real(kind=sel), intent(in)  :: tolerance
    real(kind=sel), dimension(n,m), intent(in)  :: A, B
    character(*), intent(in) :: string
    integer :: count_pass,count_fail
    logical :: equal
    
    equal = all(abs(A-B) <= tolerance)
    if (equal) then
      count_pass = count_pass+1
      print*,"test " // string//": "//char(9)//"PASSED"
    else
      count_fail = count_fail+1
      print*,"test " // string//": "//char(9)//"FAILED"
    end if
    
    return
  end subroutine testequal

  SUBROUTINE testequalarray(A,B,n,string,tolerance,count_pass,count_fail)
  !  Check if two arrays are equal (with some tolerance), and print 'test <string> PASSED/FAILED'
    implicit none
    integer, intent(in)  :: n
    real(kind=sel), intent(in)  :: tolerance
    real(kind=sel), dimension(n), intent(in)  :: A, B
    character(*), intent(in) :: string
    integer :: count_pass,count_fail
    logical :: equal
    
    equal = all(abs(A-B) <= tolerance)
    if (equal) then
      count_pass = count_pass+1
      print*,"test " // string//": "//char(9)//"PASSED"
    else
      count_fail = count_fail+1
      print*,"test " // string//": "//char(9)//"FAILED"
    end if
    
    return
  end subroutine testequalarray

  subroutine testzero(A,string,tolerance,count_pass,count_fail)
  ! check if A=0 (with some tolerance), and print 'test <string> PASSED/FAILED'
    implicit none
    real(kind=sel), intent(in)  :: tolerance, A
    character(*), intent(in) :: string
    integer :: count_pass,count_fail
    call testequalarray((/A/),(/0.d0/),1,string,tolerance,count_pass,count_fail)
    return
  end subroutine testzero
  !-------------------------------------
  subroutine checkvoigt(x,b)
    use elastic_constants
    real(kind=sel), intent(in) :: x(..)
    logical, intent(out) :: b
    real(kind=sel) :: y2(3,3), y4(3,3,3,3), y6(3,3,3,3,3,3)
    real(kind=sel) :: z1(6), z2(6,6), z3(6,6,6)
    select rank(x)
      rank(2)
        call voigt(x,z1)
        call unvoigt(z1,y2)
        b = all(x==y2)
      rank(4)
        call voigt(x,z2)
        call unvoigt(z2,y4)
        b = all(x==y4)
      rank(6)
        call voigt(x,z3)
        call unvoigt(z3,y6)
        b = all(x==y6)
      rank default
        print*,"ERROR: rank must be 2,4, or 6"
        b = .false.
      end select
  end subroutine checkvoigt
end module checks

program runtests
  use parameters
  use phononwind_subroutines
  use elastic_constants
  use checks
  implicit none
  
  real(kind=sel) :: tmpintegral, array1(5),array2(5), start_time, finish_time
  real(kind=sel), dimension(3,3) :: A, B, one=reshape((/1.d0,0.d0,0.d0,0.d0,1.d0,0.d0,0.d0,0.d0,1.d0/),(/3,3/))
  real(sel) :: a4(3,3,3,3), a6(3,3,3,3,3,3), b1(6), b2(6,6), b3(6,6,6), xtric(21)
  real(kind=sel), allocatable, dimension(:) :: x, func, integral
  logical :: istrue
  integer :: resol, nthreads, count_fail=0, count_pass=0
  character(32) :: exe_name
  resol = 10000
  allocate(x(resol), func(resol), integral(resol))
  call cpu_time(start_time)
  
  ! check for openmp
  call get_command_argument(0, exe_name)
  call ompinfo(nthreads)
  if (nthreads>0) then
    print*,exe_name, " compiled with openmp support, using ",nthreads," threads"
  else
    print*,exe_name, " compiled without openmp support"
  end if
  
  ! test linspace
  call linspace(0.2d0,1.d0,5,array1)
  array2 = (/0.2d0,0.4d0,0.6d0,0.8d0,1.d0/)
  call testequalarray(array1,array2,5,"linspace",1.d-15,count_pass,count_fail)
  
  ! test cumtrapz / trapz
  call linspace(1.d-15, 100.d0, resol, x)
  func(:) = sin(pi*x(:)/25.d0) + x(:)**3 / (exp(x(:))-1.d0)
  call cumtrapz(func,x,resol,integral)
  call testzero(pi**4/15.d0 - integral(resol),"cumtrapz",1.d-6,count_pass,count_fail)
  call trapz(func,x,resol,tmpintegral)
  call testzero(tmpintegral - integral(resol),"trapz",1.d-13,count_pass,count_fail)
  
  ! test inv
  CALL RANDOM_NUMBER(A)
  call inv(A,B)
  call testequal(one,matmul(A,B),3,3,"inv",1.d-12,count_pass,count_fail)
  
  ! test voigt
  call random_number(A)
  A = A+transpose(A)
  call checkvoigt(A,istrue)
  call testtrue(istrue,"checkvoigt_2D",count_pass,count_fail)
  call random_number(b2)
  call unvoigt(b2,a4)
  call checkvoigt(a4,istrue)
  call testtrue(istrue,"checkvoigt_4D",count_pass,count_fail)
  call random_number(b3)
  call unvoigt(b3,a6)
  call checkvoigt(a6,istrue)
  call testtrue(istrue,"checkvoigt_6D",count_pass,count_fail)
  call random_number(b1)
  call checkvoigt(b1,istrue)
  call testtrue(.not. istrue,"checkvoigt_asym (intended error printing in above line)",count_pass,count_fail)
  
  ! test C2
!~   call elasticC2((/1.d0,2.d0,3.d0/),'cubic',b2)
!~   do i=1,6
!~     print*,b2(i,:)
!~   end do
!~   call elasticC2((/1.d0,2.d0/),'iso',b2)
!~   do i=1,6
!~     print*,b2(i,:)
!~   end do
  call linspace(1.d0,21.d0,21,xtric)
  call elasticC2(xtric,'tric',b2)
!~   do i=1,6
!~     print*,b2(i,:)
!~   end do
  call testtrue(abs(sum(xtric)*2.d0-76.d0-sum(b2))<1.d-18,"elasticC2_tric",count_pass,count_fail)
  
  call cpu_time(finish_time)
  print*,"------------------------------------------------------------"
  print*,"SUMMARY:", count_pass," passed and ",count_fail," failed"
  !print*,"time: ",finish_time-start_time, "s"
  
end program runtests
