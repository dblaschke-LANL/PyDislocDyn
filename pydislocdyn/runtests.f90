! compile with:
! gfortran pydislocdyn/subroutines.f90 pydislocdyn/runtests.f90 -o runtests.x -Wall -pedantic -Wextra -fopenmp -lgomp
! or similar
module checks
  use parameters, only: sel
  implicit none
  contains
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
end module checks

program runtests
  use parameters
  use phononwind_subroutines
  use checks
  implicit none
  
  real(kind=sel) :: tmpintegral, array1(5),array2(5), start_time, finish_time
  real(kind=sel), dimension(3,3) :: A, B, one=reshape((/1.d0,0.d0,0.d0,0.d0,1.d0,0.d0,0.d0,0.d0,1.d0/),(/3,3/))
  real(kind=sel), allocatable, dimension(:) :: x, func, integral
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
  
  call cpu_time(finish_time)
  print*,"------------------------------------------------------------"
  print*,"SUMMARY:", count_pass," passed and ",count_fail," failed"
  !print*,"time: ",finish_time-start_time, "s"
  
end program runtests
