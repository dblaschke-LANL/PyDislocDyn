! standalone test suite for subroutines.f90
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 25, 2026 - Apr. 8, 2026
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

module tests
  use parameters, only: sel, pi
  use checks
  implicit none
  contains
    subroutine test_disloc(count_pass,count_fail)
      use elastic_constants
      use dislocations
      integer, intent(inout) :: count_pass,count_fail
      type(disloc) :: Cu
      real(sel) :: C2(3,3,3,3), C3(3,3,3,3,3,3)
      real(sel), allocatable :: zeros(:), Etot(:), B(:,:)
      logical :: istrue
!~       integer :: i
      
      Cu = disloc(sym="cubic",metal="Cu",rho=8960.d0,b=3.6146d-10*(/0.5d0,0.5d0,0.d0/),n0=(/-1.d0,1.d0,-1.d0/))
      Cu%lat_a = (/3.6146d-10,0.d0,0.d0/)
      Cu%cij = (/168.3d9, 121.2d9, 75.7d9/)
      Cu%cijk = (/-1271.d9, -814.d9, -50.d9, -3.d9, -780.d9, -95.d9/)
      call Cu%init() ! attempts to inver Cu%burgers from Cu%b if not normalized
!~       do i=1,6
!~         print*,Cu%C2(i,:)/1.d9
!~         print*,Cu%C3(2,i,:)/1.d9
!~       end do
!~       do i=1,3
!~         print*,Cu%rot(i,:,1),Cu%rot(i,:,2)
!~       end do
      call testzero(Cu%rot(1,3,1)+Cu%rot(1,3,2)+0.81649658,"disloc_Cu_rot",1.d-6,count_pass,count_fail)
      call testzero(sum(Cu%C2)/1.d12-1.4592d0+sum(Cu%C3)/1.d12+33.402d0,"disloc_Cu_C2_C3",1.d-12,count_pass,count_fail)
      call unvoigt(Cu%C2,C2)
      call checkvoigt(C2,istrue)
      call testtrue(istrue,"disloc_Cu_checkvoigt_C2",count_pass,count_fail)
      call unvoigt(Cu%C3,C3)
      call checkvoigt(C3,istrue)
      call testtrue(istrue,"disloc_Cu_checkvoigt_C3",count_pass,count_fail)
      call testzero(Cu%Vc*1.d29-4.7225953240136,"disloc_Cu_Vc",1.d-6,count_pass,count_fail)
      call testzero(sum(Cu%theta)-pi/2.d0,"disloc_Cu_theta",1.d-6,count_pass,count_fail)
      call testzero(sum(Cu%lat_angles)-1.5d0*pi,"disloc_Cu_lat_angles",1.d-6,count_pass,count_fail)
      call testzero(dot_product(Cu%b,Cu%b)+dot_product(Cu%n0,Cu%n0)+dot_product(Cu%b,Cu%n0)+dot_product(Cu%t(:,1),Cu%b)-3.d0 &
             +dot_product(Cu%t(:,2),Cu%b)+1.d10*(Cu%burgers-3.6146d-10/sqrt(2.d0)),"disloc_Cu_b_n0_t",1.d-6,count_pass,count_fail)
      Cu%beta = 0.5d0
      call Cu%update_uij()
      allocate(zeros(Cu%nphi))
      zeros = 0.d0
      call testequalarray(Cu%uij(:,1,1,1)+Cu%uij(:,2,2,1)+Cu%uij(:,3,3,1),zeros,Cu%nphi,&
                          "disloc_Cu_uij-screw_zero-trace",1.d-12,count_pass,count_fail)
      allocate(Etot(Cu%ntheta))
      call computeEtot(Cu%uij,Cu%beta,Cu%C2norm,Cu%Cv,Cu%phi,Cu%ntheta,Cu%nphi,Etot)
      call testzero(sum(Etot)-0.22166413212d0,"disloc_Cu_Etot",1.d-9,count_pass,count_fail)
      
      call phonondrag(B,Cu,(/0.1d0,0.5d0/))
      call testzero(sum(B)-0.0886125,"disloc_Cu_drag",1.d-6,count_pass,count_fail)
    end subroutine test_disloc
end module tests

program runtests
  use parameters
  use phononwind_subroutines
  use elastic_constants
  use checks
  use tests
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
  
  call test_disloc(count_pass,count_fail)
  
  call cpu_time(finish_time)
  print*,"------------------------------------------------------------"
  print*,"SUMMARY:", count_pass," passed and ",count_fail," failed"
  !print*,"time: ",finish_time-start_time, "s"
  
end program runtests
