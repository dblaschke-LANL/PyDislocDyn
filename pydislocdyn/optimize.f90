! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 28, 2026

!> contains optimiziation algorithms; we use very simple ones to be self-contained
!> once better external Fortran libraries become available, we will switch
!> (c-library nlopt is a pain to integrate and fmin gets stuck in local minima)
module dislocdyn_opt
  implicit none
  abstract interface
    pure function callback_func(x) result(y)
    use dislocdyn_parameters, only : sel
      real(sel), intent(in) :: x
      real(sel) :: y
    end function
  end interface
  contains
    !> brute force algorithm: devide the 1D space u-l into n0 points, evaluate fct and find the smallest value
    !> then evaluate n points between i-1 and i+1 (where i was the index of the current fmin)
    !> repeat with n points until (fmin-fmin_old)<tol or maxiter iterations achieved
    pure function minimize_simple(fct,l,u,maxiter,tol,n,n0) result(x)
    use dislocdyn_parameters, only : sel
    use dislocdyn_utilities, only : linspace
    procedure(callback_func) :: fct
    real(sel), intent(in) :: l,u, tol
    integer, intent(in) :: maxiter, n, n0
    real(sel) :: x
    ! local vars:
    real(sel) :: c0(n0), y0(n0), c(n), y(n), w1, w2, fmin, fmin_old
    integer :: i, j, idx
    i = 0
    w1 = l
    w2 = u
    call linspace(l,u,n0,c0)
    do concurrent (j=1:n0)
      y0(j) = fct(c0(j))
    end do
    idx = minloc(y0,dim=1)
    fmin = y0(idx)
    x = c0(idx)
    w1 = c0(idx-1)
    w2 = c0(idx+1)
    if ((idx==1) .or. (idx==n0)) i = maxiter
    do while (i<maxiter)
      fmin_old = fmin
      call linspace(w1,w2,n,c)
      do concurrent (j=1:n)
        y(j) = fct(c(j))
      end do
      idx = minloc(y,dim=1)
      fmin = y(idx)
      x = c(idx)
      w1 = c(idx-1)
      w2 = c(idx+1)
      if (abs(fmin-fmin_old)<tol) then
        i = maxiter
      else if ((idx==1) .or. (idx==n)) then
        i = maxiter
      end if
    end do
    end function minimize_simple
end module dislocdyn_opt
