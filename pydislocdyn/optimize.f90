! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 28, 2026 - Nov. 23, 2026

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
    c0 = linspace(l,u,n0)
    do concurrent (j=1:n0)
      y0(j) = fct(c0(j))
    end do
    idx = minloc(y0,dim=1)
    fmin = y0(idx)
    x = c0(idx)
    if ((idx==1) .or. (idx==n0)) then
      return
    end if
    w1 = c0(idx-1)
    w2 = c0(idx+1)
    do while (i<maxiter)
      fmin_old = fmin
      c = linspace(w1,w2,n)
      do concurrent (j=1:n)
        y(j) = fct(c(j))
      end do
      idx = minloc(y,dim=1)
      fmin = y(idx)
      x = c(idx)
      if ( (abs(fmin-fmin_old)<10.d0*tol) .or. (idx==1) .or. (idx==n) ) then
        return
      end if
      w1 = c(idx-1)
      w2 = c(idx+1)
    end do
    end function minimize_simple

    !> devide the 1D space u-l into n intervals, call fmin() (from fmin_module) for each interval and evaluate the fct(x) at
    !> the end points as well as the points separating the n intervals. Return x for the smallest result. If n=1, the only 
    !> difference to calling fmin() directly is the additional check if fct(l) or fct(u) is even smaller.
    !> Option 0<=olap<1/2 defines the fraction of an interval that is included also in its neighboring fmin() call
    function minimize_fmin(fct,l,u,tol,n,overlap) result(x)
    use dislocdyn_parameters, only : sel
    use dislocdyn_utilities, only : linspace
    use fmin_module, only: opt_fmin => fmin
    procedure(callback_func) :: fct
    real(sel), intent(in) :: l,u, overlap, tol
    integer, intent(in) :: n
    real(sel) :: x, olap
    ! local vars:
    real(sel) :: c0(n+1), y0(n+1), fmin, xn, fmin_new
    integer :: j, idx
    c0 = linspace(l,u,n+1)
    olap = overlap*(c0(2)-c0(1))
    do concurrent (j=1:n+1)
      y0(j) = fct(c0(j))
    end do
    idx = minloc(y0,dim=1)
    fmin = y0(idx)
    x = c0(idx)
    do j=1,n
      xn = opt_fmin(fct,c0(j)-olap,c0(j+1)+olap,tol)
      fmin_new = fct(xn)
      if (fmin_new<fmin) then
        fmin = fmin_new
        x = xn
      end if
    end do
    end function minimize_fmin
end module dislocdyn_opt
