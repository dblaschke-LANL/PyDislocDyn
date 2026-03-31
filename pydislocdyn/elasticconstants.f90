! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - Mar. 30, 2026
module elastic_constants
  implicit none
  integer, parameter :: VoigtIndices(6)= (/1,5,9,6,3,2/), UnVoigtIndices(9)= (/1,6,5,6,2,4,5,4,3/)
  private VoigtIndices, UnVoigtIndices ! f2py-incompatibilities prevent us from making more stuff private
  public voigt, unvoigt
  interface voigt
    module procedure vgt_two, vgt_four, vgt_six
  end interface voigt
  interface unvoigt
    module procedure unvgt_one, unvgt_two, unvgt_three
  end interface unvoigt
  contains
    subroutine elasticC2(cij,sym,vc)
    ! takes a list of indep. elastic constants and returns the 2nd order tensor in Voigt notation
      use parameters, only : sel
      real(sel), intent(in) :: cij(:)
      character(*), intent(in) :: sym
      real(sel), intent(out) :: vc(6,6)
      real(sel) :: x, xij(21)
      integer :: i,j,n,ii,jj
      xij = 0.d0
      vc = 0.d0
      n = size(cij)
      select case (trim(sym))
        case ("iso")
          if (n/=2) then
            print*,"error: expected length 2"
            return
          end if
          x = cij(1)+2.d0*cij(2) ! user passes (c12,c44) in this case, so x=c11
          xij(1) = x; xij(7) = x; xij(12) = x
          xij(2) = cij(1); xij(3) = cij(1); xij(8) = cij(1)
          xij(16) = cij(2); xij(19) = cij(2); xij(21) = cij(2)
        case ("cubic")
          if (n/=3) then
            print*,"error: expected length 3"
            return
          end if
          xij(1) = cij(1); xij(7) = cij(1); xij(12) = cij(1)
          xij(2) = cij(2); xij(3) = cij(2); xij(8) = cij(2)
          xij(16) = cij(3); xij(19) = cij(3); xij(21) = cij(3)
        case ("hcp")
          if (n/=5) then
            print*,"error: expected length 5"
            return
          end if
          x = 0.5d0*(cij(1)-cij(2))
          xij(1) = cij(1); xij(7) = cij(1); xij(12) = cij(4)
          xij(2) = cij(2); xij(3) = cij(3); xij(8) = cij(3)
          xij(16) = cij(5); xij(19) = cij(5); xij(21) = x
        case ("tric")
          if (n/=21) then
            print*,"error: expected length 21"
            return
          end if
          xij = cij
        case default
          print*,"error: not implemented"
          return
      end select
      do i=1,6
        do j=1,6
          ii = min(i,j)
          jj = max(i,j)
          vc(i,j) = xij((jj-ii+1)+int((ii-1)*(14-ii)/2))!xij(int((ii-1)*ii/2)+jj)
        end do
      end do
    end subroutine elasticC2
    ! -----------------------------
    subroutine vgt_two(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(3,3)
      real(kind=sel), intent(out) :: y(6)
      real(kind=sel) :: z(9)
      z = reshape(x,(/9/))
      y = z(VoigtIndices)
    end subroutine vgt_two
    subroutine vgt_four(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(3,3,3,3)
      real(kind=sel), intent(out) :: y(6,6)
      real(kind=sel) :: z(9,9)
      z = reshape(x,(/9,9/))
      y = z(VoigtIndices,VoigtIndices)
    end subroutine vgt_four
    subroutine vgt_six(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(3,3,3,3,3,3)
      real(kind=sel), intent(out) :: y(6,6,6)
      real(kind=sel) :: z(9,9,9)
      z = reshape(x,(/9,9,9/))
      y = z(VoigtIndices,VoigtIndices,VoigtIndices)
    end subroutine vgt_six
    subroutine unvgt_one(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(6)
      real(kind=sel), intent(out) :: y(3,3)
      y = reshape(x(UnVoigtIndices),(/3,3/))
    end subroutine unvgt_one
    subroutine unvgt_two(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(6,6)
      real(kind=sel), intent(out) :: y(3,3,3,3)
      y = reshape(x(UnVoigtIndices,UnVoigtIndices),(/3,3,3,3/))
    end subroutine unvgt_two
    subroutine unvgt_three(x,y)
      use parameters, only : sel
      real(kind=sel), intent(in) :: x(6,6,6)
      real(kind=sel), intent(out) :: y(3,3,3,3,3,3)
      y = reshape(x(UnVoigtIndices,UnVoigtIndices,UnVoigtIndices),(/3,3,3,3,3,3/))
    end subroutine unvgt_three
end module elastic_constants
