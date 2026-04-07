! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 30, 2026 - Apr. 7, 2026
module elastic_constants
  implicit none
  integer, parameter :: VoigtIndices(6)= (/1,5,9,6,3,2/), UnVoigtIndices(9)= (/1,6,5,6,2,4,5,4,3/)
  private VoigtIndices, UnVoigtIndices ! f2py-incompatibilities prevent us from making more stuff private
  public voigt, unvoigt, elasticC2, voigtaverage, reussaverage, hillaverage
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
          xij(:3) = cij(:3); xij(7) = cij(1); xij(8) = cij(3); xij(12) = cij(4)
          xij(16) = cij(5); xij(19) = cij(5); xij(21) = 0.5d0*(cij(1)-cij(2))
        case ("tetr")
          if (n/=6) then
            print*,"error: expected length 6"
            return
          end if
          xij(:3) = cij(:3); xij(7) = cij(1); xij(8) = cij(3); xij(12) = cij(4)
          xij(16) = cij(5); xij(19) = cij(5); xij(21) = cij(6)
        case ("trig")
          if (n/=6) then
            print*,"error: expected length 6"
            return
          end if
          xij(:4) = cij(:4); xij(7) = cij(1); xij(8) = cij(3); xij(9) = -cij(4); xij(12) = cij(5)
          xij(16) = cij(6); xij(19) = cij(6); xij(20) = cij(4); xij(21) = 0.5d0*(cij(1)-cij(2))
        case ("tetr2")
          if (n/=7) then
            print*,"error: expected length 7"
            return
          end if
          xij(:3) = cij(:3); xij(6) = cij(4); xij(7) = cij(1); xij(8) = cij(3); xij(11) = -cij(4); xij(12) = cij(5)
          xij(16) = cij(6); xij(19) = cij(6); xij(21) = cij(7)
        case ("ortho")
          if (n/=9) then
            print*,"error: expected length 9"
            return
          end if
          xij(:3) = cij(:3); xij(7) = cij(4); xij(8) = cij(5); xij(12) = cij(6)
          xij(16) = cij(7); xij(19) = cij(8); xij(21) = cij(9)
        case ("mono")
          if (n/=13) then
            print*,"error: expected length 13"
            return
          end if
          xij(:3) = cij(:3); xij(5) = cij(4); xij(7) = cij(5); xij(8) = cij(6); xij(10) = cij(7); xij(12) = cij(8)
          xij(14) = cij(9); xij(16) = cij(10); xij(18) = cij(11); xij(19) = cij(12); xij(21) = cij(13)
        case ("tric")
          if (n/=21) then
            print*,"error: expected length 21"
            return
          end if
          xij = cij
        case default
          print*,"Error: expected one of these keywords for sym: 'iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'ortho', 'mono', 'tric'."
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
    !-------------------------
    subroutine voigtaverage(C2,lambda,mu)
    use parameters, only: sel
    real(sel), intent(in) :: C2(6,6)
    real(sel), intent(out) :: lambda, mu
    ! expressions derived using PyDislocDyn (python implementation using a symbolic calculation with sym='tric')
    lambda = (C2(1,1)+4.d0*C2(1,2)+4.d0*C2(1,3)+C2(2,2)+4.d0*C2(2,3)+C2(3,3)-2.d0*C2(4,4)-2.d0*C2(5,5)-2.d0*C2(6,6))/15.d0
    mu = (C2(1,1)-C2(1,2)-C2(1,3)+C2(2,2)-C2(2,3)+C2(3,3)+3.d0*C2(4,4)+3.d0*C2(5,5)+3.d0*C2(6,6))/15.d0
    end subroutine voigtaverage
    subroutine reussaverage(C2,lambda,mu)
    use parameters, only: sel
    real(sel), intent(in) :: C2(6,6)
    real(sel), intent(out) :: lambda, mu
    ! expressions derived using PyDislocDyn (python implementation using a symbolic calculation with sym='tric')
    lambda = -3.d0 *(C2(1,1)*C2(3,3) + C2(1,2)*C2(3,3) - 2.d0*C2(1,3)**2)*(-C2(1,1)**2*C2(3,3)*C2(4,4) - 2.d0*C2(1,1)**2*C2(3,3)*C2(6,6) + 2.d0*C2(1,1)**2*C2(4,4)*C2(6,6) + 2.d0*C2(1,1)*C2(1,3)**2*C2(4,4) &
             + 4.d0*C2(1,1)*C2(1,3)**2*C2(6,6) - 16.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) + C2(1,2)**2*C2(3,3)*C2(4,4) + 2.d0*C2(1,2)**2*C2(3,3)*C2(6,6) - 2.d0*C2(1,2)**2*C2(4,4)*C2(6,6) &
             - 2.d0*C2(1,2)*C2(1,3)**2*C2(4,4) - 4.d0*C2(1,2)*C2(1,3)**2*C2(6,6) + 16.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) - 8.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,3)**2*C2(4,4)*C2(6,6))/((C2(1,1) + C2(1,2) &
             - 4.d0*C2(1,3) + 2*C2(3,3))*(3*C2(1,1)**2*C2(3,3)*C2(4,4) + 6.d0*C2(1,1)**2*C2(3,3)*C2(6,6) + 4.d0*C2(1,1)**2*C2(4,4)*C2(6,6) - 6.d0*C2(1,1)*C2(1,3)**2*C2(4,4) - 12.d0*C2(1,1)*C2(1,3)**2*C2(6,6) &
             + 8.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) + 8.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) - 3.d0*C2(1,2)**2*C2(3,3)*C2(4,4) - 6.d0*C2(1,2)**2*C2(3,3)*C2(6,6) - 4.d0*C2(1,2)**2*C2(4,4)*C2(6,6) &
             + 6.d0*C2(1,2)*C2(1,3)**2*C2(4,4) + 12.d0*C2(1,2)*C2(1,3)**2*C2(6,6) - 8.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) - 12.d0*C2(1,3)**2*C2(4,4)*C2(6,6)))
    mu = 15.d0*C2(4,4)*C2(6,6)*(C2(1,1)-C2(1,2))*(C2(1,1)*C2(3,3) + C2(1,2)*C2(3,3) - 2.d0*C2(1,3)**2)/(3.d0*C2(1,1)**2*C2(3,3)*C2(4,4) + 6.d0*C2(1,1)**2*C2(3,3)*C2(6,6) + 4.d0*C2(1,1)**2*C2(4,4)*C2(6,6) &
         - 6.d0*C2(1,1)*C2(1,3)**2*C2(4,4) - 12.d0*C2(1,1)*C2(1,3)**2*C2(6,6) + 8.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) + 8.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) - 3.d0*C2(1,2)**2*C2(3,3)*C2(4,4) - 6.d0*C2(1,2)**2*C2(3,3)*C2(6,6) &
         - 4.d0*C2(1,2)**2*C2(4,4)*C2(6,6) + 6.d0*C2(1,2)*C2(1,3)**2*C2(4,4) + 12.d0*C2(1,2)*C2(1,3)**2*C2(6,6) - 8.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) - 12.d0*C2(1,3)**2*C2(4,4)*C2(6,6))
    end subroutine reussaverage
    subroutine hillaverage(C2,lambda,mu)
    use parameters, only: sel
    real(sel), intent(in) :: C2(6,6)
    real(sel), intent(out) :: lambda, mu
    real(sel) :: lv,mv,lr,mr
    call voigtaverage(C2,lv,mv)
    call reussaverage(C2,lr,mr)
    lambda = 0.5d0*(lv+lr)
    mu = 0.5d0*(mv+mr)
    end subroutine hillaverage
end module elastic_constants
