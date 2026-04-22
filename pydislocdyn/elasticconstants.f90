! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 30, 2026 - Apr. 22, 2026
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
        case ("cubic", "fcc", "bcc")
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
        case ("orth","ortho")
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
          print*,"Error: keyword sym must be one of 'iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric'."
          return
      end select
      do i=1,6
        do j=1,6
          ii = min(i,j)
          jj = max(i,j)
          vc(i,j) = xij((jj-ii+1)+int((ii-1)*(14-ii)/2))
        end do
      end do
    end subroutine elasticC2
    !-------------------------
    subroutine elasticC3(cijk,sym,vc)
    ! takes a list of indep. elastic constants and returns the 2nd order tensor in Voigt notation
      use parameters, only : sel
      real(sel), intent(in) :: cijk(:)
      character(*), intent(in) :: sym
      real(sel), intent(out) :: vc(6,6,6)
      real(sel) :: xijk(56)
      integer :: i,j,k,n,ii,jj,kk,iind(6)
      iind = (/0,15,25,31,34,35/)
      xijk = 0.d0
      vc = 0.d0
      n = size(cijk)
      select case (trim(sym))
        case ("iso")
          if (n/=3) then
            print*,"error: expected length 3"
            return
          end if
          ! user passes (c123,c144,c456) in this case:
          xijk(8) = cijk(1); xijk(16) = cijk(2); xijk(51) = cijk(3)
          xijk(2) = xijk(8) + 2.d0*xijk(16) ! c112 = c123 + 2*c144
          xijk(21) = xijk(16) + 2.d0*xijk(51)! c166 = c144 + 2*c456
          xijk(1) = xijk(8) + 6.d0*xijk(16) + 8*xijk(51) ! c111 = c123 + 6*c144 + 8*c456
          xijk(3)=xijk(2); xijk(7)=xijk(2); xijk(12)=xijk(2); xijk(23)=xijk(2); xijk(27)=xijk(2) ! c113=c122=c133=c223=c233=c112
          xijk(19)=xijk(21); xijk(31)=xijk(21); xijk(36)=xijk(21); xijk(41)=xijk(21); xijk(44)=xijk(21) ! c155=c244=c266=c344=c355=c166
          xijk(22)=xijk(1); xijk(37)=xijk(1) ! c222=c333=c111
          xijk(34)=xijk(16); xijk(46)=xijk(16) ! c255=c366=c144
        case ("cubic", "fcc", "bcc")
          if (n/=6) then
            print*,"error: expected length 6"
            return
          end if
          ! user passes (c111,c112,c123,c144,c166,c456) in this case:
          xijk(1) = cijk(1); xijk(2) = cijk(2); xijk(8) = cijk(3)
          xijk(16) = cijk(4); xijk(21) = cijk(5); xijk(51) = cijk(6)
          xijk(3)=xijk(2); xijk(7)=xijk(2); xijk(12)=xijk(2); xijk(23)=xijk(2); xijk(27)=xijk(2) ! c113=c122=c133=c223=c233=c112
          xijk(19)=xijk(21); xijk(31)=xijk(21); xijk(36)=xijk(21); xijk(41)=xijk(21); xijk(44)=xijk(21) ! c155=c244=c266=c344=c355=c166
          xijk(22)=xijk(1); xijk(37)=xijk(1) ! c222=c333=c111
          xijk(34)=xijk(16); xijk(46)=xijk(16) ! c255=c366=c144
        case ("hcp")
          if (n/=10) then
            print*,"error: expected length 10"
            return
          end if
          ! user passes (c111,c112,c113,c123,c133,c144,c155,c222,c333,c344) in this case:
          xijk(:3) = cijk(:3); xijk(8) = cijk(4); xijk(12) = cijk(5); xijk(16) = cijk(6); xijk(19) = cijk(7)
          xijk(22) = cijk(8); xijk(37) = cijk(9); xijk(41) = cijk(10)
          xijk(23) = cijk(3); xijk(27) = cijk(5); xijk(31) = cijk(7); xijk(34) = cijk(6); xijk(44) = cijk(10)
          xijk(7) = (xijk(1)+xijk(2)-xijk(22)) !c122 = c111+c112-c222
          xijk(21) = 0.25d0*(3.d0*xijk(22)-xijk(2)-2.d0*xijk(1)) !c166 = (3*c222-c112-2*c111)/4
          xijk(36) = 0.25d0*(2*xijk(1)-xijk(2)-xijk(22)) !c266 = (2*c111-c112-c222)/4
          xijk(46) = 0.5d0*(xijk(3)-xijk(8)) !c366 = (c113-c123)/2
          xijk(51) = 0.5d0*(xijk(19)-xijk(16)) !c456 = (c155-c144)/2
        case ("tetr")
          if (n/=12) then
            print*,"error: expected length 12"
            return
          end if
          ! user passes (c111,c112,c113,c123,c133,c144,c155,c166,c333,c344,c366,c456) in this case:
          xijk(:3) = cijk(:3); xijk(8) = cijk(4); xijk(12) = cijk(5); xijk(16) = cijk(6); xijk(19) = cijk(7)
          xijk(21) = cijk(8); xijk(37) = cijk(9); xijk(41) = cijk(10); xijk(46) = cijk(11); xijk(51) = cijk(12)
          xijk(22)=xijk(1); xijk(7) = xijk(2); xijk(23) = xijk(3); xijk(27) = xijk(12) ! c222=c111, c122=c112, c223=c113, c233=c133
          xijk(31) = xijk(19); xijk(34) = xijk(16); xijk(36) = xijk(21); xijk(44) = xijk(41) ! c244=c155, c255=c144, c266=c166, c355=c344
        case ("trig")
          if (n/=14) then
            print*,"error: expected length 14"
            return
          end if
          xijk(:4) = cijk(:4); xijk(7)=(cijk(1)+cijk(2)-cijk(11)); xijk(8)=cijk(5); xijk(9)=cijk(6); xijk(12)=cijk(7)
          xijk(13)=cijk(8); xijk(16)=cijk(9); xijk(19)=cijk(10); xijk(20)=0.5d0*(cijk(4)+3.d0*cijk(6))
          xijk(21)=0.25d0*(3.d0*cijk(11)-2.d0*cijk(1)-cijk(2)); xijk(22)=cijk(11); xijk(23)=cijk(3); xijk(24)=-cijk(4)-2.d0*cijk(6)
          xijk(27)=cijk(7); xijk(28)=-cijk(8); xijk(31)=cijk(10); xijk(34)=cijk(9); xijk(35)=0.5d0*(cijk(4)-cijk(6))
          xijk(36)=0.25d0*(2.d0*cijk(1)-cijk(2)-cijk(11)); xijk(37)=cijk(12); xijk(41)=cijk(13); xijk(44)=cijk(13)
          xijk(45)=cijk(8); xijk(46)=0.5d0*(cijk(3)-cijk(5)); xijk(47)=cijk(14); xijk(50)=-cijk(14)
          xijk(51)=0.5d0*(cijk(10)-cijk(9)); xijk(52)=cijk(6)
        case ("tetr2")
          if (n/=16) then
            print*,"error: expected length 16"
            return
          end if
          xijk(:3) = cijk(:3); xijk(6)=cijk(4); xijk(7)=cijk(2); xijk(8)=cijk(5); xijk(12)=cijk(6); xijk(15:17)=cijk(7:9)
          xijk(19)=cijk(10); xijk(21)=cijk(11); xijk(22)=cijk(1); xijk(23)=cijk(3)
          xijk(26)=-cijk(4); xijk(27)=cijk(6); xijk(30)=-cijk(7); xijk(31)=cijk(10); xijk(32)=-cijk(9); xijk(34)=cijk(8)
          xijk(36:37)=cijk(11:12); xijk(41)=cijk(13); xijk(44)=cijk(13); xijk(46)=cijk(14); xijk(49)=cijk(15)
          xijk(51)=cijk(16); xijk(54)=-cijk(15)
        case ("orth", "ortho")
          if (n/=20) then
            print*,"error: expected length 20"
            return
          end if
          xijk(:3) = cijk(:3); xijk(7:8)=cijk(4:5); xijk(12)=cijk(6); xijk(16)=cijk(7); xijk(19)=cijk(8)
          xijk(21:23)=cijk(9:11); xijk(27)=cijk(12); xijk(31)=cijk(13); xijk(34)=cijk(14)
          xijk(36:37)=cijk(15:16); xijk(41)=cijk(17); xijk(44)=cijk(18); xijk(46)=cijk(19); xijk(51)=cijk(20)
        case ("mono")
          if (n/=32) then
            print*,"error: expected length 32"
            return
          end if
          xijk(:3) = cijk(:3); xijk(5)=cijk(4); xijk(7)=cijk(5); xijk(8)=cijk(6); xijk(10)=cijk(7); xijk(12)=cijk(8)
          xijk(14)=cijk(9); xijk(16)=cijk(10); xijk(18)=cijk(11); xijk(19)=cijk(12); xijk(21:23)=cijk(13:15)
          xijk(25)=cijk(16); xijk(27)=cijk(17); xijk(29)=cijk(18); xijk(31)=cijk(19); xijk(33)=cijk(20); xijk(34)=cijk(21)
          xijk(36:37)=cijk(22:23); xijk(39)=cijk(24); xijk(41)=cijk(25); xijk(43:44)=cijk(26:27); xijk(46)=cijk(28)
          xijk(48)=cijk(29); xijk(51)=cijk(30); xijk(53)=cijk(31); xijk(55)=cijk(32)
        case ("tric")
          if (n/=56) then
            print*,"error: expected length 56"
            return
          end if
          xijk = cijk
        case default
          print*,"Error: keyword sym must be one of 'iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric'."
          return
      end select
      do concurrent (i=1:6)! local(j,k,ii,jj,kk) shared(vc,xijk) local_init(iind)
        do j=1,6
          do k=1,6
            ii = min(i,j,k)
            jj = min(max(i,j),max(i,k),max(j,k))
            kk = max(i,j,k)
            vc(i,j,k) = xijk(kk-jj+1+int((14-jj)*(jj-1)/2)+iind(ii))
          end do
        end do
      end do
    end subroutine elasticC3
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
    lambda = -3.d0 *(C2(1,1)*C2(3,3) + C2(1,2)*C2(3,3) - 2.d0*C2(1,3)**2)*(-C2(1,1)**2*C2(3,3)*C2(4,4) - 2.d0*C2(1,1)**2 &
             *C2(3,3)*C2(6,6) + 2.d0*C2(1,1)**2*C2(4,4)*C2(6,6) + 2.d0*C2(1,1)*C2(1,3)**2*C2(4,4) &
             + 4.d0*C2(1,1)*C2(1,3)**2*C2(6,6) - 16.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) &
             + C2(1,2)**2*C2(3,3)*C2(4,4) + 2.d0*C2(1,2)**2*C2(3,3)*C2(6,6) - 2.d0*C2(1,2)**2*C2(4,4)*C2(6,6) &
             - 2.d0*C2(1,2)*C2(1,3)**2*C2(4,4) - 4.d0*C2(1,2)*C2(1,3)**2*C2(6,6) + 16.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) &
             - 8.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,3)**2*C2(4,4)*C2(6,6))/((C2(1,1) + C2(1,2) &
             - 4.d0*C2(1,3) + 2*C2(3,3))*(3*C2(1,1)**2*C2(3,3)*C2(4,4) + 6.d0*C2(1,1)**2*C2(3,3)*C2(6,6) + 4.d0*C2(1,1)**2 &
             *C2(4,4)*C2(6,6) - 6.d0*C2(1,1)*C2(1,3)**2*C2(4,4) - 12.d0*C2(1,1)*C2(1,3)**2*C2(6,6) &
             + 8.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) + 8.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) - 3.d0*C2(1,2)**2*C2(3,3)*C2(4,4) &
             - 6.d0*C2(1,2)**2*C2(3,3)*C2(6,6) - 4.d0*C2(1,2)**2*C2(4,4)*C2(6,6) &
             + 6.d0*C2(1,2)*C2(1,3)**2*C2(4,4) + 12.d0*C2(1,2)*C2(1,3)**2*C2(6,6) - 8.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) &
             + 4.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) - 12.d0*C2(1,3)**2*C2(4,4)*C2(6,6)))
    mu = 15.d0*C2(4,4)*C2(6,6)*(C2(1,1)-C2(1,2))*(C2(1,1)*C2(3,3) + C2(1,2)*C2(3,3) - 2.d0*C2(1,3)**2)/(3.d0*C2(1,1)**2 &
         *C2(3,3)*C2(4,4) + 6.d0*C2(1,1)**2*C2(3,3)*C2(6,6) + 4.d0*C2(1,1)**2*C2(4,4)*C2(6,6) &
         - 6.d0*C2(1,1)*C2(1,3)**2*C2(4,4) - 12.d0*C2(1,1)*C2(1,3)**2*C2(6,6) + 8.d0*C2(1,1)*C2(1,3)*C2(4,4)*C2(6,6) &
         + 8.d0*C2(1,1)*C2(3,3)*C2(4,4)*C2(6,6) - 3.d0*C2(1,2)**2*C2(3,3)*C2(4,4) - 6.d0*C2(1,2)**2*C2(3,3)*C2(6,6) &
         - 4.d0*C2(1,2)**2*C2(4,4)*C2(6,6) + 6.d0*C2(1,2)*C2(1,3)**2*C2(4,4) + 12.d0*C2(1,2)*C2(1,3)**2*C2(6,6) &
         - 8.d0*C2(1,2)*C2(1,3)*C2(4,4)*C2(6,6) + 4.d0*C2(1,2)*C2(3,3)*C2(4,4)*C2(6,6) - 12.d0*C2(1,3)**2*C2(4,4)*C2(6,6))
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
    !--------------------------
    subroutine kroeneraverage(C2,lambda,mu)
    !! Warning: use for cubic crystals only - we do not check C2 for cubic symmetry in this routine!
    use parameters, only: sel
    real(sel), intent(in) :: C2(6,6)
    real(sel), intent(out) :: lambda, mu
    real(sel) :: C11, C12, C44
    complex(sel) :: zero, m3
    C11 = C2(1,1); C12 = C2(1,2); C44 = C2(4,4)
    zero = cmplx(0.d0,0.d0,kind=sel)
    m3 = -5.d0*C11/24.d0 - C12/6.d0 - (21.d0*C11*C44/8.d0 - 3.d0*C12*C44/2.d0 + (5.d0*C11/8.d0 + C12/2.d0)**2)/(3.d0&
         *cmplx(-0.5d0,0.5d0*sqrt(3.d0),sel)*(-27.d0*C11**2*C44/16.d0 - 27.d0*C11*C12*C44/16.d0 + 27.d0*C12**2*C44/8.d0 &
         + (5.d0*C11/8.d0 + C12/2.d0)**3 - (45.d0*C11/8.d0 + 9.d0*C12/2.d0)*(-7.d0*C11*C44/8.d0 + C12*C44/2.d0)/2.d0 &
         + sqrt(zero-4.d0*(21.d0*C11*C44/8.d0 - 3.d0*C12*C44/2.d0 + (5.d0*C11/8.d0 + C12/2.d0)**2)**3 + (-27.d0*C11**2*C44/8.d0 &
         - 27.d0*C11*C12*C44/8.d0 + 27.d0*C12**2*C44/4.d0 + 2.d0*(5.d0*C11/8.d0 + C12/2.d0)**3 - (45.d0*C11/8.d0 + 9.d0*C12/2.d0)&
         *(-7.d0*C11*C44/8.d0 + C12*C44/2.d0))**2)/2.d0)**(1.d0/3.d0)) - cmplx(-0.5d0,0.5d0*sqrt(3.d0),sel)&
         *(-27.d0*C11**2*C44/16.d0 - 27.d0*C11*C12*C44/16.d0 + 27.d0*C12**2*C44/8.d0 + (5.d0*C11/8.d0 + C12/2.d0)**3 &
         - (45.d0*C11/8.d0 + 9.d0*C12/2.d0)*(-7.d0*C11*C44/8.d0 + C12*C44/2.d0)/2.d0 + sqrt(zero-4.d0*(21.d0*C11*C44/8.d0 &
         - 3.d0*C12*C44/2.d0 + (5.d0*C11/8.d0 + C12/2.d0)**2)**3 + (-27.d0*C11**2*C44/8.d0 - 27.d0*C11*C12*C44/8.d0 &
         + 27.d0*C12**2*C44/4.d0 + 2.d0*(5.d0*C11/8.d0 + C12/2.d0)**3 - (45.d0*C11/8.d0 + 9.d0*C12/2.d0)*(-7.d0*C11*C44/8.d0 &
         + C12*C44/2.d0))**2)/2.d0)**(1.d0/3.d0)/3.d0
    if (abs(m3%im/C44)>1.d-12) then
      print*,"ERROR: solution is imaginary, i.e. mu/c44=",m3/C44
    end if
    mu = m3%re
    lambda = (C2(1,1) + 2.d0*C2(1,2) - 2.d0*mu)/3.d0
    end subroutine kroeneraverage
end module elastic_constants
