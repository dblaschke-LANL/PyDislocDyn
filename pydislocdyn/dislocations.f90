! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - Aug. 11, 2026
module dislocdyn_dislocations
  use dislocdyn_parameters, only : sel, rzero, pi ! defined in subroutines.f90
  use dislocdyn_utilities, only : linspace, operator(.cross.) ! defined in subroutines.f90
  use dislocdyn_subroutines, only : strohgeometry, computeuij, computeEtot ! defined in subroutines.f90
  use dislocdyn_elasticconstants ! defined in elasticconstants.f90
  use dislocdyn_crystals, only : crystal
  implicit none
  private
  !> The 'disloc' derived type extends 'crystal' by including information about a dislocation (slip plane etc.).
  !> It represents the fortran version of the Dislocation class found in PyDislocDyn, implementing a subset of the latter.
  !> Type-bound procedures include subroutines to calculate the dislocation displacement field and other properties.
  type, extends(crystal), public :: disloc
    real(sel) :: b(3)=0.d0  !< Burgers vector
    real(sel) :: n0(3)=0.d0 !< slip plane normal
    real(sel) :: burgers=0.d0 !< Burgers vector length
    real(sel) :: beta=0.d0  !< ratio of gliding velocity over transverse sound speed
    integer :: ntheta=2 !< number of character angles between 0 and pi/2
    integer :: nphi=500 !< resolution in polar angle phi
    real(sel), allocatable :: theta(:), phi(:), rot(:,:,:), t(:,:), C2aligned(:,:,:)
    real(sel), allocatable :: m0(:,:), M(:,:,:), N(:,:,:), Cv(:,:,:,:,:)
    real(sel), allocatable :: uij(:,:,:,:)
    contains
      procedure :: update_slipplane => update_slipplane
      procedure :: update_theta => set_character_angles
      procedure :: update_stroh => computestroh
      procedure :: update_rot => computerot
      procedure :: init => init_disloc
      procedure :: update_uij => compute_uij
      procedure :: update_elasticE => compute_elasticE
      procedure :: computevcrit_screw => computevcrit_screw
      procedure :: computevcrit_edge => computevcrit_edge
      procedure :: computevcrit_barnett => computevcrit_barnett
      procedure :: computevcrit => computevcrit
  end type
  public :: set_character_angles, computerot, phonondrag, computevcrit_screw, computevcrit_edge, &
            computevcrit_barnett, computevcrit
  !-------------------------
  contains
    subroutine update_slipplane(disl,Millerb,Millern0)
      class(disloc), intent(inout) :: disl
      real(sel), intent(in) :: Millerb(:), Millern0(:)
      if (disl%burgers<rzero) then
        call disl%Miller_to_Cart(millerv=Millerb,normalize=.false.,reziprocal=.false.,v=disl%b)
      else
        call disl%Miller_to_Cart(millerv=Millerb,normalize=.true.,reziprocal=.false.,v=disl%b)
      end if
      call disl%Miller_to_Cart(millerv=Millern0,normalize=.true.,reziprocal=.true.,v=disl%n0)
    end subroutine update_slipplane
    !------------------------------
    !> initializes an array of dislocation character angles to be used in the computations
    !> default is to use character angles between 0 (screw) and +pi/2 (edge); set optional
    !> parameter positive_theta = .false. to swap signs of these character angles
    !> reason: not all slip systems are symmetric regarding positive and negative mixed
    !> dislocation character angles; in these cases, the user can initialize two dislocations,
    !> in two variables, one for positive theta, one for negative theta
    subroutine set_character_angles(disl,positive_theta)
      class(disloc), intent(inout) :: disl
      logical, optional :: positive_theta
      logical :: positive_character
      positive_character = .true.
      if (present(positive_theta)) then
        positive_character = positive_theta
      end if
      if (allocated(disl%theta)) deallocate(disl%theta)
      if (allocated(disl%C2aligned)) deallocate(disl%C2aligned)
      allocate(disl%theta(disl%ntheta),disl%C2aligned(6,6,disl%ntheta))
      call linspace(0.d0,pi/2.d0,disl%ntheta,disl%theta)
      if (.not. positive_character) then
        disl%theta = -disl%theta
      end if
    end subroutine set_character_angles
    !> computes several arrays to be used in the computation of a dislocation displacement gradient field for crystals
    !> using the integral version of the Stroh method
    subroutine computestroh(disl)
      class(disloc), intent(inout) :: disl
      if (allocated(disl%t)) deallocate(disl%t); if (allocated(disl%m0)) deallocate(disl%m0)
      if (allocated(disl%phi)) deallocate(disl%phi); if (allocated(disl%Cv)) deallocate(disl%Cv)
      if (allocated(disl%M)) deallocate(disl%M); if (allocated(disl%N)) deallocate(disl%N)
      allocate(disl%t(3,disl%ntheta),disl%m0(3,disl%ntheta),disl%phi(disl%nphi))
      allocate(disl%M(disl%nphi,3,disl%ntheta),disl%N(disl%nphi,3,disl%ntheta),disl%Cv(3,3,3,3,disl%ntheta))
      call linspace(0.d0,2.d0*pi,disl%nphi,disl%phi)
      call strohgeometry(disl%b,disl%n0,disl%t,disl%m0,disl%M,disl%N,disl%Cv,disl%theta,disl%phi,disl%ntheta,disl%nphi)
    end subroutine computestroh
    !>determines the rotation matrices necessary to align each dislocation of character angle theta with z 
    !>and its slip plane normal with y; then rotates the tensor of elastic constants C2 to align with each dislocation
    !>the latter array of aligne C2s is saved in Voigt notation as %C2aligned
    subroutine computerot(disl)
      class(disloc), intent(inout) :: disl
      real(sel) :: C2(3,3,3,3), C2aligned(3,3,3,3), rot(3,3)
      integer :: th, i, ii, j, jj
      if (allocated(disl%rot)) deallocate(disl%rot)
      allocate(disl%rot(3,3,disl%ntheta))
      call unvgt_two(disl%C2,C2)
      do th=1,disl%ntheta
        rot(1,:) = disl%n0 .cross. disl%t(:,th)
        rot(2,:) = disl%n0
        rot(3,:) = disl%t(:,th)
        disl%rot(:,:,th) = rot
        C2aligned = 0.d0
        do ii=1,3
          do i=1,3
            do jj=1,3
              do j=1,3
                C2aligned(:,:,j,i) = C2aligned(:,:,j,i) + matmul(matmul(rot,C2(:,:,jj,ii)),transpose(rot))*rot(j,jj)*rot(i,ii)
              end do
            end do
          end do
        end do
        call vgt_four(C2aligned,disl%C2aligned(:,:,th))
      end do
    end subroutine computerot
    !-------------------------
    !> initializes various properties of the dislocation
    subroutine init_disloc(disl,Millerb,Millern0,positive_theta)
      class(disloc), intent(inout) :: disl
      real(sel), optional :: Millerb(:), Millern0(:)
      logical, optional :: positive_theta
      real(sel) :: tmp_len
      call disl%init_crystal()
      if (present(positive_theta)) then
        call disl%update_theta(positive_theta = positive_theta)
      else
        call disl%update_theta()
      end if
      ! next, normalize b, n0, and decide if we need to derive burgers
      if (present(Millerb) .and. present(Millern0)) then
        call disl%update_slipplane(Millerb,Millern0)
      end if
      tmp_len = sqrt(dot_product(disl%n0,disl%n0))
      if (abs(tmp_len-1.d0)>1.d-9) then
        disl%n0 = disl%n0/tmp_len
      end if
      tmp_len = sqrt(dot_product(disl%b,disl%b))
      if (abs(tmp_len-1.d0)>1.d-9) then
        disl%b = disl%b/tmp_len
        if (disl%burgers<1.d-15) then
          disl%burgers = tmp_len ! infer from b unless set by user
        end if
      end if
      if (abs(dot_product(disl%b,disl%n0))>1.d-9) then
        error stop "invalid slip system; b and n0 must be normal!"
      end if
      call disl%update_stroh()
      call disl%update_rot()
    end subroutine init_disloc
    !-------------------------
    !>Computes the dislocation displacement gradient field according to the integral method
    subroutine compute_uij(disl)
      class(disloc), intent(inout) :: disl
      if (allocated(disl%uij)) deallocate(disl%uij)
      allocate(disl%uij(disl%nphi,3,3,disl%ntheta))
      call computeuij(disl%beta,disl%C2norm,disl%Cv,disl%b,disl%M,disl%N,disl%phi,disl%ntheta,disl%nphi,disl%uij)
    end subroutine compute_uij
    !-------------------------
    !> Computes the elastic self energy logarithmic prefactor of a straight dislocation uij moving at velocity beta.
    !> Specifically, the self energy is Etot*ln(R/r0) where Etot is computed in this method
    !> and R and r0 are the outer and inner radius, see Phil. Mag. 98 (2018) 2397.
    subroutine compute_elasticE(disl, Wtot)
      class(disloc), intent(in) :: disl
      real(sel), intent(out) :: Wtot(disl%ntheta)
      call computeEtot(disl%uij, disl%beta, disl%C2norm, disl%Cv, disl%phi, disl%ntheta, disl%nphi, Wtot)
    end subroutine compute_elasticE
    !-------------------------
    !>Calculate the limiting velocity for a pure screw dislocation assuming the plane perpendicular to the dislocation line
    !>is a reflection plane; Note: the reflection plane property must be checked separately, this function will not.
    subroutine computevcrit_screw(disl,vlim)
      class(disloc), intent(in) :: disl
      real(sel), intent(out) :: vlim
      vlim = sqrt((disl%C2aligned(5,5,1)-4.d0*disl%C2aligned(4,5,1)**2/(4.d0*disl%C2aligned(4,4,1)))/disl%rho)
    end subroutine computevcrit_screw
    !-------------------------
    !>Compute the limiting velocity of a pure edge dislocation, assuming the slip plane is a reflection plane.
    !>Note: the reflection plane property must be checked separately, this function will not.
    !>If elastic constants c16 and c26 are zero, we use the analytic solution of L. J. Teutonico 1961, Phys. Rev. 124:1039.
    !>Otherwise, we adapt the method of Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5, to the present decoupled 2D
    !>special case.
    subroutine computevcrit_edge(disl,vlim)
      use dislocdyn_opt, only : minimize_simple
      use dislocdyn_utilities, only : elbrak1d
      use dislocdyn_subroutines, only : edgevlim_of_phi
      class(disloc), intent(in) :: disl
      real(sel), intent(out) :: vlim
      real(sel) :: c11, c12, c22, c66, c16, c26, tmpvlim, norm
      real(sel) :: C2(3,3,3,3)
      c11=disl%C2aligned(1,1,disl%ntheta)
      c22=disl%C2aligned(2,2,disl%ntheta)
      c66=disl%C2aligned(6,6,disl%ntheta)
      c12=disl%C2aligned(1,2,disl%ntheta)
      c16=disl%C2aligned(1,6,disl%ntheta)
      c26=disl%C2aligned(2,6,disl%ntheta)
      vlim  = -1.d0 ! if this is returned, something went wrong below (e.g. C2 did not have the required properties)
      if (abs(c16/disl%C2(4,4))+abs(c26/disl%C2(4,4)) < 1.d-12) then
        vlim = sqrt(min(c66,c11)/disl%rho)
        if ((((c11*c22-c12**2-2.d0*c12*c66) - (c22+c66)*min(c66,c11))/(c22*c66))<0) then
          ! analytic solution to Re(lambda=0) in eq. (39) (with sp.solve); sqrt below is real because of if statement above:
          tmpvlim = (2.d0*sqrt(c22*c66*(-c11*c22 + c11*c66 + c12**2 + 2.d0*c12*c66 + c22*c66))*(c12 + c66) - &
                    (-c11*c22**2 + c11*c22*c66 + c12**2*c22 + c12**2*c66 + 2.d0*c12*c22*c66 + 2.d0*c12*c66**2 + &
                     2.d0*c22*c66**2))/((c22 - c66)**2)
          vlim = min(vlim,sqrt(tmpvlim/disl%rho))
        end if
      else
        call unvoigt(disl%C2aligned(:,:,disl%ntheta)/disl%C2(4,4),C2)
        norm=(disl%C2(4,4)/disl%rho)
        vlim = f_edge_l(minimize_simple(f_edge_l,-0.5d0*pi,0.5d0*pi,2,1.d-4,int(disl%nphi/5),20))
      end if
      contains
        pure function f_edge_l(x) result(y)
          real(sel), intent(in) :: x
          real(sel) :: y
          y = edgevlim_of_phi(x,-1,C2,norm)
        end function f_edge_l
    end subroutine computevcrit_edge
    !-------------------------
    !> Computes the limiting velocities following Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5.
    pure subroutine computevcrit_barnett(disl,th,vlim)
      use dislocdyn_opt, only : minimize_simple
      use dislocdyn_utilities, only : elbrak1d
      use dislocdyn_subroutines, only : vlim_of_phi
      class(disloc), intent(in) :: disl
      integer, intent(in) :: th !< index of the character angle disl%theta(th)
      real(sel), intent(out) :: vlim(3) !< 3 branches to consider, the lowest value is the (lowest) limiting velocity
      real(sel) :: norm, C2(3,3,3,3), tmp, ub, lb
      integer :: i, j
      call unvoigt(disl%C2/disl%C2(4,4),C2)
      norm=(disl%C2(4,4)/disl%rho)
      lb = -0.5d0*pi
      ub = 0.5d0*pi
      vlim(1) = f1(minimize_simple(f1,lb,ub,2,1.d-4,int(disl%nphi/5),60))
      vlim(2) = f2(minimize_simple(f2,lb,ub,2,1.d-4,int(disl%nphi/5),40))
      vlim(3) = f3(minimize_simple(f3,lb,ub,2,1.d-4,int(disl%nphi/5),20))
      if (.not. ((vlim(1)<=vlim(2)) .and. (vlim(2)<=vlim(3)))) then
        do i = 1, 2
          do j = i + 1, 3
            if (vlim(i) > vlim(j)) then
              ! print*,"sorting barnett result"
              tmp = vlim(i)
              vlim(i) = vlim(j)
              vlim(j) = tmp
            end if
          end do
        end do
      end if
      contains
        pure function f1(x) result(y)
          real(sel), intent(in) :: x
          real(sel) :: y
          y = vlim_of_phi(x,1,C2,norm,disl%m0(:,th),disl%n0)
        end function f1
        pure function f2(x) result(y)
          real(sel), intent(in) :: x
          real(sel) :: y
          y = vlim_of_phi(x,2,C2,norm,disl%m0(:,th),disl%n0)
        end function f2
        pure function f3(x) result(y)
          real(sel), intent(in) :: x
          real(sel) :: y
          y = vlim_of_phi(x,3,C2,norm,disl%m0(:,th),disl%n0)
        end function f3
    end subroutine computevcrit_barnett
    !-------------------------
    !> Computes all limiting velocities for all dislocation character angles
    !> In the special case of reflection symmetry for the screw or edge dislocation, the appropriate values from the 
    !> general numerical algorithm computevcrit_barnett() will be replaced by the analytic results of computevcrit screw/edge
    !> the results are assembled into a 2-dim array where each row represents the 3 branches (lowest, 2nd and highest vlim) for
    !> a character angle theta = [0,...,disl%ntheta]
    subroutine computevcrit(disl,vlim)
      class(disloc), intent(in) :: disl
      real(sel), intent(out) :: vlim(disl%ntheta,3)
      real(sel) :: tmp
      integer th
      !$OMP PARALLEL DO
      do th=1,disl%ntheta
        call computevcrit_barnett(disl,th,vlim(th,:))
      end do
      !$OMP END PARALLEL DO
      tmp = 0.d0
      if (CheckReflectionSymmetry(disl%C2aligned(:,:,1))) then
        call disl%computevcrit_screw(tmp)
        vlim(1,:) = tmp
      end if
      if (CheckReflectionSymmetry(disl%C2aligned(:,:,disl%ntheta))) then
        call disl%computevcrit_edge(tmp)
        vlim(disl%ntheta,:2) = tmp
      !else if (disl%sym=="fcc") then  ! would need to check the slip system as well and computevcrit_barnett() is accurate enough
      !  vlim(1) = sqrt(min(disl%C2(4,4),0.5d0*(disl%C2(1,1)-disl%C2(1,2)))/disl%rho)
      end if
    end subroutine computevcrit
    !-------------------------
    !>Calculates the dislocation drag coefficient from phonon wind for all character angles defined in dislocation 'disl' 
    !>and gliding velocities 'beta'=v/ct
    subroutine phonondrag(drag,disl,beta,nphi,nq)
      use dislocdyn_phononwind
      class(disloc), intent(in) :: disl
      real(sel), intent(in) :: beta(:)
      real(sel), intent(out), allocatable :: drag(:,:)
      integer, optional :: nphi, nq
      real(sel) :: uij(disl%nphi,3,3,disl%ntheta), C3norm(3,3,3,3,3,3), A3(3,3,3,3,3,3), A3rot(3,3,3,3,3,3,disl%ntheta)
      real(sel) :: rot(3,3), uijaligned(disl%nphi,3,3,disl%ntheta), ct, cl, qBZ
      real(sel), allocatable :: phi(:), q(:), sincos(:,:), fourieruij(:,:,:,:), dragTT(:), dragLL(:), dragTL(:), dragLT(:)
      integer :: lenph, lenq, th, nbeta, bt, lent, ntdyn, i, ii, j, jj, k, kk, l, ll
      if (present(nphi)) then
        lenph = nphi
      else
        lenph = 50
      end if
      if (present(nq)) then
        lenq = nq
      else
        lenq = 50
      end if
      nbeta = size(beta)
      lent = 321 ! todo: make this user-configurable
      allocate(phi(lenph),q(lenq),sincos(disl%nphi,lenph),fourieruij(lenph,3,3,disl%ntheta),drag(disl%ntheta,nbeta))
      allocate(dragTT(disl%ntheta),dragLL(disl%ntheta),dragLT(disl%ntheta),dragTL(disl%ntheta))
      ct = sqrt(disl%mu/disl%rho)
      cl = sqrt((disl%lam+2.d0*disl%mu)/disl%rho)
      qBZ = (6.d0*pi**2/disl%Vc)**(1.d0/3.d0)
      call unvoigt(disl%C3/disl%mu,C3norm)
      call elasticA3(disl%C2norm, C3norm, A3)
      ! -- some additional preparations for anisotropic case:
      call linspace(0.d0,2.d0*pi,lenph,phi)
      call linspace(0.d0,1.d0,lenq,q)
      call fourieruij_sincos(sincos,0.d0,250.d0*pi,disl%phi,q(4:lenq-4),phi,disl%nphi,lenq-7,lenph)
      A3rot = 0.d0
      do concurrent (th=1:disl%ntheta)! local(i, ii, j, jj, k, kk, l, ll, rot) shared(disl,A3rot,A3) ! requires gfortran>=15
        rot = disl%rot(:,:,th)
        do ii=1,3
          do i=1,3
            do jj=1,3
              do j=1,3
                do kk=1,3
                  do k=1,3
                    do ll=1,3
                      do l=1,3
                        A3rot(:,:,l,k,j,i,th) = A3rot(:,:,l,k,j,i,th) + matmul(matmul(rot,A3(:,:,ll,kk,jj,ii)),transpose(rot)) &
                                                      *rot(l,ll)*rot(k,kk)*rot(j,jj)*rot(i,ii)
                      end do
                    end do
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
      ! ---
      do bt=1,nbeta
        call computeuij(beta(bt),disl%C2norm,disl%Cv,disl%b,disl%M,disl%N,disl%phi,disl%ntheta,disl%nphi,uij)
        do th=1,disl%ntheta
          rot = disl%rot(:,:,th)
          do i=1,disl%nphi
            uijaligned(i,:,:,th) = matmul(matmul(rot,uij(i,:,:,th)),transpose(rot))
          end do
        end do
        call fourieruij_nocut(fourieruij,uijaligned,disl%phi,sincos,disl%ntheta,lenph,disl%nphi)
        ntdyn = int((1.d0+beta(bt))*lent)
        call phononwind_xx(fourieruij,A3rot,qBZ,ct,0.d0,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,[1,0],-1.d0,.true.,dragTT)
        call phononwind_xy(fourieruij,A3rot,qBZ,cl,ct,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,[1,0],-1.d0,.true.,dragLT)
        ntdyn = int((1.d0+0.5d0*beta(bt))*lent)
        call phononwind_xx(fourieruij,A3rot,qBZ,ct,cl,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,[1,0],-1.d0,.true.,dragLL)
        call phononwind_xy(fourieruij,A3rot,qBZ,ct,cl,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,[1,0],-1.d0,.true.,dragTL)
        drag(:,bt) = dragTT+dragLL+dragTL+dragLT
      end do
    end subroutine phonondrag
end module dislocdyn_dislocations
