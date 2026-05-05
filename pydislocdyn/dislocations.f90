! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - May 5, 2026
module dislocations
  use parameters, only : sel, pi ! defined in subroutines.f90
  use utilities, only : linspace, operator(.cross.) ! defined in subroutines.f90
  use various_subroutines, only : strohgeometry, computeuij ! defined in subroutines.f90
  use elastic_constants ! defined in elasticconstants.f90
  implicit none
  private
  !> The 'metalprops' derived type is used to store material information for a crystal.
  !> It represents the fortran version of the metalprops class found in PyDislocDyn, implementing a subset of the latter.
  type, public :: metalprops
    character(:), allocatable :: sym !< defines the symmetry via keyword
    character(:), allocatable :: metal !< metal is a name given to this instance
    real(sel), allocatable :: cij(:) !< store linearly independent 2nd order elastic constants only
    real(sel), allocatable :: cijk(:) !< store linearly independent 3rd order elastic constants only
    real(sel) :: rho=0.d0 !< material density
    real(sel) :: C2(6,6)=0.d0 !< tensor of 2nd order elastic constants in Voigt notation
    real(sel) :: C3(6,6,6)=0.d0 !< tensor of 3rd order elastic constants in Voigt notation
    real(sel) :: lat_a(3)=0.d0 !< lattice constants
    real(sel) :: lat_angles(3)=0.d0 !< angles between lattice constants
    real(sel) :: Vc=0.d0 !< unit cell volume
    !> Lame constants (polycryst. averages)
    real(sel) :: lam=0.d0, mu=0.d0
    real(sel) :: C2norm(3,3,3,3)=0.d0 !< will be used to store unvoigt(C2)/mu
    real(sel) :: Temp=300.d0 !< temperature associated with C2, rho, etc.
    contains
      procedure :: update_Vc => volume_unitcell ! define as type-bound procedure
  end type metalprops
  !> The 'disloc' derived type extends 'metalprops' by including information about a dislocation (slip plane etc.).
  !> It represents the fortran version of the Dislocation class found in PyDislocDyn, implementing a subset of the latter.
  !> Type-bound procedures include subroutines to calculate the dislocation displacement field and other properties.
  type, extends(metalprops), public :: disloc
    real(sel) :: b(3)=0.d0  !< Burgers vector
    real(sel) :: n0(3)=0.d0 !< slip plane normal
    real(sel) :: burgers=0.d0 !< Burgers vector length
    real(sel) :: beta=0.d0  !< ratio of gliding velocity over transverse sound speed
    integer :: ntheta=2 !< number of character angles between 0 and pi/2
    integer :: nphi=500 !< resolution in polar angle phi
    real(sel), allocatable :: theta(:), phi(:), rot(:,:,:), t(:,:)
    real(sel), allocatable :: m0(:,:), M(:,:,:), N(:,:,:), Cv(:,:,:,:,:)
    real(sel), allocatable :: uij(:,:,:,:)
    contains
      procedure :: update_theta => set_character_angles
      procedure :: update_stroh => computestroh
      procedure :: update_rot => computerot
      procedure :: init => init_disloc
      procedure :: update_uij => compute_uij
  end type
  public :: volume_unitcell, set_character_angles, computerot, phonondrag
  !-------------------------
  contains
    !> computes the unit cell volume
    subroutine volume_unitcell(mat)
      class(metalprops), intent(inout) :: mat
      select case (trim(mat%sym))
        case ("iso","cubic", "fcc", "bcc")
          mat%Vc = mat%lat_a(1)**3
        case ("hcp")
          mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)*sqrt(3.d0)*3.d0/2.d0
        case ("tetr","tetr2")
          mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)
        case ("orth")
          mat%Vc = mat%lat_a(1)*mat%lat_a(2)*mat%lat_a(3)
        case ("mono")
          mat%Vc = mat%lat_a(1)*mat%lat_a(2)*mat%lat_a(3)*sin(mat%lat_angles(2))
        case ("tric")
          mat%Vc = mat%lat_a(1)*mat%lat_a(2)*mat%lat_a(3)*sqrt(1.d0 - cos(mat%lat_angles(1))**2 - cos(mat%lat_angles(2))**2 &
                 - cos(mat%lat_angles(3))**2 + 2.d0*cos(mat%lat_angles(1))*cos(mat%lat_angles(2))*cos(mat%lat_angles(3)))
        case default
          print*,"error: not implemented"
          return
      end select
    end subroutine volume_unitcell
    !> initializes an array of dislocation character angles to be used in the computations
    subroutine set_character_angles(disl)
      class(disloc), intent(inout) :: disl
      ! todo: use select case to determine whether or not we need negative theta as well
      if (allocated(disl%theta)) deallocate(disl%theta)
      allocate(disl%theta(disl%ntheta))
      call linspace(0.d0,pi/2.d0,disl%ntheta,disl%theta)
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
    !>and its slip plane normal with y
    subroutine computerot(disl)
      class(disloc), intent(inout) :: disl
      integer :: th
      if (allocated(disl%rot)) deallocate(disl%rot)
      allocate(disl%rot(3,3,disl%ntheta))
      do th=1,disl%ntheta
        disl%rot(1,:,th) = disl%n0 .cross. disl%t(:,th)
        disl%rot(2,:,th) = disl%n0
        disl%rot(3,:,th) = disl%t(:,th)
      end do
    end subroutine computerot
    !-------------------------
    !> initializes various properties of the dislocation
    subroutine init_disloc(disl)
      class(disloc), intent(inout) :: disl
      real(sel) :: tmp_len
      select case (trim(disl%sym))
        case ("iso", "cubic", "fcc", "bcc")
          disl%lat_angles = 0.5d0*[pi,pi,pi]
        case default
          print*,"Error: keyword sym must be one of 'iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric'."
          return
      end select
      call volume_unitcell(disl)
      call disl%update_theta()
      ! next, normalize b, n0, and decide if we need to derive burgers
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
        print*,"ERROR: invalid slip system; b and n0 must be normal!"
      end if
      call elasticC2(disl%cij,disl%sym,disl%C2)
      call elasticC3(disl%cijk,disl%sym,disl%C3)
      call disl%update_stroh()
      call disl%update_rot()
      if (disl%mu<1.d-9) then
        select case (trim(disl%sym))
          case ("iso")
            disl%lam = disl%cij(1)
            disl%mu = disl%cij(2)
          case ("cubic", "fcc", "bcc")
            call kroeneraverage(disl%C2,disl%lam,disl%mu)
          case default
            call hillaverage(disl%C2,disl%lam,disl%mu)
        end select
      end if
      call unvoigt(disl%C2/disl%mu,disl%C2norm)
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
    !>Calculates the dislocation drag coefficient from phonon wind for all character angles defined in dislocation 'disl' 
    !>and gliding velocities 'beta'=v/ct
    subroutine phonondrag(drag,disl,beta,nphi,nq)
      use phononwind
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
                          .false.,-1.d0,.true.,dragTT)
        call phononwind_xy(fourieruij,A3rot,qBZ,cl,ct,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,-1.d0,.true.,dragLT)
        ntdyn = int((1.d0+0.5d0*beta(bt))*lent)
        call phononwind_xx(fourieruij,A3rot,qBZ,ct,cl,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,-1.d0,.true.,dragLL)
        call phononwind_xy(fourieruij,A3rot,qBZ,ct,cl,beta(bt),disl%burgers,disl%Temp,disl%ntheta,ntdyn,lenph,400,50,&
                          .false.,-1.d0,.true.,dragTL)
        drag(:,bt) = dragTT+dragLL+dragTL+dragLT
      end do
    end subroutine phonondrag
end module dislocations
