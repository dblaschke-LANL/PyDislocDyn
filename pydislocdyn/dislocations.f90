! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - Apr. 8, 2026
module dislocations
  use parameters, only : sel, pi ! defined in subroutines.f90
  use elastic_constants ! defined in elasticconstants.f90
  implicit none
  private
  type, public :: metalprops
    character(:), allocatable :: sym, metal ! sym defines the symmetry via keyword, metal is a name given to this instance
    real(sel), allocatable :: cij(:), cijk(:) ! store linearly independent elastic constants only
    real(sel) :: rho=0.d0, C2(6,6)=0.d0, C3(6,6,6)=0.d0 ! density and elastic constants
    real(sel) :: lat_a(3)=0.d0, lat_angles(3)=0.d0 ! lattice constants and angles
    real(sel) :: Vc=0.d0 ! unit cell volume
    contains
      procedure :: update_Vc => volume_unitcell ! define as type-bound procedure
  end type metalprops
  type, extends(metalprops), public :: disloc
    real(sel) :: b(3)=0.d0, n0(3)=0.d0 ! Burgers vector and slip plane normal
    real(sel) :: burgers=0.d0 ! Burgers vector length
    integer :: ntheta=2, nphi=500 ! number of character angles between 0 and pi/2; resolution in polar angle phi
    real(sel), allocatable :: theta(:), phi(:), rot(:,:,:), t(:,:)
    real(sel), allocatable :: m0(:,:), M(:,:,:), N(:,:,:), Cv(:,:,:,:,:)
    contains
      procedure :: update_theta => set_character_angles
      procedure :: update_stroh => computestroh
      procedure :: update_rot => computerot
      procedure :: init => init_disloc
  end type
  public :: volume_unitcell, set_character_angles, computerot
  contains
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
    subroutine set_character_angles(disl)
      class(disloc), intent(inout) :: disl
      ! todo: use select case to determine whether or not we need negative theta as well
      if (allocated(disl%theta)) deallocate(disl%theta)
      allocate(disl%theta(disl%ntheta))
      call linspace(0.d0,pi/2.d0,disl%ntheta,disl%theta)
    end subroutine set_character_angles
    subroutine computestroh(disl)
      class(disloc), intent(inout) :: disl
      if (allocated(disl%t)) deallocate(disl%t); if (allocated(disl%m0)) deallocate(disl%m0)
      if (allocated(disl%phi)) deallocate(disl%phi); if (allocated(disl%Cv)) deallocate(disl%Cv)
      if (allocated(disl%M)) deallocate(disl%M); if (allocated(disl%N)) deallocate(disl%N)
      allocate(disl%t(3,disl%ntheta),disl%m0(3,disl%ntheta),disl%phi(disl%nphi))
      allocate(disl%M(disl%nphi,3,disl%ntheta),disl%N(disl%nphi,3,disl%ntheta),disl%Cv(3,3,3,3,disl%ntheta))
      call strohgeometry(disl%b,disl%n0,disl%t,disl%m0,disl%M,disl%N,disl%Cv,disl%theta,disl%phi,disl%ntheta,disl%nphi)
    end subroutine computestroh
    subroutine computerot(disl)
      class(disloc), intent(inout) :: disl
      integer :: th
      if (allocated(disl%rot)) deallocate(disl%rot)
      allocate(disl%rot(3,3,disl%ntheta))
      do th=1,disl%ntheta
        call cross(disl%n0,disl%t(:,th),disl%rot(1,:,th))
        disl%rot(2,:,th) = disl%n0
        disl%rot(3,:,th) = disl%t(:,th)
      end do
    end subroutine computerot
    subroutine init_disloc(disl)
      class(disloc), intent(inout) :: disl
      real(sel) :: tmp_len
      select case (trim(disl%sym))
        case ("iso", "cubic", "fcc", "bcc")
          disl%lat_angles = 0.5d0*(/pi,pi,pi/)
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
    end subroutine init_disloc
end module dislocations
