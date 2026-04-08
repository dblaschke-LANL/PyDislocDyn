! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - Apr. 7, 2026
module dislocations
  use parameters, only : sel, pi ! defined in subroutines.f90
  implicit none
  private
  type, public :: metalprops
    character(:), allocatable :: sym, metal ! sym defines the symmetry via keyword, metal is a name given to this instance
    real(sel), allocatable :: cij(:) ! store linearly independent elastic constants only
    real(sel) :: rho, C2(6,6), C3(6,6,6) ! density and elastic constants
    real(sel) :: lat_a(3), lat_angles(3) ! lattice constants and angles
    real(sel) :: Vc ! unit cell volume
    contains
      procedure :: update_Vc => volume_unitcell ! define as type-bound procedure
  end type metalprops
  type, extends(metalprops), public :: disloc
    real(sel) :: b(3), n0(3) ! Burgers vector and slip plane normal
    real(sel) :: burgers ! Burgers vector length
    integer :: ntheta=2, nphi=500 ! number of character angles between 0 and pi/2; resolution in polar angle phi
    real(sel), allocatable :: theta(:), phi(:), rot(:,:,:), t(:,:)
    real(sel), allocatable :: m0(:,:), M(:,:,:), N(:,:,:), Cv(:,:,:,:,:)
    contains
      procedure :: update_theta => set_character_angles
  end type
  public :: volume_unitcell, set_character_angles, computerot, computestroh
  contains
    subroutine volume_unitcell(mat)
      class(metalprops), intent(inout) :: mat
      select case (trim(mat%sym))
        case ("iso","cubic")
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
      allocate(disl%theta(disl%ntheta))
      call linspace(0.d0,pi/2.d0,disl%ntheta,disl%theta)
    end subroutine set_character_angles
    subroutine computestroh(disl)
      class(disloc), intent(inout) :: disl
      allocate(disl%t(3,disl%ntheta),disl%m0(3,disl%ntheta),disl%phi(disl%nphi))
      allocate(disl%M(disl%nphi,3,disl%ntheta),disl%N(disl%nphi,3,disl%ntheta),disl%Cv(3,3,3,3,disl%ntheta))
      call strohgeometry(disl%b,disl%n0,disl%t,disl%m0,disl%M,disl%N,disl%Cv,disl%theta,disl%phi,disl%ntheta,disl%nphi)
    end subroutine computestroh
    subroutine computerot(disl)
      class(disloc), intent(inout) :: disl
      integer :: th
      allocate(disl%rot(3,3,disl%ntheta))
      do th=1,disl%ntheta
        call cross(disl%n0,disl%t,disl%rot(1,:,th))
        disl%rot(2,:,th) = disl%n0
        disl%rot(3,:,th) = disl%t(:,th)
      end do
    end subroutine computerot
end module dislocations
