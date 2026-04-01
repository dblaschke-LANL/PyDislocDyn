! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - Mar. 31, 2026
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
    real(sel) :: b(3), n(3) ! Burgers vector and slip plane normal
    real(sel) :: burgers ! Burgers vector length
    integer :: ntheta, nphi ! number of character angles between 0 and pi/2; resolution in polar angle phi
    real(sel), allocatable :: theta(:), phi(:)
    contains
      procedure :: update_theta => set_character_angles
  end type
  public :: volume_unitcell, set_character_angles
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
end module dislocations
