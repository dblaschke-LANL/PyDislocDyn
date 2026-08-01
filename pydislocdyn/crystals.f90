! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Mar. 31, 2026 - Aug. 1, 2026
module dislocdyn_crystals
  use dislocdyn_parameters, only : sel, rzero, pi ! defined in subroutines.f90
  use dislocdyn_utilities, only : operator(.cross.) ! defined in subroutines.f90
  use dislocdyn_elasticconstants ! defined in elasticconstants.f90
  implicit none
  private
  !> The 'crystal' derived type is used to store material information for a crystal.
  !> It represents the fortran version of the metalprops class found in PyDislocDyn, implementing a subset of the latter.
  type, public :: crystal
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
      procedure :: init_crystal => init_crystal
      procedure :: Miller_to_Cart => Miller_to_Cart
      procedure :: computesound => computesound
      procedure :: anisotropy_index => anisotropy_index
  end type crystal
  public :: volume_unitcell, Miller_to_Cart, computesound, anisotropy_index
  !-------------------------
  contains
    !> computes the unit cell volume
    subroutine volume_unitcell(mat)
      class(crystal), intent(inout) :: mat
      select case (trim(mat%sym))
        case ("iso","cubic", "fcc", "bcc")
          mat%Vc = mat%lat_a(1)**3
        case ("hcp")
          mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)*sqrt(3.d0)*3.d0/2.d0
        case ("tetr","tetr2")
          mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)
        case ("orth", "ortho")
          mat%Vc = mat%lat_a(1)*mat%lat_a(2)*mat%lat_a(3)
        case ("trig")
          mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)*sqrt(3.d0)/2.d0
          if (abs(mat%lat_angles(1)-mat%lat_angles(2))+abs(mat%lat_angles(1)-mat%lat_angles(3)) < rzero) then
            mat%Vc = mat%lat_a(1)*mat%lat_a(1)*mat%lat_a(3)*sqrt(1.d0-3.d0*cos(mat%lat_angles(1))**2+2.d0*cos(mat%lat_angles(1))**3)
          end if
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
    !-------------------------
    !> initializes various properties of the crystal, such as angles, unit cell volume and tensors of elastic constants
    subroutine init_crystal(mat)
      class(crystal), intent(inout) :: mat
      integer :: i
      do i=1,3
        if (mat%lat_angles(i)>pi) then ! convert degrees to radians
          mat%lat_angles(i) = mat%lat_angles(i)*pi/180.d0
        end if
      end do
      select case (trim(mat%sym))
        case ("iso", "cubic", "fcc", "bcc", "tetr", "tetr2", "orth", "ortho")
          mat%lat_angles = 0.5d0*[pi,pi,pi]
        case ("hcp")
          mat%lat_angles = [0.5d0*pi,0.5d0*pi,2.d0*pi/3.d0]
        case ("trig")
          if (abs(mat%lat_angles(1))<rzero) then
            mat%lat_angles = [0.5d0*pi,0.5d0*pi,2.d0*pi/3.d0]
          end if
        case ("mono")
          mat%lat_angles(1) = 0.5d0*pi
          mat%lat_angles(3) = 0.5d0*pi
          if (abs(mat%lat_angles(2))<rzero) then
            print*,"Error: missing angle beta between lattice vactors a and c!"
          end if
        case ("tric")
          if (abs(mat%lat_angles(1)*mat%lat_angles(2)*mat%lat_angles(3))<rzero) then
            print*,"Error: missing angles between lattice vactors!"
            return
          end if
        case default
          print*,symkwerror
          return
      end select
      call volume_unitcell(mat)
      call elasticC2(mat%cij,mat%sym,mat%C2)
      if (allocated(mat%cijk)) then
        call elasticC3(mat%cijk,mat%sym,mat%C3) ! skip if not set
      end if
      if (mat%mu<1.d-9) then
        select case (trim(mat%sym))
          case ("iso")
            mat%lam = mat%cij(1)
            mat%mu = mat%cij(2)
          case ("cubic", "fcc", "bcc")
            call kroeneraverage(mat%C2,mat%lam,mat%mu)
          case default
            call hillaverage(mat%C2,mat%lam,mat%mu)
        end select
      end if
      call unvoigt(mat%C2/mat%mu,mat%C2norm)
    end subroutine init_crystal
    !-----------------------------------------
    !> Converts Miller indices to a Cartesian vector v, which is normalized if option normalize=true (default)
    !> another optional parameter, reziprocal (false by default) can be set to use the reziprocal crystal basis for conversion
    !> (needed if Millerv describes the normal to a plane for example)
    subroutine Miller_to_Cart(mat,millerv,normalize,reziprocal,v)
      class(crystal), intent(in) :: mat ! needed for the lattice vectors and angles
      real(sel), intent(in) :: millerv(:)
      logical, optional :: normalize, reziprocal
      real(sel), intent(out) :: v(3)
      logical :: norm, rezi
      real(sel) :: a, b, c, d, T(3,3), R(3,3), RV
      integer :: n, i
      norm = .true.
      rezi = .false.
      if (present(normalize)) norm = normalize
      if (present(reziprocal)) rezi = reziprocal
      n = size(millerv)
      if (n/=3 .and. trim(mat%sym)/='hcp') then
        error stop "expected 3 Miller indices"
      else if (trim(mat%sym)=='hcp' .and. n/=4) then
        error stop "crystal sym is hcp; expected 4 Miller indices"
      end if
      a = mat%lat_a(1); b = mat%lat_a(2); c = mat%lat_a(3)
      ! choose sensible default lengths for missing vectors:
      if (a<rzero) a=1.d0
      if (b<rzero) b=a
      if (c<rzero) c=a
      d = c*(cos(mat%lat_angles(1))-cos(mat%lat_angles(3))*cos(mat%lat_angles(2)))/sin(mat%lat_angles(3))
      T(1,:) = [a,b*cos(mat%lat_angles(3)),c*cos(mat%lat_angles(2))]
      T(2,:) = [0.d0,b*sin(mat%lat_angles(3)),d]
      T(3,:) = [0.d0,0.d0,sqrt((c*sin(mat%lat_angles(2)))**2-d**2)]
      if (any(["iso","fcc","bcc"]==trim(mat%sym)) .or. trim(mat%sym)=="cubic") then
        v=Millerv*a
      else if (rezi) then
        RV = dot_product(T(:,1).cross.T(:,2),T(:,3))
        R(:,1) = (T(:,2).cross.T(:,3))/RV
        R(:,2) = (T(:,3).cross.T(:,1))/RV
        R(:,3) = (T(:,1).cross.T(:,2))/RV
        if (n==4 .and. abs(sum(millerv(:3)))<rzero) then
          v = matmul(R,[millerv(1)+millerv(3),millerv(2)-millerv(3),millerv(4)])
        else
          v = matmul(R,millerv)
        end if
      else
        if (n==4) then
          v = matmul(T,[millerv(1)-millerv(3),millerv(2)-millerv(3),millerv(4)])
        else
          v = matmul(T,millerv)
        end if
      end if
      if (norm) then
        v = v / sqrt(dot_product(v,v))
      end if
      do i=1,3 ! remove noise
        if (abs(v(i))<1.d-15) v(i)=0.d0
      end do
    end subroutine Miller_to_Cart
    !------------------------------
    !> Computes the sound speeds of the crystal propagating in the direction of unit vector v (Cartesian coordinates).
    !> Use function Miller_to_Cart() to convert Miller indices to Cartesian coordinates prior to calling this routine.
    !> The present numerical method is derived from Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5 for the special 
    !> case of z=v and psi=0.
    subroutine computesound(mat,v,sound)
      use dislocdyn_parameters, only : sel
      use dislocdyn_subroutines, only : vlim_of_phi
      class(crystal), intent(in) :: mat
      real(sel), intent(in) :: v(3)
      real(sel), intent(out) :: sound(3)
      real(sel) :: vnorm(3), zero(3), C2(3,3,3,3), norm
      integer :: i
      vnorm = v / sqrt(dot_product(v,v))
      zero = [0.d0,0.d0,0.d0]
      norm = (mat%C2(4,4)/mat%rho)
      call unvoigt(mat%C2/mat%C2(4,4),C2)
      do i=1,3
        sound(i) = vlim_of_phi(0.d0,i,C2,norm,vnorm,zero)
      end do
    end subroutine computesound
    !------------------------------
    !> Computes a number quantifying the anisotropy of a crystal following the
    !> recommendation of Kube 2016. In particular, we compute a measure of the
    !> difference between Voigt and Reuss averages of shear and bulk modulus, 
    !> known also as the universal log-Euclidean anisotropy index:
    !> A_L = sqrt([ln(B_V/B_R)]^2 + 5*[ln(G_V/G_R)]^2), see AIP Advances 6, 095209 (2016).
    subroutine anisotropy_index(mat,anisidx)
      use dislocdyn_parameters, only : sel
      class(crystal), intent(in) :: mat
      real(sel), intent(out) :: anisidx
      real(sel) :: bulkV, bulkR, muV, muR, lambda
      call voigtaverage(mat%C2,lambda,muV)
      bulkV = lambda + 2.d0*muV/3.d0
      call reussaverage(mat%C2,lambda,muR)
      bulkR = lambda + 2.d0*muR/3.d0
      anisidx = sqrt((log(bulkV/bulkR))**2 + 5.d0*(log(muV/muR))**2)
    end subroutine anisotropy_index

end module dislocdyn_crystals
