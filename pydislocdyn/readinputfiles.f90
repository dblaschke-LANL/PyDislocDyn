! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - June 15, 2026
module readinputfiles
  use parameters, only : sel, rzero ! defined in subroutines.f90
  use elastic_constants, only : symkwerror, number_of_elasticC
  use dislocations ! defined in dislocations.f90
  implicit none
  private
  type :: string_t
    character(:), allocatable :: str
  end type string_t
  !>data structure to store the input deck information
  type, public :: inputdeck
    type(string_t), allocatable :: sim_type(:) ! choose between 'drag' and 'vlimit'
    character(:), allocatable :: logfile
    real(sel) :: betamin, betamax, Millernorm
    real(sel), allocatable :: b(:), n0(:), beta(:)
    integer :: nbeta, ntheta, nphi
    logical :: echoinput
  end type inputdeck
  
  !-------------------------
  public :: scan_inputdeck, read_inputdeck, read_materialfile
  contains
    !>counts the number of lines starting with 'sim_type' in an input deck file
    subroutine scan_inputdeck(filename,nsims)
      character(*), intent(in) :: filename
      integer, intent(out) :: nsims
      ! local variables
      integer :: ios
      character(32) :: key
      character(256) :: line, values
      nsims = 0
      
      open(unit=42, file=trim(filename), action="read", iostat=ios, status='old')
      if (ios/=0) then
        close(unit=42)
        error stop "File not found: " // filename
      end if
      do
        read(42,'(a)',iostat=ios) line
        if (ios/=0) exit
        if ((line /= " ") .and. (line(1:1) /= "#")) then
          read(line,*) key,values
        else
          ! skip empty lines
          key = trim(line)
        end if
        if (key=='sim_type') then
          nsims = nsims + 1
        end if
      end do ! read file
      close(unit=42)
!~       print*,"found",nsims,"sims"
    end subroutine scan_inputdeck
    !>reads an input deck file and stores its info in 'sim_plan' of derived type 'inputdeck' 
    subroutine read_inputdeck(filename,sim_plan,sym,nsims)
      use utilities, only: linspace
      character(*), intent(in) :: filename
      integer, intent(in) :: nsims
      type(inputdeck), intent(out) :: sim_plan
      character(*), optional :: sym
      ! local variables
      integer :: ios, j, n, p
      character(32) :: key
      character(256) :: line, values, dummy
      p = 1
      ! default values:
      allocate(sim_plan%sim_type(nsims))
      sim_plan%sim_type(1)%str = ''
      sim_plan%nbeta = 1
      sim_plan%ntheta = 2
      sim_plan%nphi = 500
      sim_plan%betamin = 0.01d0
      sim_plan%betamax = 0.99d0
      sim_plan%Millernorm = 1.d0 ! will divide Millerb by this number so as to avoid having things like 0.333333 in the inputdeck
      sim_plan%echoinput = .true.
      sim_plan%logfile = 'dislocdyn.log'
      n = 3 ! expect 3 Miller indices for each b and n0 in the file
      if (present(sym) .and. trim(sym)=='hcp') n = 4 ! expect 4 Miller indices for each b and n0 in the file
      allocate(sim_plan%b(n),sim_plan%n0(n))
      sim_plan%b = 0.d0
      sim_plan%n0 = 0.d0
      
      open(unit=42, file=trim(filename), action="read", iostat=ios, status='old')
      if (ios/=0) then
        close(unit=42)
        error stop "File not found: " // filename
      end if
      do
        read(42,'(a)',iostat=ios) line
        if (ios/=0) exit
        if ((line /= " ") .and. (line(1:1) /= "#")) then
          read(line,*) key,dummy,values
          if (trim(dummy) /= "=") then
            key = " "
            print*,"skipping ", line, " (unknown format)"
          end if
        else
          ! skip empty lines
          key = trim(line)
        end if
        if (key=='sim_type') then
          if (p>nsims) error stop "too many sim_types in file"
          sim_plan%sim_type(p)%str = trim(values)
          p = p+1
        end if
        if (key=='logfile') sim_plan%logfile = trim(values)
        if (key=='echoinput') read(line,*) key,dummy,sim_plan%echoinput
        if (key=='b' .or. key=='Millerb') read(line,*) key,dummy,(sim_plan%b(j), j=1,n)
        if (key=='n0' .or. key=='Millern0') read(line,*) key,dummy,(sim_plan%n0(j), j=1,n)
        if (key=='betamin') read(line,*) key,dummy,sim_plan%betamin
        if (key=='betamax') read(line,*) key,dummy,sim_plan%betamax
        if (key=='Millernorm') read(line,*) key,dummy,sim_plan%Millernorm
        if (key=='nbeta') then
          read(line,*) key,dummy,sim_plan%nbeta
          allocate(sim_plan%beta(sim_plan%nbeta))
          sim_plan%beta = 0.d0
        end if
        if (key=='beta') read(line,*) key,dummy,(sim_plan%beta(j), j=1,sim_plan%nbeta)
        if (key=='ntheta') read(line,*) key,dummy,sim_plan%ntheta
        if (key=='nphi') read(line,*) key,dummy,sim_plan%nphi
      
      end do ! read file
      close(unit=42)

      if (.not.allocated(sim_plan%beta)) then
        allocate(sim_plan%beta(sim_plan%nbeta))
        call linspace(sim_plan%betamin,sim_plan%betamax,sim_plan%nbeta,sim_plan%beta)
      else if (all(abs(sim_plan%beta)<rzero)) then
        if (allocated(sim_plan%beta)) deallocate(sim_plan%beta); allocate(sim_plan%beta(sim_plan%nbeta))
        call linspace(sim_plan%betamin,sim_plan%betamax,sim_plan%nbeta,sim_plan%beta)
      end if
    end subroutine read_inputdeck

    !-------------------------
    !>reads a material file and stores the information in 'disl' of derived type 'disloc'
    subroutine read_materialfile(filename,disl)
      ! populates an instance of the dislocation derived type using data read from filename
      character(*), intent(in) :: filename
      type(disloc), intent(out) :: disl
      ! local variables
      integer :: ios, lencij, lencijk, j
      character(32) :: key, metal, sym
      character(256) :: line, values, dummy
      real(sel) :: c11, c12, c13, c33, c44, c66
      real(sel) :: c111, c112, c113, c123, c133, c144, c155, c166, c222, c333, c344, c366, c456
      
      open(unit=42, file=trim(filename), action="read", iostat=ios, status='old')
      if (ios/=0) then
        close(unit=42)
        error stop "File not found: " // filename
      end if
      do
        read(42,'(a)',iostat=ios) line
        if (ios/=0) exit
        key="" ! reset the key
        if ((line /= " ") .and. (line /= "") .and. (line(1:1) /= "#")) then
          read(line,*) key,dummy,values
          if (trim(dummy) /= "=") then
            key = " "
            print*,"skipping ", line, " (unknown format)"
          end if
        end if
        if (key=='name') then
          read(values,*)metal
          disl%metal = trim(metal)
        end if
        if (key=='sym') then
          read(values,*)sym
          disl%sym = trim(sym)
          call number_of_elasticC(disl%sym,lencij,lencijk)
          if (allocated(disl%cij)) deallocate(disl%cij)
          if (allocated(disl%cijk)) deallocate(disl%cijk)
          allocate(disl%cij(lencij),disl%cijk(lencijk))
        end if
        if (key=='T') read(values,*)disl%Temp
        if (key=='a') read(values,*)disl%lat_a(1)
        if (key=='lcb') read(values,*)disl%lat_a(2) ! b already used for Burgers vector
        if (key=='c') read(values,*)disl%lat_a(3)
        if (key=='lat_a') read(line,*) key,dummy,(disl%lat_a(j), j=1,3) ! allow reading all lattice vectors from one line
        if (key=='lat_angles') read(line,*) key,dummy,(disl%lat_angles(j), j=1,3)
        if (key=='alpha') read(values,*)disl%lat_angles(1)
        if (key=='beta') read(values,*)disl%lat_angles(2)
        if (key=='gamma') read(values,*)disl%lat_angles(3)
        if (key=='rho') read(values,*)disl%rho
        ! read SOEC
        if (key=='c11') read(values,*)c11
        if (key=='c12') read(values,*)c12
        if (key=='c13') read(values,*)c13
        if (key=='c33') read(values,*)c33
        if (key=='c44') read(values,*)c44
        if (key=='c66') read(values,*)c66
        if (key=='lam') read(values,*)disl%lam
        if (key=='mu') read(values,*)disl%mu
        if (key=='cij') read(line,*) key,dummy,(disl%cij(j), j=1,lencij)
        ! read TOEC
        if (key=='c111') read(values,*)c111
        if (key=='c112') read(values,*)c112
        if (key=='c113') read(values,*)c113
        if (key=='c123') read(values,*)c123
        if (key=='c133') read(values,*)c133
        if (key=='c144') read(values,*)c144
        if (key=='c155') read(values,*)c155
        if (key=='c166') read(values,*)c166
        if (key=='c222') read(values,*)c222
        if (key=='c333') read(values,*)c333
        if (key=='c344') read(values,*)c344
        if (key=='c366') read(values,*)c366
        if (key=='c456') read(values,*)c456
        if (key=='cijk') read(line,*) key,dummy,(disl%cijk(j), j=1,lencijk)
      
      end do ! read file
      close(unit=42)
      
      ! start postprocessing / initializing depending on symmetry
      if (abs(disl%cij(1))<rzero) then
        select case (trim(disl%sym))
          case ("iso")
            disl%cij = [c12,c44]
            disl%cijk = [c123,c144,c456]
          case ("cubic", "fcc", "bcc")
            disl%cij = [c11,c12,c44]
            disl%cijk = [c111,c112,c123,c144,c166,c456]
          case ("hcp")
            disl%cij = [c11,c12,c13,c33,c44]
            disl%cijk = [c111,c112,c113,c123,c133,c144,c155,c222,c333,c344]
          case ("tetr")
            disl%cij = [c11,c12,c13,c33,c44,c66]
            disl%cijk = [c111,c112,c113,c123,c133,c144,c155,c166,c333,c344,c366,c456]
        end select
      end if
      
    end subroutine read_materialfile
end module readinputfiles
