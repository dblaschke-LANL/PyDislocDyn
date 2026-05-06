! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - May 6, 2026
module readinputfiles
  use parameters, only : sel, rzero ! defined in subroutines.f90
  use elastic_constants, only : symkwerror, number_of_elasticC
  use dislocations ! defined in dislocations.f90
  implicit none
  !>data structure to store the input deck information
  type, public :: inputdeck
    character(:), allocatable :: sim_type, logfile ! choose between 'drag' (others not yet implemented)
    real(sel) :: b(3), n0(3), betamin, betamax ! slip plane in Cartesian coordinates (TODO: implement Miller indices)
    real(sel), allocatable :: beta(:)
    integer :: nbeta, ntheta
    logical :: echoinput
  end type inputdeck
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  contains
    !>reads an input deck file and stores its info in 'sim_plan' of derived type 'inputdeck' 
    subroutine read_inputdeck(filename,sim_plan)
      use utilities, only: linspace
      ! reads an inputdeck from filename, and stores all values in 'sim_plan' of derived type 'inputdeck'
      character(*), intent(in) :: filename
      type(inputdeck), intent(out) :: sim_plan
      ! local variables
      integer :: ios, j
      character(32) :: key
      character(256) :: line, values, dummy
      ! default values:
      sim_plan%sim_type = '' !'drag'
      sim_plan%nbeta = 1
      sim_plan%ntheta = 2
      sim_plan%betamin = 0.01d0
      sim_plan%betamax = 0.99d0
      sim_plan%b = 0.d0
      sim_plan%n0 = 0.d0
      sim_plan%echoinput = .true.
      sim_plan%logfile = 'dislocdyn.log'
      
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
        if (key=='sim_type') sim_plan%sim_type = trim(values)
        if (key=='logfile') sim_plan%logfile = trim(values)
        if (key=='echoinput') read(line,*) key,dummy,sim_plan%echoinput
        if (key=='b') read(line,*) key,dummy,sim_plan%b(1),sim_plan%b(2),sim_plan%b(3)
        if (key=='n0') read(line,*) key,dummy,sim_plan%n0(1),sim_plan%n0(2),sim_plan%n0(3)
        if (key=='betamin') read(line,*) key,dummy,sim_plan%betamin
        if (key=='betamax') read(line,*) key,dummy,sim_plan%betamax
        if (key=='nbeta') then
          read(line,*) key,dummy,sim_plan%nbeta
          allocate(sim_plan%beta(sim_plan%nbeta))
          sim_plan%beta = 0.d0
        end if
!~         if (key=='beta') read(line,*) key,dummy,(sim_plan%beta(j), j=1,sim_plan%nbeta)
        if (key=='ntheta') read(line,*) key,dummy,sim_plan%ntheta
      
      end do ! read file
      close(unit=42)

      if (.not.allocated(sim_plan%beta)) then
        allocate(sim_plan%beta(sim_plan%nbeta))
        call linspace(sim_plan%betamin,sim_plan%betamax,sim_plan%nbeta,sim_plan%beta)
      else if (all(abs(sim_plan%beta)<rzero)) then
        if (allocated(sim_plan%beta)) deallocate(sim_plan%beta); allocate(sim_plan%beta(sim_plan%nbeta))
        call linspace(sim_plan%betamin,sim_plan%betamax,sim_plan%nbeta,sim_plan%beta)
      end if
    end subroutine

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        !! TODO: remove commas in line before reading arrays and find a way to read fractions!
!~         if (key=='Millerb') read(line,*) key,dummy,(disl%b(j), j=1,3) ! cubic only, TODO: generalize
!~         if (key=='Millern0') read(line,*) key,dummy,(disl%n0(j), j=1,3)
        if (key=='T') read(values,*)disl%Temp
        if (key=='a') read(values,*)disl%lat_a(1)
        if (key=='lcb') read(values,*)disl%lat_a(2) ! b already used for Burgers vector
        if (key=='c') read(values,*)disl%lat_a(3)
!~         if (key=='lat_a') read(line,*) key,dummy,(disl%lat_a(j), j=1,3) ! allow reading all lattice vectors from one line
!~         if (key=='lat_angles') read(line,*) key,dummy,(disl%lat_angles(j), j=1,3)
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
