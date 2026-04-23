! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: Apr. 10, 2026 - Apr. 22, 2026
module readinputfiles
  use parameters, only : sel ! defined in subroutines.f90
  use dislocations ! defined in dislocations.f90
  implicit none
  contains
    subroutine read_materialfile(filename,disl)
      ! populates an instance of the dislocation derived type using data read from filename
      character(*), intent(in) :: filename
      type(disloc), intent(out) :: disl
      ! local variables
      integer :: ios, j
      character(32) :: key, metal, sym
      character(256) :: line, values, dummy
      real(sel) :: c11, c12, c13, c33, c44, c66
      real(sel) :: c111, c112, c113, c123, c133, c144, c155, c166, c222, c333, c344, c366, c456
      
      open(unit=42, file=trim(filename), action="read", iostat=ios, status='old')
      if (ios/=0) then
        close(unit=42)
        print*, "Error: file not found, tried ", filename
        stop
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
        end if
        if (key=='name') then
          read(values,*)metal
          disl%metal = trim(metal)
        end if
        if (key=='sym') then
          read(values,*)sym
          disl%sym = trim(sym)
        end if
        !! TODO: remove commas in line before reading arrays and find a way to read fractions!
!~         if (key=='Millerb') read(line,*) key,dummy,(disl%b(j), j=1,3) ! cubic only, TODO: generalize
!~         if (key=='Millern0') read(line,*) key,dummy,(disl%n0(j), j=1,3)
        if (key=='T') read(values,*)disl%Temp
        if (key=='a') read(values,*)disl%lat_a(1)
        if (key=='lcb') read(values,*)disl%lat_a(2) ! b already used for Burgers vector
        if (key=='c') read(values,*)disl%lat_a(3)
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
        ! TODO: support lower symmetries
      
      end do ! read file
      close(unit=42)
      
      ! start postprocessing / initializing depending on symmetry
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
        case default
          print*,"Error: keyword sym must be one of 'iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric'."
          return
      end select
      
    end subroutine read_materialfile
end module readinputfiles
