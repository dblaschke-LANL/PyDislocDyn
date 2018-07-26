! Some subroutines meant to be compiled for phononwind.py via f2py
! run 'f2py -c phononwindsubroutines.f95 -m phononwindsubroutines' to use
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Los Alamos National Security, LLC. All rights reserved.
! Date: July 23, 2018 - July 24, 2018

subroutine thesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1,lenph1,lentph)

implicit none

integer :: i, j, k, l, ii, jj, kk, n, nn, m, p
integer, intent(in) :: lentph, lenph1
real, intent(in), dimension(lenph1) :: phi1, dphi1
real, dimension(lentph,3) :: qt, qtshift
real, dimension(lentph,3,3,3,3) :: A3qt2, part1, part2
real, intent(in), dimension(lentph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
real, intent(in), dimension(lentph,3) :: qv
real, intent(in), dimension(3,3) :: delta1, delta2
real, intent(in), dimension(3,3,3,3,3,3) :: A3
real, intent(out), dimension(lentph,3,3,3,3) :: output

output(:,:,:,:,:) = 0.0
qt(:,:) = 0.0
qtshift(:,:) = 0.0

do p = 1,lenph1

qt(:,1) = tcosphi - sqrtsinphi*cos(phi1(p))
qt(:,2) = tsinphi + sqrtcosphi*cos(phi1(p))
qt(:,3) = sqrtt*sin(phi1(p))
qtshift = qt - qv

A3qt2(:,:,:,:,:) = 0.0
do kk = 1,3
   do k = 1,3
      do j = 1,3
         do i = 1,3
            do jj = 1,3
               do ii = 1,3
                  A3qt2(:,i,j,k,kk) = A3qt2(:,i,j,k,kk) + qt(:,ii)*qtshift(:,jj)*A3(i,ii,j,jj,k,kk)
               end do
            end do
         end do
      end do
   end do
end do

part1(:,:,:,:,:) = 0.0
do kk = 1,3
   do k = 1,3
      do j = 1,3
         do l = 1,3
            do i = 1,3
               part1(:,l,j,k,kk) = part1(:,l,j,k,kk) + (delta1(i,l) - qt(:,l)*qt(:,i))*A3qt2(:,i,j,k,kk)
            end do
         end do
      end do
   end do
end do

part2(:,:,:,:,:) = 0.0
do nn = 1,3
   do n = 1,3
      do j = 1,3
         do m = 1,3
            do l = 1,3
               part2(:,l,j,n,nn) = part2(:,l,j,n,nn) + (delta2(j,m) - qtshift(:,j)*qtshift(:,m)/mag)*A3qt2(:,l,m,n,nn)
            end do
         end do
      end do
   end do
end do
part2(:,:,:,:,:) = part2(:,:,:,:,:)*dphi1(p)

do nn = 1,3
   do n = 1,3
      do kk = 1,3
         do k = 1,3
            do j = 1,3
               do l = 1,3
                  output(:,k,kk,n,nn) = output(:,k,kk,n,nn) + part1(:,l,j,k,kk)*part2(:,l,j,n,nn)
               end do
            end do
         end do
      end do
   end do
end do

end do

return
end subroutine


