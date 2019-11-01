! Some subroutines meant to be compiled for phononwind.py and dislocations.py via f2py
! run 'f2py -c subroutines.f90 -m subroutines' to use
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - Oct. 31, 2019

subroutine thesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1,lenph1,lentph)

implicit none
integer,parameter :: sel = selected_real_kind(6)
integer :: i, j, k, l, ii, jj, kk, n, nn, m, p
integer, intent(in) :: lentph, lenph1
real(kind=sel), intent(in), dimension(lenph1) :: phi1, dphi1
real(kind=sel), dimension(lentph,3) :: qt, qtshift
real(kind=sel), dimension(lentph,3,3,3,3) :: A3qt2, part1, part2
real(kind=sel), intent(in), dimension(lentph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
real(kind=sel), intent(in), dimension(lentph,3) :: qv
real(kind=sel), intent(in), dimension(3,3) :: delta1, delta2
real(kind=sel), intent(in), dimension(3,3,3,3,3,3) :: A3
real(kind=sel), intent(out), dimension(lentph,3,3,3,3) :: output

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
end subroutine thesum

!!**********************************************************************

subroutine dragintegrand(output,prefactor,dij,poly,lent,lenph)

implicit none

integer,parameter :: sel = selected_real_kind(6)
integer :: i, k, kk, n, nn
integer, intent(in) :: lent, lenph
real(kind=sel), intent(in), dimension(lenph,lent) :: prefactor
real(kind=sel), intent(in), dimension(lenph,3,3) :: dij
real(kind=sel), intent(in), dimension(lenph,3,3,3,3,lent) :: poly
real(kind=sel), intent(out), dimension(lenph,lent) :: output

output(:,:) = 0.0

do i = 1,lent
   do nn=1,3
      do n=1,3
         do kk=1,3
            do k=1,3
               output(:,i) = output(:,i) - dij(:,k,kk)*dij(:,n,nn)*poly(:,k,kk,n,nn,i)
            end do
         end do
      end do
   end do
end do

output(:,:) = prefactor(:,:)*output(:,:)

return
end subroutine dragintegrand

!!**********************************************************************

SUBROUTINE elbrak(a,b,Cmat,Ntheta,Nphi,AB)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: Ntheta,Nphi
  REAL(KIND=sel), DIMENSION(Nphi,3,Ntheta), INTENT(IN)  :: a, b
  REAL(KIND=sel), DIMENSION(3,3,3,3,Ntheta), INTENT(IN)  :: Cmat
!-----------------------------------------------------------------------
  REAL(KIND=sel), DIMENSION(Nphi,3,3,Ntheta), INTENT(OUT) :: AB
!-----------------------------------------------------------------------
  integer l,o,k,p,i
  AB(:,:,:,:) = 0.d0
  do i=1, Ntheta
    do p=1,3
      do o=1,3
        do l=1,3
          do k=1,3
            AB(:,l,o,i) = AB(:,l,o,i) + a(:,k,i)*Cmat(k,l,o,p,i)*b(:,p,i)
          enddo
        enddo
      enddo
    enddo
  enddo
  
  RETURN
END SUBROUTINE elbrak

!!**********************************************************************

SUBROUTINE trapz(f,x,n,intf)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: n
  REAL(KIND=sel), INTENT(IN), DIMENSION(n)  :: f, x
  REAL(KIND=sel), INTENT(OUT) :: intf
 
  intf = sum(0.5d0*(f(2:n)+f(1:n-1))*(x(2:n)-x(1:n-1)))
  
  RETURN
END SUBROUTINE trapz

!!**********************************************************************

SUBROUTINE computeEtot(uij, betaj, C2, Cv, phi, Ntheta, Nphi, Wtot)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: Ntheta, Nphi
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi,3,3,Ntheta)  :: uij
  REAL(KIND=sel), INTENT(IN) :: betaj
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3)  :: C2
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3,Ntheta)  :: Cv
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi)  :: phi
  REAL(KIND=sel), INTENT(OUT), DIMENSION(Ntheta) :: Wtot
  
  REAL(KIND=sel), DIMENSION(Nphi,Ntheta) :: Wdensity
  INTEGER :: th, k, l, o, p
 
  Wdensity = 0.d0
  Wtot = 0.d0
  do th=1,Ntheta
    do p=1,3
      do o=1,3
        do l=1,3
          do k=1,3
            Wdensity(:,th) = Wdensity(:,th) + 0.5d0*uij(:,l,k,th)*uij(:,o,p,th)*(C2(k,l,o,p) + betaj*betaj*Cv(k,l,o,p,th))
          enddo
        enddo
      enddo
    enddo
    call trapz(Wdensity(:,th),phi,Nphi,Wtot(th))
  enddo
  
  RETURN
END SUBROUTINE computeEtot

!!**********************************************************************

SUBROUTINE inv(A,invA)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3)  :: A
  REAL(KIND=sel), INTENT(OUT), DIMENSION(3,3)  :: invA
  REAL(KIND=sel) :: det !, eps(3,3,3)
!~   INTEGER :: i, j, k, l, m, n
  
!~   eps = 0.d0
!~   eps(1,2,3) = 1.d0; eps(3,1,2) = 1.d0; eps(2,3,1) = 1.d0
!~   eps(3,2,1) = -1.d0; eps(1,3,2) = -1.d0; eps(2,1,3) = -1.d0
  
  ! note: hard coding this loop speeds up things by factor 6/27
!~   do i=1,3; do j=1,3; do k=1,3
!~     det = det + eps(i,j,k)*A(1,i)*A(2,j)*A(3,k)
!~   enddo; enddo; enddo
  det = A(1,1)*A(2,2)*A(3,3) + A(1,3)*A(2,1)*A(3,2) + A(1,2)*A(2,3)*A(3,1) &
        - A(1,3)*A(2,2)*A(3,1) - A(1,1)*A(2,3)*A(3,2) - A(1,2)*A(2,1)*A(3,3)
  
  ! note: hard coding this loop speeds up things by factor 18/27**2
!~   invA = 0.d0
!~   do i=1,3; do j=1,3; do k=1,3
!~     do l=1,3; do m=1,3; do n=1,3
!~       invA(i,j) = invA(i,j) + 0.5d0*eps(j,k,l)*eps(i,m,n)*A(k,m)*A(l,n)
!~     enddo; enddo; enddo
!~   enddo; enddo; enddo
  invA(1,1) = A(2,2)*A(3,3) - A(2,3)*A(3,2)
  invA(2,2) = A(1,1)*A(3,3) - A(1,3)*A(3,1)
  invA(3,3) = A(1,1)*A(2,2) - A(1,2)*A(2,1)
  invA(1,2) = A(3,2)*A(1,3) - A(3,3)*A(1,2)
  invA(2,1) = A(2,3)*A(3,1) - A(3,3)*A(2,1)
  invA(2,3) = A(1,3)*A(2,1) - A(1,1)*A(2,3)
  invA(3,2) = A(3,1)*A(1,2) - A(1,1)*A(3,2)
  invA(3,1) = A(2,1)*A(3,2) - A(2,2)*A(3,1)
  invA(1,3) = A(1,2)*A(2,3) - A(2,2)*A(1,3)
  
  invA = invA/det
  
  RETURN
END SUBROUTINE inv

!!**********************************************************************

SUBROUTINE computeuij(beta, C2, Cv, b, M, N, phi, Ntheta, Nphi, uij)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN) :: beta
  INTEGER, INTENT(IN) :: Ntheta, Nphi
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3)  :: C2
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3,Ntheta)  :: Cv
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi)  :: phi
  REAL(KIND=sel), INTENT(IN), DIMENSION(3)  :: b
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi,3,Ntheta)  :: M, N
  REAL(KIND=sel), INTENT(OUT), DIMENSION(Nphi,3,3,Ntheta)  :: uij
  integer :: i, j, k, th, ph
  real(kind=sel), dimension(Nphi,3,3,Ntheta) :: MM, NN, MN, NM, NNinv, Sphi, Bphi
  real(kind=sel), dimension(3,3,3,3,Ntheta) :: tmpC
  real(kind=sel), dimension(3,3,Ntheta) :: S, BB
  real(kind=sel), dimension(3,Ntheta) :: Sb, BBb
  real(kind=sel) :: pi2 = (4.d0*atan(1.d0))**2
!~   pi2 = (0.5d0*phi(Nphi))**2
  do th=1,Ntheta
    tmpC(:,:,:,:,th) = C2(:,:,:,:) - beta*beta*Cv(:,:,:,:,th)
  enddo
  call elbrak(M,M,tmpC,Ntheta,Nphi,MM)
  call elbrak(M,N,tmpC,Ntheta,Nphi,MN)
  call elbrak(N,M,tmpC,Ntheta,Nphi,NM)
  call elbrak(N,N,tmpC,Ntheta,Nphi,NN)
  Sphi = 0.d0; Bphi = 0.d0
  do th=1,Ntheta; do ph=1,Nphi
    call inv(NN(ph,:,:,th),NNinv(ph,:,:,th))
  enddo; enddo
  do th=1,Ntheta; do j=1,3; do k=1,3; do i=1,3; do ph=1,Nphi
    Sphi(ph,i,j,th) = Sphi(ph,i,j,th) - NNinv(ph,i,k,th)*NM(ph,k,j,th)
  enddo; enddo; enddo; enddo; enddo
  do th=1,Ntheta; do j=1,3; do i=1,3; do ph=1,Nphi
    Bphi(ph,i,j,th) = MM(ph,i,j,th)
    do k=1,3
      Bphi(ph,i,j,th) = Bphi(ph,i,j,th) + MN(ph,i,k,th)*Sphi(ph,k,j,th)
    enddo
  enddo; enddo; enddo; enddo
  do th=1,Ntheta
    do i=1,3; do j=1,3
      call trapz(Sphi(:,j,i,th),phi,Nphi,S(j,i,th))
      call trapz(Bphi(:,j,i,th),phi,Nphi,BB(j,i,th))
    enddo; enddo
    Sb(:,th) = (0.25d0/pi2)*MATMUL(S(:,:,th),b)
    BBb(:,th) = (0.25d0/pi2)*MATMUL(BB(:,:,th),b)
  enddo
  uij = 0.d0
  do th=1,Ntheta; do j=1,3; do i=1,3; do ph=1,Nphi
    uij(ph,i,j,th) = uij(ph,i,j,th) - Sb(i,th)*M(ph,j,th) &
                    + N(ph,j,th)*(DOT_PRODUCT(NNinv(ph,i,:,th),BBb(:,th)) - DOT_PRODUCT(Sphi(ph,i,:,th),Sb(:,th)))
  enddo; enddo; enddo; enddo
  RETURN
END SUBROUTINE computeuij

!!**********************************************************************

SUBROUTINE integratetphi(B,beta,t,phi,updatet,kthchk,Nphi,Nt,Bresult)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: Nphi, Nt, kthchk
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nt,Nphi)  :: B
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nt)  :: t
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi)  :: phi
  REAL(KIND=sel), INTENT(IN)  :: beta
  LOGICAL, INTENT(IN)  :: updatet
  REAL(KIND=sel), INTENT(OUT) :: Bresult
  integer :: p, NBtmp
  real(kind=sel) :: limit(Nphi), Bt(Nphi)
  real(kind=sel), dimension(:), allocatable :: Btmp, t1
  logical ::  tmask(Nt)
  
  limit = beta*abs(cos(phi))
  Bt = 0.d0
  do p=1,Nphi
    tmask = (t>limit(p))
    t1 = pack(t,tmask)
    Btmp = pack(B(:,p),tmask)
    NBtmp = size(Btmp)
    Bt(p) = 0.d0
    if (NBtmp.gt.1) then
      if ((updatet.eqv..True.).or.(kthchk.eq.0)) then
        Btmp(1) = 2.d0*Btmp(1)
      endif
      if (updatet.eqv..True.) then
        Btmp(ubound(Btmp)) = 2.d0*Btmp(ubound(Btmp))
      endif
      call trapz(Btmp(:),t1(:),NBtmp,Bt(p))
    endif
  enddo
  call trapz(Bt,phi,Nphi,Bresult)
  
  RETURN
END SUBROUTINE integratetphi

!!**********************************************************************

SUBROUTINE integrateqtildephi(B,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,Nphi,Nt,Bresult)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: Nphi, Nt, kthchk, Nchunks
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nt,Nphi)  :: B, t
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nt)  :: qtilde
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi)  :: phi
  REAL(KIND=sel), INTENT(IN)  :: beta1
  LOGICAL, INTENT(IN)  :: updatet
  REAL(KIND=sel), INTENT(OUT) :: Bresult
  integer :: p, NBtmp
  real(kind=sel) :: qtlimit(Nphi), Bt(Nphi)
  real(kind=sel), dimension(:), allocatable :: Btmp, qt
  logical ::  tmask(Nt)
  
  qtlimit = 1/(beta1*abs(cos(phi)))
  Bt = 0.d0
  do p=1,Nphi
    tmask = (abs(t(:,p))<1).and.(qtilde(:)<qtlimit(p))
    qt = pack(qtilde,tmask)
    Btmp = pack(B(:,p),tmask)
    NBtmp = size(Btmp)
    Bt(p) = 0.d0
    if (NBtmp.gt.1) then
      if ((updatet.eqv..True.).or.(kthchk.eq.0)) then
        Btmp(1) = 2.d0*Btmp(1)
      endif
      if ((updatet.eqv..True.).or.(kthchk.eq.(Nchunks-1))) then
        Btmp(ubound(Btmp)) = 2.d0*Btmp(ubound(Btmp))
      endif
      call trapz(Btmp(:),qt(:),NBtmp,Bt(p))
    endif
  enddo
  call trapz(Bt,phi,Nphi,Bresult)
  
  RETURN
END SUBROUTINE integrateqtildephi
