! Some subroutines meant to be compiled for phononwind.py and dislocations.py via f2py
! run 'python -m numpy.f2py -c subroutines.f90 -m subroutines' to use
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - Mar. 3, 2021

subroutine version(versionnumber)
  integer, intent(out) :: versionnumber
  versionnumber=20210303
end subroutine version

subroutine ompinfo(nthreads)
!$   Use omp_lib
integer, intent(out) :: nthreads
nthreads = 0
!$   nthreads = omp_get_max_threads()
!~ !$   print*, 'OpenMP: using ',nthreads,' of ',omp_get_num_procs(),' processors'
!~ !$   print*, 'type "export OMP_NUM_THREADS=n" before running this prog. to change'
return
end subroutine ompinfo

subroutine parathesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1, &
                      lenp,lent,lenph1,lentph)
!$   Use omp_lib
implicit none
integer,parameter :: sel = selected_real_kind(6)

integer, intent(in) :: lentph, lenph1, lenp, lent
real(kind=sel), intent(in), dimension(lenph1) :: phi1, dphi1
real(kind=sel), intent(in), dimension(lentph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
real(kind=sel), intent(in), dimension(lentph,3) :: qv
real(kind=sel), intent(in), dimension(3,3) :: delta1, delta2
real(kind=sel), intent(in), dimension(3,3,3,3,3,3) :: A3
real(kind=sel), intent(out), dimension(lentph,3,3,3,3) :: output
!$ integer :: i,j,k, nthreads
output(:,:,:,:,:) = 0.0
!$ call ompinfo(nthreads)
!$ if (nthreads .ge. 2) then
!$OMP PARALLEL DO default(shared), private(i,j,k)
!$ do i=1,lenp
!$ j=(i-1)*lent+1
!$ k=i*lent
!$ call thesum(output(j:k,:,:,:,:),tcosphi(j:k),sqrtsinphi(j:k),tsinphi(j:k),sqrtcosphi(j:k),sqrtt(j:k), &
!$              qv(j:k,:),delta1,delta2,mag(j:k),A3,phi1,dphi1,lenph1,lent)
!$ enddo
!$OMP END PARALLEL DO
!$ else
call thesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1,lenph1,lentph)
!$ endif
end subroutine parathesum

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
real(kind=sel), intent(inout), dimension(lentph,3,3,3,3) :: output

!~ ! problem: openmp puts private variables in the stack, and if maxrec>3 arrays can get too large and cause a segfault
!~ !$OMP PARALLEL DO default(shared), private(i, j, k, l, ii, jj, kk, n, nn, m, p, &
!~ !$omp                      qt,qtshift,A3qt2,part1,part2) &
!~ !$omp& reduction(+:output)
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
!~ !$OMP END PARALLEL DO

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
  !$OMP PARALLEL DO IF(Ntheta > 20) DEFAULT(SHARED) PRIVATE(i,p,o,l,k)
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
  !$OMP END PARALLEL DO
  
  RETURN
END SUBROUTINE elbrak

SUBROUTINE elbrak1d(a,b,Cmat,Nphi,AB)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: Nphi
  REAL(KIND=sel), DIMENSION(Nphi,3), INTENT(IN)  :: a, b
  REAL(KIND=sel), DIMENSION(3,3,3,3), INTENT(IN)  :: Cmat
!-----------------------------------------------------------------------
  REAL(KIND=sel), DIMENSION(Nphi,3,3), INTENT(OUT) :: AB
!-----------------------------------------------------------------------
  integer l,o,k,p
  AB(:,:,:) = 0.d0
  do p=1,3
    do o=1,3
      do l=1,3
        do k=1,3
          AB(:,l,o) = AB(:,l,o) + a(:,k)*Cmat(k,l,o,p)*b(:,p)
        enddo
      enddo
    enddo
  enddo
  
  RETURN
END SUBROUTINE elbrak1d

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

SUBROUTINE dragcoeff_iso_phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi,distri)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: lenq1,lent,lenphi
  REAL(KIND=sel), INTENT(IN) :: T, c1qBZ, c2qBZ
  REAL(KIND=sel), INTENT(IN) :: q1(lenq1), q1h4(lenq1), prefac(lent,lenphi), OneMinBtqcosph1(lent,lenphi)
  REAL(KIND=sel), INTENT(OUT), DIMENSION(lent,lenphi,lenq1) :: distri
  INTEGER :: i
  REAL(KIND=sel) :: phonon1, phonon2(lent,lenphi), hbar=1.0545718d-34, kB=1.38064852d-23, beta
  
  beta = hbar/(kB*T)
  !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,phonon1,phonon2)
  do i=1,lenq1
    phonon1 = 1.d0/(exp(beta*c1qBZ*q1(i))-1.d0) 
    phonon2 = 1.d0/(exp(beta*c2qBZ*q1(i)*OneMinBtqcosph1(:,:))-1.d0) 
    distri(:,:,i) = prefac(:,:)*(phonon1 - phonon2(:,:))*q1h4(i)
  end do
  !$OMP END PARALLEL DO
  
  RETURN
END SUBROUTINE dragcoeff_iso_phonondistri

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
  
  REAL(KIND=sel), DIMENSION(Nphi) :: Wdensity
  INTEGER :: th, k, l, o, p
 
  Wtot = 0.d0
  !$OMP PARALLEL DO IF(Ntheta > 20) DEFAULT(SHARED) PRIVATE(th,p,o,l,k,Wdensity)
  do th=1,Ntheta
    Wdensity = 0.d0
    do p=1,3
      do o=1,3
        do l=1,3
          do k=1,3
            Wdensity(:) = Wdensity(:) + 0.5d0*uij(:,l,k,th)*uij(:,o,p,th)*(C2(k,l,o,p) + betaj*betaj*Cv(k,l,o,p,th))
          enddo
        enddo
      enddo
    enddo
    call trapz(Wdensity,phi,Nphi,Wtot(th))
  enddo
  !$OMP END PARALLEL DO
  
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
  real(kind=sel), dimension(Nphi,3,3) :: MM, NN, MN, NM, NNinv, Sphi, Bphi
  real(kind=sel), dimension(3,3,3,3) :: tmpC
  real(kind=sel), dimension(3,3) :: S, BB
  real(kind=sel), dimension(3) :: Sb, BBb
  real(kind=sel) :: pi2 = (4.d0*atan(1.d0))**2
  uij = 0.d0
  !$OMP PARALLEL DO default(shared), private(th,ph,j,k,i, &
  !$OMP                   MM,NN,MN,NM,NNinv,Sphi,Bphi,tmpC,S,BB,Sb,BBb)
  do th=1,Ntheta
  Sphi = 0.d0; Bphi = 0.d0
  tmpC(:,:,:,:) = C2(:,:,:,:) - beta*beta*Cv(:,:,:,:,th)
  call elbrak1d(M(:,:,th),M(:,:,th),tmpC,Nphi,MM)
  call elbrak1d(M(:,:,th),N(:,:,th),tmpC,Nphi,MN)
  call elbrak1d(N(:,:,th),M(:,:,th),tmpC,Nphi,NM)
  call elbrak1d(N(:,:,th),N(:,:,th),tmpC,Nphi,NN)
  do ph=1,Nphi
    call inv(NN(ph,:,:),NNinv(ph,:,:))
  enddo
  do j=1,3; do k=1,3; do i=1,3; do ph=1,Nphi
    Sphi(ph,i,j) = Sphi(ph,i,j) - NNinv(ph,i,k)*NM(ph,k,j)
  enddo; enddo; enddo; enddo
  do j=1,3; do i=1,3; do ph=1,Nphi
    Bphi(ph,i,j) = MM(ph,i,j)
    do k=1,3
      Bphi(ph,i,j) = Bphi(ph,i,j) + MN(ph,i,k)*Sphi(ph,k,j)
    enddo
  enddo; enddo; enddo
    do i=1,3; do j=1,3
      call trapz(Sphi(:,j,i),phi,Nphi,S(j,i))
      call trapz(Bphi(:,j,i),phi,Nphi,BB(j,i))
    enddo; enddo
    Sb(:) = (0.25d0/pi2)*MATMUL(S(:,:),b)
    BBb(:) = (0.25d0/pi2)*MATMUL(BB(:,:),b)
  do j=1,3; do i=1,3; do ph=1,Nphi
    uij(ph,i,j,th) = uij(ph,i,j,th) - Sb(i)*M(ph,j,th) &
                    + N(ph,j,th)*(DOT_PRODUCT(NNinv(ph,i,:),BBb(:)) - DOT_PRODUCT(Sphi(ph,i,:),Sb(:)))
  enddo; enddo; enddo; enddo
  !$OMP END PARALLEL DO
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
  !$OMP PARALLEL DO default(shared), private(p,tmask,t1,Btmp,NBtmp)
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
  !$OMP END PARALLEL DO
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
  !$OMP PARALLEL DO default(shared), private(p,tmask,qt,Btmp,NBtmp)
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
  !$OMP END PARALLEL DO
  call trapz(Bt,phi,Nphi,Bresult)
  
  RETURN
END SUBROUTINE integrateqtildephi
