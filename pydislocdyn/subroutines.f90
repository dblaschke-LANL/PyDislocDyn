! Some subroutines meant to be compiled for pydislocdyn via f2py
! run 'python -m numpy.f2py -c subroutines.f90 -m subroutines' to use
! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - Sept. 14, 2025

subroutine version(versionnumber)
  integer, intent(out) :: versionnumber
  versionnumber=20250914
end subroutine version

module parameters
implicit none
integer,parameter :: sel = selected_real_kind(10)
real(kind=sel), parameter :: hbar = 1.0545718d-34       ! reduced Planck constant
real(kind=sel), parameter :: kB = 1.38064852d-23        ! Boltzmann constant
real(kind=sel), parameter :: pi = (4.d0*atan(1.d0)) ! pi
real(kind=sel), parameter :: pi2 = (4.d0*atan(1.d0))**2 ! pi squared
end module parameters

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
! this wrapper parallelizes thesum() which is a subroutine of dragcoeff_iso_computepoly() in phononwind.py
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
! this is a subroutine of dragcoeff_iso_computepoly() in phononwind.py
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

subroutine dragintegrand(output,prefactor,dij,flatpoly,lent,lenph)
! this is a subroutine of dragcoeff_iso() in phononwind.py
implicit none

integer,parameter :: sel = selected_real_kind(6)
integer :: i, j, ij, k, kk, n, nn
integer, intent(in) :: lent, lenph
real(kind=sel), intent(in), dimension(lent,lenph) :: prefactor
real(kind=sel), intent(in), dimension(lenph,3,3) :: dij
real(kind=sel), intent(in), dimension(lent*lenph,3,3,3,3) :: flatpoly
real(kind=sel), intent(out), dimension(lent,lenph) :: output

output(:,:) = 0.0


do nn=1,3
  do n=1,3
    do kk=1,3
      do k=1,3
        do j = 1,lenph
          do i = 1,lent
            ij = (i-1)*lenph+j
            output(i,j) = output(i,j) - dij(j,k,kk)*dij(j,n,nn)*flatpoly(ij,k,kk,n,nn)
          end do
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
! Compute the bracket (A,B) := A.Cmat.B, where Cmat is a tensor of 2nd order elastic constants.
! All three variables have an additional disloc. character dependence.
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
! Compute the bracket (A,B) := A.Cmat.B, where Cmat is a tensor of 2nd order elastic constants.
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
! integrate using the trapezoidal rule
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

SUBROUTINE cumtrapz(f,x,n,intf)
! cumulatively integrate using the trapezoidal rule
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: n
  REAL(KIND=sel), INTENT(IN), DIMENSION(n)  :: f, x
  REAL(KIND=sel), INTENT(OUT), DIMENSION(n) :: intf
  REAL(KIND=sel) :: tmp
  INTEGER :: i
 
  intf(1) = 0.d0; tmp =0.d0
  do i=1,n-1
    tmp = tmp + 0.5d0*(f(i+1)+f(i))*(x(i+1)-x(i))
    intf(i+1) = tmp
  end do
  
  RETURN
END SUBROUTINE cumtrapz

!!**********************************************************************

SUBROUTINE fourieruij_sincos(sincos,ra,rb,phix,q,ph,phixres,nq,phres)
! subroutine for one of the inputs of fourieruij_nocut()
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: phixres,nq,phres
  REAL(KIND=sel), INTENT(IN) :: ra, rb, phix(phixres), q(nq), ph(phres)
  REAL(KIND=sel), INTENT(OUT) :: sincos(phixres,phres)
  integer i,j
  real(kind=sel) :: cosphimph
  
  do j=1,phres
    do i=1,phixres
      cosphimph = cos(phix(i)-ph(j))
      sincos(i,j) = sum(cos(q*ra*cosphimph)-cos(q*rb*cosphimph))/cosphimph/nq
    end do
  end do
  
  RETURN
END SUBROUTINE fourieruij_sincos

!!**********************************************************************

SUBROUTINE fourieruij_nocut(fourieruij,uij,phix,sincos,ntheta,phres,phixres)
! Fourier transform of angular part of uij (needs result of subroutine fourieruij_sincos for sincos)
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: ntheta,phixres,phres
  REAL(KIND=sel), INTENT(IN) :: uij(phixres,3,3,ntheta), sincos(phixres,phres), phix(phixres)
  REAL(KIND=sel), INTENT(OUT) :: fourieruij(3,3,ntheta,phres)
  integer i,j,th,ph
  
  do ph=1,phres
    do th=1,ntheta
      do j=1,3
        do i=1,3
          call trapz(uij(:,i,j,th)*sincos(:,ph),phix,phixres,fourieruij(i,j,th,ph))
        end do
      end do
    end do
  end do
  
  RETURN
END SUBROUTINE fourieruij_nocut

!!**********************************************************************

SUBROUTINE accscrew_xyintegrand(integrand,x,y,t,xpr,a,b,c,ct,abc,ca,xcomp)
! subroutine of computeuij_acc_screw
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  logical, intent(in) :: xcomp
  REAL(KIND=sel), INTENT(IN) :: x,y,t,xpr,a,b,c,ct,abc,ca
  REAL(KIND=sel), INTENT(OUT) :: integrand
  real(kind=sel) :: rpr, eta, etatilde, tau, tau2, tau_min_R, tau_min_R2, stepfct, stepfct2
  
  rpr = sqrt((x-xpr)**2 - (x-xpr)*y*b/c + y**2/ct)
  eta = sqrt(2.d0*xpr/a)
  etatilde = sign(1.0d0,x)*sqrt(2.d0*abs(x)/a)*0.5d0*(1.d0+xpr/x)
  tau = t - eta
  tau_min_R = sqrt(abs(tau**2*abc/ct - rpr**2/(ct*ca**2)))
  stepfct = 0.5d0*(sign(1.d0,(t - eta - rpr/(ca*sqrt(abc))))+1.d0)
  tau2 = t - etatilde
  tau_min_R2 = sqrt(abs(tau2**2*abc/ct - rpr**2/(ct*ca**2)))
  stepfct2 =0.5d0*(sign(1.d0,(t - etatilde - rpr/(ca*sqrt(abc))))+1.d0)
  if (xcomp) then
    integrand = stepfct*((x-xpr-y*b/(2*c))*y/rpr**4)*(tau_min_R + tau**2*(abc/ct)/tau_min_R) &
                - (stepfct2*((x-xpr-y*b/(2*c))*y/rpr**4)*(tau_min_R2 + tau2**2*(abc/ct)/tau_min_R2)) !! subtract pole
  else
    integrand = stepfct*(1.d0/rpr**4)*((tau**2*y**2*abc/ct**2 - (x-xpr)*y*(b/(2.d0*c))*rpr**2/(ct*ca**2))/tau_min_R &
                   - (x-xpr)**2*(tau_min_R)) &
                - (stepfct2*(1.d0/rpr**4)*((tau2**2*y**2*abc/ct**2 - (x-xpr)*y*(b/(2*c))*rpr**2/(ct*ca**2))/tau_min_R2 &
                   - (x-xpr)**2*tau_min_R2))
  endif
  
  RETURN
END SUBROUTINE accscrew_xyintegrand

!!**********************************************************************

SUBROUTINE dragcoeff_iso_phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi,distri)
! this is a subroutine of dragcoeff_iso() in phononwind.py
!-----------------------------------------------------------------------
  use parameters
  IMPLICIT NONE
!-----------------------------------------------------------------------
  INTEGER, INTENT(IN) :: lenq1,lent,lenphi
  REAL(KIND=sel), INTENT(IN) :: T, c1qBZ, c2qBZ
  REAL(KIND=sel), INTENT(IN) :: q1(lenq1), q1h4(lenq1), prefac(lent,lenphi), OneMinBtqcosph1(lent,lenphi)
  REAL(KIND=sel), INTENT(OUT), DIMENSION(lent,lenphi,lenq1) :: distri
  INTEGER :: i
  REAL(KIND=sel) :: phonon1, phonon2(lent,lenphi), beta
  
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

SUBROUTINE elasticA3(C2, C3, A3)
! Returns the tensor of elastic constants as it enters the interaction of dislocations with phonons.
! Required inputs are the tensors of SOEC and TOEC.
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN)  :: C2(3,3,3,3), C3(3,3,3,3,3,3)
  REAL(KIND=sel), INTENT(OUT) :: A3(3,3,3,3,3,3)
  INTEGER :: i
  REAL(KIND=sel), DIMENSION(3,3,3,3) :: C2swap
  
  C2swap = reshape(C2, (/ 3, 3, 3, 3/), order = (/2,3,1,4/) )
  A3 = C3
  do i=1,3
    A3(:,:,i,:,i,:) = A3(:,:,i,:,i,:) + C2
    A3(i,:,:,:,i,:) = A3(i,:,:,:,i,:) + C2swap
    A3(i,:,i,:,:,:) = A3(i,:,i,:,:,:) + C2
  end do
  
  RETURN
END SUBROUTINE elasticA3

!!**********************************************************************

SUBROUTINE computeEtot(uij, betaj, C2, Cv, phi, Ntheta, Nphi, Wtot)
! Computes the self energy of a straight dislocation uij moving at velocity beta.
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
! invert 3x3 matrix A
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

SUBROUTINE linspace(start,finish,num,output)
! fortran implementation of np.linspace() for real numbers
!-----------------------------------------------------------------------
  IMPLICIT NONE
  integer,parameter :: sel = selected_real_kind(10)
!-----------------------------------------------------------------------
  real(kind=sel), intent(in) :: start, finish
  integer, intent(in) :: num
  real(kind=sel), intent(out), dimension(num) :: output
!--------- local vars --------------------------------------------------
  integer :: i
  real(kind=sel) :: step
  
  step = (finish - start) / (num-1.d0)
  output = (/(start + (i-1)*step, i=1,num)/)
  
  return
END SUBROUTINE linspace

!!**********************************************************************

SUBROUTINE computeuk(beta, C2, Cv, b, M, N, phi, r, Ntheta, Nphi, Nr, uk)
! Compute the dislocation displacement field uk.
!-----------------------------------------------------------------------
  use parameters
  IMPLICIT NONE
!-----------------------------------------------------------------------
  REAL(KIND=sel), INTENT(IN) :: beta
  INTEGER, INTENT(IN) :: Ntheta, Nphi, Nr
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3)  :: C2
  REAL(KIND=sel), INTENT(IN), DIMENSION(3,3,3,3,Ntheta)  :: Cv
  REAL(KIND=sel), INTENT(IN)  :: phi(Nphi), r(Nr)
  REAL(KIND=sel), INTENT(IN), DIMENSION(3)  :: b
  REAL(KIND=sel), INTENT(IN), DIMENSION(Nphi,3,Ntheta)  :: M, N
  REAL(KIND=sel), INTENT(OUT), DIMENSION(Nphi,3,Ntheta,Nr)  :: uk
  integer :: i, j, k, th, ph
  real(kind=sel), dimension(Nphi,3,3) :: MM, NN, MN, NM, NNinv, Sphi, Bphi
  real(kind=sel), dimension(3,3,3,3) :: tmpC
  real(kind=sel), dimension(3,3) :: S, BB
  real(kind=sel), dimension(3) :: Sb, BBb
  real(kind=sel) :: uiphi(Nphi,3), tmpu(Nphi,3)
  uk = 0.d0
  !$OMP PARALLEL DO default(shared), private(th,ph,j,k,i, &
  !$OMP                   MM,NN,MN,NM,NNinv,Sphi,Bphi,tmpC,S,BB,Sb,BBb,tmpu,uiphi)
  do th=1,Ntheta
  tmpu = 0.d0
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
  do i=1,3; do ph=1,Nphi
    tmpu(ph,i) = DOT_PRODUCT(NNinv(ph,i,:),BBb(:)) - DOT_PRODUCT(Sphi(ph,i,:),Sb(:))
  enddo; enddo
  uiphi = 0d0
  do i=1,3
    call cumtrapz(tmpu(:,i),phi,Nphi,uiphi(:,i))
  enddo
  do i=1,3; do j=1,Nr
    uk(:,i,th,j) = uiphi(:,i) - Sb(i)*log(r(j)/r(1))
  enddo; enddo; enddo
  !$OMP END PARALLEL DO
  RETURN
END SUBROUTINE computeuk


!!**********************************************************************

SUBROUTINE computeuij(beta, C2, Cv, b, M, N, phi, Ntheta, Nphi, uij)
! Compute the dislocation displacement gradient field uij.
!-----------------------------------------------------------------------
  use parameters
  IMPLICIT NONE
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
! this is a subroutine of dragcoeff_iso() in phononwind.py
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
! this is a subroutine of dragcoeff_iso() in phononwind.py
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

!!**********************************************************************

SUBROUTINE phononwind_xx(dij,A3,qBZ,ct,cl,beta,burgers,Temp,lentheta,lent,lenph,lenq1,lenph1,updatet,chunks,r0cut,debye,dragb)
! this is a subroutine of dragcoeff_iso() in phononwind.py (TT and LL modes)
!-----------------------------------------------------------------------
  use parameters
  IMPLICIT NONE
  integer,parameter :: selsm = selected_real_kind(6)
!-----------------------------------------------------------------------
  integer, intent(in) :: lentheta, lent, lenph, lenq1, lenph1
  real(kind=sel), intent(in), dimension(:,:,:,:,:,:,:) :: A3
  real(kind=sel), intent(in) :: qBZ, ct, cl, beta, burgers, Temp, r0cut
  logical, intent(in) :: updatet, debye
  integer, intent(in), dimension(2) :: chunks
  real(kind=sel), intent(in), dimension(lenph,3,3,lentheta) :: dij
  real(kind=sel), intent(out), dimension(lentheta) :: dragb
!----------- local vars ------------------------------------------------
  real(kind=sel) :: phi(lenph), q1(lenq1-1), phi1(lenph1), t(lent), q1h4(lenq1-1), betafactor(lent)
  real(kind=sel) :: qvec(lenph,3), csphi(lenph), distri(lent,lenph,lenq1-1)
  real(kind=sel) :: qtilde(lent,lenph), prefac(lent,lenph), OneMinBtqcosph1(lent,lenph), Bmx(lent,lenph)
  real(kind=sel) :: dt, tmin, tmax, prefac1, ctovcl, cqBZ, hbarcsqBZ_TkB
  real(kind=selsm) :: Bmix(lent,lenph), prefactor1(lent,lenph), flatpoly(lent*lenph,3,3,3,3), a3sm(3,3,3,3,3,3)
  real(kind=selsm) :: dijc(lenph,3,3), qv(lent*lenph,3), dphi1(lenph1-1), ph1(lenph1-1), delta(3,3)
  real(kind=selsm), dimension(lent*lenph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
  integer :: Nchunks, kthchk, Nt_total, i, j, k, th
  
  Nchunks = chunks(1)
  kthchk = chunks(2)
  Nt_total = 1 + Nchunks*(lent-1)
  call linspace(0.d0,2.d0*pi,lenph,phi)
  call linspace(0.d0,2.d0*pi,lenph1,phi1)
  call linspace(1.d0/(lenq1-1.d0),1.d0,lenq1-1,q1)
  qvec(:,1) = cos(phi)
  qvec(:,2) = sin(phi)
  qvec(:,3) = 0.d0
  csphi = abs(cos(phi))
  q1h4 = (qBZ*q1)**4
  ! if mode=LL, multiply by ct/cl in many places and keep delta=0, if mode=TT (i.e. cl=0) then delta-identity
  delta = 0.d0
  if (cl>0) then
    ctovcl = ct/cl
    cqBZ = cl*qBZ
  else
    ctovcl = 1.d0
    cqBZ = ct*qBZ
    do i=1,3
      delta(i,i) = 1.d0
    enddo
  endif
  prefac1 = (1.d3*pi*hbar*qBZ*burgers**2*ctovcl**3/(2*beta*(2*pi)**5))
  dphi1 = phi1(2:lenph1) - phi1(1:lenph1-1)
  ph1 = phi1(1:lenph1-1)
  
  if (Nchunks > 1) then
    tmin = (1.d0*kthchk)/Nchunks
    tmax = (1.d0+kthchk)/Nchunks
    if (updatet) then
      dt = (tmax-tmin)/(2.d0*lent)
      call linspace(tmin+dt,tmax-dt,lent,t)
    else
      call linspace(tmin,tmax,lent,t)
    endif
  else
    if (updatet) then
      dt = 1.d0/(2.d0*lent)
      call linspace(dt,1.d0-dt,lent,t)
    else
      call linspace(0.d0,1.d0,lent,t)
    endif
  endif
  
  do i=1,lenph
    qtilde(:,i) = 2.d0*(t-beta*ctovcl*csphi(i))/(1.d0-(beta*ctovcl*cos(phi(i)))**2) + tiny(1.)
    prefac(:,i) = prefac1*csphi(i)/(1.d0-(beta*ctovcl*csphi(i))**2)/qtilde(:,i)
    OneMinBtqcosph1(:,i) = 1.d0 - beta*ctovcl*qtilde(:,i)*csphi(i)
  enddo
  
  ! if debye, use a high temperature expansion of the Debye-fcts instead of (slower) integration over q1
  if (debye) then
    do j=1,lenph1
      betafactor = OneMinBtqcosph1(:,j)
      hbarcsqBZ_TkB = hbar*cqBZ/(Temp*kB)
      prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-(beta*ctovcl/2.d0)*qtilde(:,j)*csphi(j)/betafactor &
              +(hbarcsqBZ_TkB**2/36.d0)*(beta*ctovcl*qtilde(:,j)*csphi(j)) &
              -(hbarcsqBZ_TkB**4/(30.d0*4.d0*24.d0))*(1.d0-(betafactor)**3) &
              +(hbarcsqBZ_TkB**6/(42.d0*5.d0*720.d0))*(1.d0-(betafactor)**5))
    enddo
  else
    call dragcoeff_iso_phonondistri(prefac,Temp,cqBZ,cqBZ,q1,q1h4,OneMinBtqcosph1,lenq1-1,lent,lenph,distri)
    ! we cut off q1=0 to prevent divisions by zero, so compensate by doubling first interval
    distri(:,:,1) = 2.d0*distri(:,:,1)
    ! include cutoff if r0cut>0:
    if (r0cut>0.d0) then
      do i=1,(lenq1-1)
        do j=1,lenph1
          distri(:,j,i) = distri(:,j,i)/(1.d0 + (qBZ*r0cut)**2*q1(i)**2*qtilde(:,j)**2)
        enddo
      enddo
    endif
    prefac = 0.d0 ! reset and reuse variable for distri integrated over q1
    ! integrate over last axis (q1), speedup by looping over last variable instead of calling subroutine trapz
    do i=1,lenq1-2
      prefac = prefac + 0.5d0*(distri(:,:,i+1)+distri(:,:,i))*(q1(i+1)-q1(i))
    enddo
  endif
  prefactor1 = real(prefac,kind=selsm) ! fct dragintegrand needs kind=selsm
  !!!
  do i=1,lenph
    do j=1,lent
      do k=1,3
        qv((j-1)*lenph+i,k) = qtilde(j,i)*qvec(i,k)
      enddo !k
      k = (j-1)*lenph+i
      mag(k) = 1.d0 + qtilde(j,i)**2 - 2.d0*t(j)*qtilde(j,i)
      sqrtt(k) = sqrt(1.d0-t(j)**2)
      tcosphi(k) = t(j)*cos(phi(i))
      sqrtsinphi(k) = sqrtt(k)*sin(phi(i))
      tsinphi(k) = t(j)*sin(phi(i))
      sqrtcosphi(k) = sqrtt(k)*cos(phi(i))
    enddo !j
  enddo !i
  !!!
  if (size(A3,7)==1) then
    ! no need to call bottleneck parathesum() more than once in the isotropic limit
    a3sm = A3(:,:,:,:,:,:,1)
    call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta,delta,mag,a3sm, &
                      ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
    do th=1,lentheta
      dijc = dij(:,:,:,th)
      call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
      Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
      call integratetphi(Bmx,beta*ctovcl,t,phi,updatet,kthchk,lenph,lent,dragb(th))
    enddo
  else
    do th=1,lentheta
      dijc = dij(:,:,:,th)
      a3sm = A3(:,:,:,:,:,:,th)
      call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta,delta,mag,a3sm, &
                      ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
      call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
      Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
      call integratetphi(Bmx,beta*ctovcl,t,phi,updatet,kthchk,lenph,lent,dragb(th))
    enddo !th
  endif
  
  RETURN
END SUBROUTINE phononwind_xx

!!**********************************************************************

SUBROUTINE phononwind_xy(dij,A3,qBZ,cx,cy,beta,burgers,Temp,lentheta,lent,lenph,lenq1,lenph1,updatet,chunks,r0cut,debye,dragb)
! this is a subroutine of dragcoeff_iso() in phononwind.py (mixed modes)
!-----------------------------------------------------------------------
  use parameters
  IMPLICIT NONE
  integer,parameter :: selsm = selected_real_kind(6)
!-----------------------------------------------------------------------
  integer, intent(in) :: lentheta, lent, lenph, lenq1, lenph1
  real(kind=sel), intent(in), dimension(:,:,:,:,:,:,:) :: A3
  real(kind=sel), intent(in) :: qBZ, cx, cy, beta, burgers, Temp, r0cut
  logical, intent(in) :: updatet, debye
  integer, intent(in), dimension(2) :: chunks
  real(kind=sel), intent(in), dimension(lenph,3,3,lentheta) :: dij
  real(kind=sel), intent(out), dimension(lentheta) :: dragb
!----------- local vars ------------------------------------------------
  real(kind=sel) :: phi(lenph), q1(lenq1-1), phi1(lenph1), qtilde(lent), q1h4(lenq1-1), betafactor(lent)
  real(kind=sel) :: qvec(lenph,3), csphi(lenph), distri(lent,lenph,lenq1-1), q1limit(lent,lenph), qlimitratio(lent)
  real(kind=sel) :: t(lent,lenph), prefac(lent,lenph), OneMinBtqcosph1(lent,lenph), Bmx(lent,lenph)
  real(kind=sel) :: dt, subqtmin, subqtmax, prefac1, ctovcl, cqBZ, beta1, beta2, qt_min, qt_max, hbarcsqBZ_TkB
  real(kind=selsm) :: Bmix(lent,lenph), prefactor1(lent,lenph), flatpoly(lent*lenph,3,3,3,3), a3sm(3,3,3,3,3,3)
  real(kind=selsm) :: dijc(lenph,3,3), qv(lent*lenph,3), dphi1(lenph1-1), ph1(lenph1-1), delta1(3,3), delta2(3,3)
  real(kind=selsm), dimension(lent*lenph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
  integer :: Nchunks, kthchk, Nt_total, i, j, k, th
  
  Nchunks = chunks(1)
  kthchk = chunks(2)
  Nt_total = 1 + Nchunks*(lent-1)
  call linspace(0.d0,2.d0*pi,lenph,phi)
  call linspace(0.d0,2.d0*pi,lenph1,phi1)
  call linspace(1.d0/(lenq1-1.d0),1.d0,lenq1-1,q1)
  qvec(:,1) = cos(phi)
  qvec(:,2) = sin(phi)
  qvec(:,3) = 0.d0
  csphi = abs(cos(phi))
  q1h4 = (qBZ*q1)**4
  delta1 = 0.d0
  delta2 = 0.d0
  cqBZ = cx*qBZ ! energy-cons delta fct. relating omega_2 to omega_1-Omega_q eliminated c2, beta from phonon-distri
  qt_max = 1+cx/cy
  if (cx>cy) then
    ctovcl = cy/cx
    beta1 = beta*ctovcl
    beta2 = beta
    do i=1,3
      delta2(i,i) = 1.d0
    enddo
  else
    ctovcl = cx/cy
    beta1 = beta
    beta2 = beta*ctovcl
    do i=1,3
      delta1(i,i) = 1.d0
    enddo
  endif
  qt_min = abs(1-cx/cy)/(1+beta2)
  prefac1 = -(1.d3*pi*hbar*qBZ*burgers**2*ctovcl**2/(4*beta1*(2*pi)**5))
  dphi1 = phi1(2:lenph1) - phi1(1:lenph1-1)
  ph1 = phi1(1:lenph1-1)
  
  if (Nchunks > 1) then
    subqtmin = qt_min + (qt_max-qt_min)*kthchk/Nchunks
    subqtmax = qt_min + (qt_max-qt_min)*(1.d0+kthchk)/Nchunks
    if (updatet) then
      dt = (subqtmax-subqtmin)/(2.d0*lent)
      call linspace(subqtmin+dt,subqtmax-dt,lent,qtilde)
    else
      call linspace(subqtmin,subqtmax,lent,qtilde)
    endif
  else
    if (updatet) then
      dt = 1.d0/(2.d0*lent)
      call linspace(qt_min+dt,qt_max-dt,lent,qtilde)
    else
      call linspace(qt_min,qt_max,lent,qtilde)
    endif
  endif
  
  do i=1,lenph
    t(:,i) = (qtilde+(1.d0-cx**2/cy**2)/qtilde)/2.d0 + (cx*beta2/cy)*csphi(i) - qtilde*(beta2*csphi(i))**2/2.d0
    prefac(:,i) = prefac1*csphi(i)/qtilde
    OneMinBtqcosph1(:,i) = 1.d0 - beta1*qtilde*csphi(i)
  enddo
  
  ! if debye, use a high temperature expansion of the Debye-fcts instead of (slower) integration over q1
  ! Note: for cx>cy the integration range is reduced (see below) and this expansion is not valid, skip in that case
  if (debye) then
    hbarcsqBZ_TkB = (hbar*cqBZ/(Temp*kB))**2
    do j=1,lenph1
      betafactor = OneMinBtqcosph1(:,j)
      if (cx>cy) then
        qlimitratio = (min(1.d0,ctovcl/OneMinBtqcosph1(:,j)))**2
        prefac(:,j) = prefac(:,j)*qlimitratio**2*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*( &
              -(beta1/2.d0)*qtilde*csphi(j)/betafactor &
              +(qlimitratio*hbarcsqBZ_TkB/36.d0)*(beta1*qtilde*csphi(j)) &
              -(qlimitratio**2*hbarcsqBZ_TkB**2/2.88d3)*(1.d0-(betafactor)**3) &
              +(qlimitratio**3*hbarcsqBZ_TkB**3/1.512d5)*(1.d0-(betafactor)**5))
      else
        prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-(beta1/2.d0)*qtilde*csphi(j)/betafactor &
              +(hbarcsqBZ_TkB/36.d0)*(beta1*qtilde*csphi(j)) &
              -(hbarcsqBZ_TkB**2/2.88d3)*(1.d0-(betafactor)**3) &
              +(hbarcsqBZ_TkB**3/1.512d5)*(1.d0-(betafactor)**5))
      endif
    enddo
  else
    call dragcoeff_iso_phonondistri(prefac,Temp,cqBZ,cqBZ,q1,q1h4,OneMinBtqcosph1,lenq1-1,lent,lenph,distri)
    ! we cut off q1=0 to prevent divisions by zero, so compensate by doubling first interval
    distri(:,:,1) = 2.d0*distri(:,:,1)
    distri(:,:,lenq1-1) = 2*distri(:,:,lenq1-1) ! see python code for explanation
    ! include cutoff if r0cut>0:
    if (r0cut>0.d0) then
      do i=1,(lenq1-1)
        do j=1,lenph1
          distri(:,j,i) = distri(:,j,i)/(1.d0 + (qBZ*r0cut)**2*q1(i)**2*qtilde(:)**2)
        enddo
      enddo
    endif
    ! if cx>cy, we need to limit the integration range of q1<=(cy/cx)/(1-beta1*qtilde*csphi) in addition to q1<=1
    if (cx>cy) then
      q1limit = ctovcl/OneMinBtqcosph1
      do i=1,(lenq1-1)
        do j=1,lenph1
          do k=1,lent
            if (q1(i)>q1limit(k,j)) then
              distri(k,j,i) = 0.d0
            endif
          enddo !k
        enddo !j
      enddo !i
    endif
    prefac = 0.d0 ! reset and reuse variable for distri integrated over q1
    ! integrate over last axis (q1), speedup by looping over last variable instead of calling subroutine trapz
    do i=1,lenq1-2
      prefac = prefac + 0.5d0*(distri(:,:,i+1)+distri(:,:,i))*(q1(i+1)-q1(i))
    enddo
  endif
  prefactor1 = real(prefac,kind=selsm) ! fct dragintegrand needs kind=selsm
  !!!
  do i=1,lenph
    do j=1,lent
      do k=1,3
        qv((j-1)*lenph+i,k) = qtilde(j)*qvec(i,k)
      enddo !k
      k = (j-1)*lenph+i
      mag(k) = 1.d0 + qtilde(j)**2 - 2.d0*t(j,i)*qtilde(j)
      sqrtt(k) = sqrt(abs(1.d0-t(j,i)**2))
      tcosphi(k) = t(j,i)*cos(phi(i))
      sqrtsinphi(k) = sqrtt(k)*sin(phi(i))
      tsinphi(k) = t(j,i)*sin(phi(i))
      sqrtcosphi(k) = sqrtt(k)*cos(phi(i))
    enddo !j
  enddo !i
  !!!
  if (size(A3,7)==1) then
    ! no need to call bottleneck parathesum() more than once in the isotropic limit
    a3sm = A3(:,:,:,:,:,:,1)
    call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,a3sm, &
                      ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
    do th=1,lentheta
      dijc = dij(:,:,:,th)
      call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
      Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
      call integrateqtildephi(Bmx,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,lenph,lent,dragb(th))
    enddo
  else
    do th=1,lentheta
      dijc = dij(:,:,:,th)
      a3sm = A3(:,:,:,:,:,:,th)
      call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,a3sm, &
                      ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
      call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
      Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
      call integrateqtildephi(Bmx,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,lenph,lent,dragb(th))
    enddo !th
  endif
  
  RETURN
END SUBROUTINE phononwind_xy
