! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - July 21, 2026

!> this module contains subroutines for phononwind_xx() and phononwind_xy()
module dislocdyn_phononwind_subroutines
  implicit none
  private :: thesum
  public :: phonondistri, parathesum, dragintegrand, integratetphi, integrateqtildephi
  contains
    SUBROUTINE phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi,distri)
      use dislocdyn_parameters, only : sel, hbar, kb
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
      
    END SUBROUTINE phonondistri

    !!**********************************************************************

    subroutine thesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1,lenph1,lentph)
    use dislocdyn_parameters, only : selsm
    implicit none
    integer :: i, j, k, l, ii, jj, kk, n, nn, m, p
    integer, intent(in) :: lentph, lenph1
    real(kind=selsm), intent(in), dimension(lenph1) :: phi1, dphi1
    real(kind=selsm), dimension(lentph,3) :: qt, qtshift
    real(kind=selsm), dimension(lentph,3,3,3,3) :: A3qt2, part1, part2
    real(kind=selsm), intent(in), dimension(lentph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
    real(kind=selsm), intent(in), dimension(lentph,3) :: qv
    real(kind=selsm), intent(in), dimension(3,3) :: delta1, delta2
    real(kind=selsm), intent(in), dimension(3,3,3,3,3,3) :: A3
    real(kind=selsm), intent(inout), dimension(lentph,3,3,3,3) :: output

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
    do concurrent (kk = 1:3, k = 1:3, j = 1:3, i = 1:3)! local(jj,ii) shared(qt,qtshift,A3) reduce(+:A3qt2)
      do jj = 1,3
        do ii = 1,3
          A3qt2(:,i,j,k,kk) = A3qt2(:,i,j,k,kk) + qt(:,ii)*qtshift(:,jj)*A3(i,ii,j,jj,k,kk)
        end do
      end do
    end do

    part1(:,:,:,:,:) = 0.0
    do concurrent (kk = 1:3, k = 1:3, j = 1:3, l = 1:3)! local(i) shared(A3qt2,qt,delta1) reduce(+:part1)
      do i = 1,3
         part1(:,l,j,k,kk) = part1(:,l,j,k,kk) + (delta1(i,l) - qt(:,l)*qt(:,i))*A3qt2(:,i,j,k,kk)
      end do
    end do

    part2(:,:,:,:,:) = 0.0
    do concurrent (nn = 1:3, n = 1:3, j = 1:3, l = 1:3)! local(m) shared(A3qt2,qtshift,mag,delta2) reduce(+:part2)
      do m = 1,3
       part2(:,l,j,n,nn) = part2(:,l,j,n,nn) + (delta2(j,m) - qtshift(:,j)*qtshift(:,m)/mag)*A3qt2(:,l,m,n,nn)
      end do
    end do
    part2(:,:,:,:,:) = part2(:,:,:,:,:)*dphi1(p)

    do concurrent (nn = 1:3, n = 1:3, kk = 1:3, k = 1:3)! local(j,l) shared(part1,part2) reduce(+:output)
      do j = 1,3
         do l = 1,3
            output(:,k,kk,n,nn) = output(:,k,kk,n,nn) + part1(:,l,j,k,kk)*part2(:,l,j,n,nn)
         end do
      end do
    end do

    end do
    !~ !$OMP END PARALLEL DO

    end subroutine thesum
    
    ! **********************************************************************
    
    !> this wrapper parallelizes thesum()
    subroutine parathesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1, &
                          lenp,lent,lenph1,lentph)
    !$   Use omp_lib
    use dislocdyn_parameters, only : selsm
    use dislocdyn_utilities, only : ompinfo
    implicit none

    integer, intent(in) :: lentph, lenph1, lenp, lent
    real(kind=selsm), intent(in), dimension(lenph1) :: phi1, dphi1
    real(kind=selsm), intent(in), dimension(lentph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
    real(kind=selsm), intent(in), dimension(lentph,3) :: qv
    real(kind=selsm), intent(in), dimension(3,3) :: delta1, delta2
    real(kind=selsm), intent(in), dimension(3,3,3,3,3,3) :: A3
    real(kind=selsm), intent(out), dimension(lentph,3,3,3,3) :: output
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
    !$ end do
    !$OMP END PARALLEL DO
    !$ else
    call thesum(output,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,A3,phi1,dphi1,lenph1,lentph)
    !$ end if
    end subroutine parathesum

    !!**********************************************************************
    
    subroutine dragintegrand(output,prefactor,dij,flatpoly,lent,lenph)
    use dislocdyn_parameters, only : selsm
    implicit none

    integer :: i, j, ij, k, kk, n, nn
    integer, intent(in) :: lent, lenph
    real(kind=selsm), intent(in), dimension(lent,lenph) :: prefactor
    real(kind=selsm), intent(in), dimension(lenph,3,3) :: dij
    real(kind=selsm), intent(in), dimension(lent*lenph,3,3,3,3) :: flatpoly
    real(kind=selsm), intent(out), dimension(lent,lenph) :: output

    output(:,:) = 0.0
    do nn=1,3
      do n=1,3
        do kk=1,3
          do k=1,3
            do concurrent (j = 1:lenph, i = 1:lent)! local(ij) shared(dij,flatpoly) reduce(+:output)
              ij = (i-1)*lenph+j
              output(i,j) = output(i,j) - dij(j,k,kk)*dij(j,n,nn)*flatpoly(ij,k,kk,n,nn)
            end do
          end do
        end do
      end do 
    end do

    output(:,:) = prefactor(:,:)*output(:,:)

    end subroutine dragintegrand

    !!**********************************************************************

    SUBROUTINE integratetphi(B,beta,t,phi,updatet,kthchk,Nphi,Nt,Bresult)
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : trapz
      IMPLICIT NONE
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
        NBtmp = count(tmask)
        t1 = pack(t,tmask)
        Btmp = pack(B(:,p),tmask)
        Bt(p) = 0.d0
        if (NBtmp>1) then
          if ((updatet.eqv..True.).or.(kthchk==0)) then
            Btmp(1) = 2.d0*Btmp(1)
          end if
          if (updatet.eqv..True.) then
            Btmp(ubound(Btmp)) = 2.d0*Btmp(ubound(Btmp))
          end if
          call trapz(Btmp(:),t1(:),NBtmp,Bt(p))
        end if
      end do
      !$OMP END PARALLEL DO
      call trapz(Bt,phi,Nphi,Bresult)
      
    END SUBROUTINE integratetphi

    !!**********************************************************************

    SUBROUTINE integrateqtildephi(B,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,Nphi,Nt,Bresult)
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : trapz
      IMPLICIT NONE
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
        NBtmp = count(tmask)
        qt = pack(qtilde,tmask)
        Btmp = pack(B(:,p),tmask)
        Bt(p) = 0.d0
        if (NBtmp>1) then
          if ((updatet.eqv..True.).or.(kthchk==0)) then
            Btmp(1) = 2.d0*Btmp(1)
          end if
          if ((updatet.eqv..True.).or.(kthchk==(Nchunks-1))) then
            Btmp(ubound(Btmp)) = 2.d0*Btmp(ubound(Btmp))
          end if
          call trapz(Btmp(:),qt(:),NBtmp,Bt(p))
        end if
      end do
      !$OMP END PARALLEL DO
      call trapz(Bt,phi,Nphi,Bresult)
      
    END SUBROUTINE integrateqtildephi

end module dislocdyn_phononwind_subroutines

!!**********************************************************************
!> this module contains various subroutines for phonondrag() (both Fortran and Python implementations)
module dislocdyn_phononwind
  implicit none
  public :: phononwind_xx, phononwind_xy, elasticA3, fourieruij_sincos, fourieruij_nocut
  contains
    !> Returns the tensor of elastic constants as it enters the interaction of dislocations with phonons.
    !> Required inputs are the tensors of SOEC and TOEC.
    SUBROUTINE elasticA3(C2, C3, A3)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      REAL(KIND=sel), INTENT(IN)  :: C2(3,3,3,3), C3(3,3,3,3,3,3)
      REAL(KIND=sel), INTENT(OUT) :: A3(3,3,3,3,3,3)
      INTEGER :: i,j,k,l
      REAL(KIND=sel), DIMENSION(3,3,3,3) :: C2swap
      
      C2swap = reshape(C2, [3, 3, 3, 3], order = [2,3,1,4])
      if (sum(abs(C2swap-C2))<1.d-9) then
        print*,"ERROR: compiler does not support reshape intrinsic with optional 'order' parameter!!"
        print*,"using fall back code instead"
        do concurrent (i=1:3, j=1:3, k=1:3, l=1:3)
          C2swap(i,j,k,l) = C2(j,k,i,l)
        end do
      end if
      A3 = C3
      do i=1,3
        A3(:,:,i,:,i,:) = A3(:,:,i,:,i,:) + C2
        A3(i,:,:,:,i,:) = A3(i,:,:,:,i,:) + C2swap
        A3(i,:,i,:,:,:) = A3(i,:,i,:,:,:) + C2
      end do
      
    END SUBROUTINE elasticA3

    !!**********************************************************************

    !> subroutine for one of the inputs of fourieruij_nocut()
    SUBROUTINE fourieruij_sincos(sincos,ra,rb,phix,q,ph,phixres,nq,phres)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      INTEGER, INTENT(IN) :: phixres,nq,phres
      REAL(KIND=sel), INTENT(IN) :: ra, rb, phix(phixres), q(nq), ph(phres)
      REAL(KIND=sel), INTENT(OUT) :: sincos(phixres,phres)
      integer i,j
      real(kind=sel) :: cosphimph
      
      do concurrent (j=1:phres, i=1:phixres)! local(cosphimph) shared(phix,ph,q,ra,rb,nq)
          cosphimph = cos(phix(i)-ph(j))
          sincos(i,j) = sum(cos(q*ra*cosphimph)-cos(q*rb*cosphimph))/cosphimph/nq
      end do
      
    END SUBROUTINE fourieruij_sincos

    !!**********************************************************************

    !> Fourier transform of angular part of uij (needs result of subroutine fourieruij_sincos for sincos)
    SUBROUTINE fourieruij_nocut(fourieruij,uij,phix,sincos,ntheta,phres,phixres)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : trapz
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      INTEGER, INTENT(IN) :: ntheta,phixres,phres
      REAL(KIND=sel), INTENT(IN) :: uij(phixres,3,3,ntheta), sincos(phixres,phres), phix(phixres)
      REAL(KIND=sel), INTENT(OUT) :: fourieruij(phres,3,3,ntheta)
      integer i,j,th,ph
      
      do concurrent (th=1:ntheta, j=1:3, i=1:3, ph=1:phres)
        call trapz(uij(:,i,j,th)*sincos(:,ph),phix,phixres,fourieruij(ph,i,j,th))
      end do
      
    END SUBROUTINE fourieruij_nocut

    !!**********************************************************************

    !> this is a subroutine of phonondrag() (TT and LL modes, used by both fortran and python implementations)
    SUBROUTINE phononwind_xx(dij,A3,qBZ,ct,cl,beta,burgers,Temp,lentheta,lent,lenph,lenq1,lenph1,updatet,chunks,r0cut,debye,dragb)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel, selsm, hbar, kb, pi
      use dislocdyn_utilities, only : linspace
      use dislocdyn_phononwind_subroutines
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      integer, intent(in) :: lentheta, lent, lenph, lenq1, lenph1
      real(kind=sel), intent(in), dimension(:,:,:,:,:,:,:) :: A3
      real(kind=sel), intent(in) :: qBZ, ct, cl, beta, burgers, Temp, r0cut
      logical, intent(in) :: updatet, debye
      integer, intent(in), dimension(2) :: chunks
      real(kind=sel), intent(in), dimension(lenph,3,3,lentheta) :: dij
      real(kind=sel), intent(out), dimension(lentheta) :: dragb
    !----------- local vars ------------------------------------------------
      real(kind=sel) :: phi(lenph), q1(lenq1-1), phi1(lenph1), t(lent), q1h4(lenq1-1), betafactor(lent), otherfactor(lent)
      real(kind=sel) :: qvec(lenph,3), csphi(lenph), distri(lent,lenph,lenq1-1)
      real(kind=sel) :: qtilde(lent,lenph), prefac(lent,lenph), OneMinBtqcosph1(lent,lenph), Bmx(lent,lenph)
      real(kind=sel) :: dt, tmin, tmax, prefac1, ctovcl, cqBZ, hbarcsqBZ_TkB
      real(kind=selsm) :: Bmix(lent,lenph), prefactor1(lent,lenph), flatpoly(lent*lenph,3,3,3,3), a3sm(3,3,3,3,3,3)
      real(kind=selsm) :: dijc(lenph,3,3), qv(lent*lenph,3), dphi1(lenph1-1), ph1(lenph1-1), delta(3,3)
      real(kind=selsm), dimension(lent*lenph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
      integer :: Nchunks, kthchk, i, j, k, th
      
      Nchunks = chunks(1)
      kthchk = chunks(2)
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
        end do
      end if
      prefac1 = (1.d3*pi*hbar*qBZ*burgers**2*ctovcl**3/(2*(2*pi)**5))
      dphi1 = real(phi1(2:lenph1) - phi1(1:lenph1-1), kind=selsm)
      ph1 = real(phi1(1:lenph1-1), kind=selsm)
      
      if (Nchunks > 1) then
        tmin = (1.d0*kthchk)/Nchunks
        tmax = (1.d0+kthchk)/Nchunks
        if (updatet) then
          dt = (tmax-tmin)/(2.d0*lent)
          call linspace(tmin+dt,tmax-dt,lent,t)
        else
          call linspace(tmin,tmax,lent,t)
        end if
      else
        if (updatet) then
          dt = 1.d0/(2.d0*lent)
          call linspace(dt,1.d0-dt,lent,t)
        else
          call linspace(0.d0,1.d0,lent,t)
        end if
      end if
      
      do concurrent (i=1:lenph)
        qtilde(:,i) = 2.d0*(t-beta*ctovcl*csphi(i))/(1.d0-(beta*ctovcl*cos(phi(i)))**2) + tiny(1.)
        prefac(:,i) = prefac1*csphi(i)/(1.d0-(beta*ctovcl*csphi(i))**2)/qtilde(:,i)
        OneMinBtqcosph1(:,i) = 1.d0 - beta*ctovcl*qtilde(:,i)*csphi(i)
      end do
      
      !> if debye, use a high temperature expansion of the Debye-fcts instead of (slower) integration over q1
      if (debye) then
        hbarcsqBZ_TkB = hbar*cqBZ/(Temp*kB)
        if (abs(beta)<1.d-12) then
          do concurrent (j=1:lenph)
            betafactor = OneMinBtqcosph1(:,j)
            otherfactor = ctovcl*qtilde(:,j)*csphi(j)
            prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-0.5d0*otherfactor/betafactor &
                    +(hbarcsqBZ_TkB**2/36.d0)*otherfactor &
                    -(hbarcsqBZ_TkB**4/(2.88d3))*(3*otherfactor) &
                    +(hbarcsqBZ_TkB**6/(1.512d5))*5*otherfactor)
          end do
        else
          do concurrent (j=1:lenph)
            betafactor = OneMinBtqcosph1(:,j)
            otherfactor = ctovcl*qtilde(:,j)*csphi(j)
            prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-0.5d0*otherfactor/betafactor &
                    +(hbarcsqBZ_TkB**2/36.d0)*otherfactor &
                    -(hbarcsqBZ_TkB**4/(2.88d3))*(1.d0-(betafactor)**3)/beta &
                    +(hbarcsqBZ_TkB**6/(1.512d5))*(1.d0-(betafactor)**5)/beta)
          end do
        end if
      else
        call phonondistri(prefac/beta,Temp,cqBZ,cqBZ,q1,q1h4,OneMinBtqcosph1,lenq1-1,lent,lenph,distri)
        ! we cut off q1=0 to prevent divisions by zero, so compensate by doubling first interval
        distri(:,:,1) = 2.d0*distri(:,:,1)
        !> include cutoff if r0cut>0:
        if (r0cut>0.d0) then
          do concurrent (i=1:(lenq1-1), j=1:lenph)
            distri(:,j,i) = distri(:,j,i)/(1.d0 + (qBZ*r0cut)**2*q1(i)**2*qtilde(:,j)**2)
          end do
        end if
        prefac = 0.d0 ! reset and reuse variable for distri integrated over q1
        ! integrate over last axis (q1), speedup by looping over last variable instead of calling subroutine trapz
        do i=1,lenq1-2
          prefac = prefac + 0.5d0*(distri(:,:,i+1)+distri(:,:,i))*(q1(i+1)-q1(i))
        end do
      end if
      prefactor1 = real(prefac,kind=selsm) ! fct dragintegrand needs kind=selsm
      !-------------------------
      do concurrent (i=1:lenph, j=1:lent)
          do k=1,3
            qv((j-1)*lenph+i,k) = real(qtilde(j,i)*qvec(i,k), kind=selsm)
          end do
          k = (j-1)*lenph+i
          mag(k) = real(1.d0 + qtilde(j,i)**2 - 2.d0*t(j)*qtilde(j,i), kind=selsm)
          sqrtt(k) = real(sqrt(1.d0-t(j)**2), kind=selsm)
          tcosphi(k) = real(t(j)*cos(phi(i)), kind=selsm)
          sqrtsinphi(k) = real(sqrtt(k)*sin(phi(i)), kind=selsm)
          tsinphi(k) = real(t(j)*sin(phi(i)), kind=selsm)
          sqrtcosphi(k) = real(sqrtt(k)*cos(phi(i)), kind=selsm)
      end do
      !-------------------------
      if (size(A3,7)==1) then
        ! no need to call bottleneck parathesum() more than once in the isotropic limit
        a3sm = real(A3(:,:,:,:,:,:,1), kind=selsm)
        call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta,delta,mag,a3sm, &
                          ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
        do th=1,lentheta
          dijc = real(dij(:,:,:,th), kind=selsm)
          call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
          Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
          call integratetphi(Bmx,beta*ctovcl,t,phi,updatet,kthchk,lenph,lent,dragb(th))
        end do
      else
        do th=1,lentheta
          dijc = real(dij(:,:,:,th), kind=selsm)
          a3sm = real(A3(:,:,:,:,:,:,th), kind=selsm)
          call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta,delta,mag,a3sm, &
                          ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
          call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
          Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
          call integratetphi(Bmx,beta*ctovcl,t,phi,updatet,kthchk,lenph,lent,dragb(th))
        end do !th
      end if
      
    END SUBROUTINE phononwind_xx

    !!**********************************************************************

    !> this is a subroutine of phonondrag() (mixed modes, used by both fortran and python implementations)
    SUBROUTINE phononwind_xy(dij,A3,qBZ,cx,cy,beta,burgers,Temp,lentheta,lent,lenph,lenq1,lenph1,updatet,chunks,r0cut,debye,dragb)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel, selsm, hbar, kb, pi
      use dislocdyn_utilities, only : linspace
      use dislocdyn_phononwind_subroutines
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      integer, intent(in) :: lentheta, lent, lenph, lenq1, lenph1
      real(kind=sel), intent(in), dimension(:,:,:,:,:,:,:) :: A3
      real(kind=sel), intent(in) :: qBZ, cx, cy, beta, burgers, Temp, r0cut
      logical, intent(in) :: updatet, debye
      integer, intent(in), dimension(2) :: chunks
      real(kind=sel), intent(in), dimension(lenph,3,3,lentheta) :: dij
      real(kind=sel), intent(out), dimension(lentheta) :: dragb
    !----------- local vars ------------------------------------------------
      real(kind=sel) :: phi(lenph), q1(lenq1-1), phi1(lenph1), qtilde(lent), q1h4(lenq1-1), betafactor(lent), otherfactor(lent)
      real(kind=sel) :: qvec(lenph,3), csphi(lenph), distri(lent,lenph,lenq1-1), q1limit(lent,lenph), qlimitratio(lent)
      real(kind=sel) :: t(lent,lenph), prefac(lent,lenph), OneMinBtqcosph1(lent,lenph), Bmx(lent,lenph)
      real(kind=sel) :: dt, subqtmin, subqtmax, prefac1, ctovcl, cqBZ, beta1, beta2, qt_min, qt_max, hbarcsqBZ_TkB
      real(kind=selsm) :: Bmix(lent,lenph), prefactor1(lent,lenph), flatpoly(lent*lenph,3,3,3,3), a3sm(3,3,3,3,3,3)
      real(kind=selsm) :: dijc(lenph,3,3), qv(lent*lenph,3), dphi1(lenph1-1), ph1(lenph1-1), delta1(3,3), delta2(3,3)
      real(kind=selsm), dimension(lent*lenph) :: mag, tcosphi, sqrtsinphi, tsinphi, sqrtcosphi, sqrtt
      integer :: Nchunks, kthchk, i, j, k, th
      
      Nchunks = chunks(1)
      kthchk = chunks(2)
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
        end do
      else
        ctovcl = cx/cy
        beta1 = beta
        beta2 = beta*ctovcl
        do i=1,3
          delta1(i,i) = 1.d0
        end do
      end if
      qt_min = abs(1-cx/cy)/(1+beta2)
      prefac1 = -(1.d3*pi*hbar*qBZ*burgers**2*ctovcl**2/(4*(2*pi)**5))
      dphi1 = real(phi1(2:lenph1) - phi1(1:lenph1-1), kind=selsm)
      ph1 = real(phi1(1:lenph1-1), kind=selsm)
      
      if (Nchunks > 1) then
        subqtmin = qt_min + (qt_max-qt_min)*kthchk/Nchunks
        subqtmax = qt_min + (qt_max-qt_min)*(1.d0+kthchk)/Nchunks
        if (updatet) then
          dt = (subqtmax-subqtmin)/(2.d0*lent)
          call linspace(subqtmin+dt,subqtmax-dt,lent,qtilde)
        else
          call linspace(subqtmin,subqtmax,lent,qtilde)
        end if
      else
        if (updatet) then
          dt = 1.d0/(2.d0*lent)
          call linspace(qt_min+dt,qt_max-dt,lent,qtilde)
        else
          call linspace(qt_min,qt_max,lent,qtilde)
        end if
      end if
      
      do concurrent (i=1:lenph)
        t(:,i) = (qtilde+(1.d0-cx**2/cy**2)/qtilde)/2.d0 + (cx*beta2/cy)*csphi(i) - qtilde*(beta2*csphi(i))**2/2.d0
        prefac(:,i) = prefac1*csphi(i)/qtilde
        OneMinBtqcosph1(:,i) = 1.d0 - beta1*qtilde*csphi(i)
      end do
      
      !> if debye, use a high temperature expansion of the Debye-fcts instead of (slower) integration over q1
      ! Note: for cx>cy the integration range is reduced (see below) and this expansion is not valid, skip in that case
      if (debye) then
        hbarcsqBZ_TkB = (hbar*cqBZ/(Temp*kB))**2
        if (abs(beta1)<1.d-12) then
          do concurrent (j=1:lenph)
            betafactor = OneMinBtqcosph1(:,j)
            otherfactor = qtilde*csphi(j)
            if (cx>cy) then
              qlimitratio = (min(1.d0,ctovcl/OneMinBtqcosph1(:,j)))**2
              prefac(:,j) = prefac(:,j)*qlimitratio**2*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*( &
                    -0.5d0*otherfactor/betafactor &
                    +(qlimitratio*hbarcsqBZ_TkB/36.d0)*otherfactor &
                    -(qlimitratio**2*hbarcsqBZ_TkB**2/2.88d3)*3*otherfactor &
                    +(qlimitratio**3*hbarcsqBZ_TkB**3/1.512d5)*5*otherfactor)
            else
              prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-0.5d0*otherfactor/betafactor &
                    +(hbarcsqBZ_TkB/36.d0)*otherfactor &
                    -(hbarcsqBZ_TkB**2/2.88d3)*3*otherfactor &
                    +(hbarcsqBZ_TkB**3/1.512d5)*5*otherfactor)
            end if
          end do
        else
          do concurrent (j=1:lenph)
            betafactor = OneMinBtqcosph1(:,j)
            otherfactor = qtilde*csphi(j)
            if (cx>cy) then
              qlimitratio = (min(1.d0,ctovcl/OneMinBtqcosph1(:,j)))**2
              prefac(:,j) = prefac(:,j)*qlimitratio**2*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*( &
                    -0.5d0*otherfactor/betafactor &
                    +(qlimitratio*hbarcsqBZ_TkB/36.d0)*otherfactor &
                    -(qlimitratio**2*hbarcsqBZ_TkB**2/2.88d3)*(1.d0-(betafactor)**3) &
                    +(qlimitratio**3*hbarcsqBZ_TkB**3/1.512d5)*(1.d0-(betafactor)**5))
            else
              prefac(:,j) = prefac(:,j)*(kB*Temp*qBZ**4/(2.d0*cqBZ*hbar))*(-0.5d0*otherfactor/betafactor &
                    +(hbarcsqBZ_TkB/36.d0)*otherfactor &
                    -(hbarcsqBZ_TkB**2/2.88d3)*(1.d0-(betafactor)**3) &
                    +(hbarcsqBZ_TkB**3/1.512d5)*(1.d0-(betafactor)**5))
            end if
          end do
        end if
      else
        call phonondistri(prefac/beta1,Temp,cqBZ,cqBZ,q1,q1h4,OneMinBtqcosph1,lenq1-1,lent,lenph,distri)
        ! we cut off q1=0 to prevent divisions by zero, so compensate by doubling first interval
        distri(:,:,1) = 2.d0*distri(:,:,1)
        distri(:,:,lenq1-1) = 2*distri(:,:,lenq1-1) ! see python code for explanation
        !> include cutoff if r0cut>0:
        if (r0cut>0.d0) then
          do concurrent (i=1:(lenq1-1), j=1:lenph)
            distri(:,j,i) = distri(:,j,i)/(1.d0 + (qBZ*r0cut)**2*q1(i)**2*qtilde(:)**2)
          end do
        end if
        ! if cx>cy, we need to limit the integration range of q1<=(cy/cx)/(1-beta1*qtilde*csphi) in addition to q1<=1
        if (cx>cy) then
          q1limit = ctovcl/OneMinBtqcosph1
          do concurrent (i=1:(lenq1-1), j=1:lenph, k=1:lent)
            if (q1(i)>q1limit(k,j)) then
              distri(k,j,i) = 0.d0
            end if
          end do !i
        end if
        prefac = 0.d0 ! reset and reuse variable for distri integrated over q1
        ! integrate over last axis (q1), speedup by looping over last variable instead of calling subroutine trapz
        do i=1,lenq1-2
          prefac = prefac + 0.5d0*(distri(:,:,i+1)+distri(:,:,i))*(q1(i+1)-q1(i))
        end do
      end if
      prefactor1 = real(prefac,kind=selsm) ! fct dragintegrand needs kind=selsm
      !-------------------------
      do concurrent (i=1:lenph, j=1:lent)
        do k=1,3
          qv((j-1)*lenph+i,k) = real(qtilde(j)*qvec(i,k), kind=selsm)
        end do
        k = (j-1)*lenph+i
        mag(k) = real(1.d0 + qtilde(j)**2 - 2.d0*t(j,i)*qtilde(j), kind=selsm)
        sqrtt(k) = real(sqrt(abs(1.d0-t(j,i)**2)), kind=selsm)
        tcosphi(k) = real(t(j,i)*cos(phi(i)), kind=selsm)
        sqrtsinphi(k) = real(sqrtt(k)*sin(phi(i)), kind=selsm)
        tsinphi(k) = real(t(j,i)*sin(phi(i)), kind=selsm)
        sqrtcosphi(k) = real(sqrtt(k)*cos(phi(i)), kind=selsm)
      end do
      !-------------------------
      if (size(A3,7)==1) then
        ! no need to call bottleneck parathesum() more than once in the isotropic limit
        a3sm = real(A3(:,:,:,:,:,:,1), kind=selsm)
        call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,a3sm, &
                          ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
        do th=1,lentheta
          dijc = real(dij(:,:,:,th), kind=selsm)
          call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
          Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
          call integrateqtildephi(Bmx,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,lenph,lent,dragb(th))
        end do
      else
        do th=1,lentheta
          dijc = real(dij(:,:,:,th), kind=selsm)
          a3sm = real(A3(:,:,:,:,:,:,th), kind=selsm)
          call parathesum(flatpoly,tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv,delta1,delta2,mag,a3sm, &
                          ph1,dphi1,lenph,lent,lenph1-1,lent*lenph)
          call dragintegrand(Bmix,prefactor1,dijc,flatpoly,lent,lenph)
          Bmx = real(Bmix,kind=sel) ! integratetphi needs kind=sel again
          call integrateqtildephi(Bmx,beta1,qtilde,t,phi,updatet,kthchk,Nchunks,lenph,lent,dragb(th))
        end do !th
      end if
      
    END SUBROUTINE phononwind_xy
end module dislocdyn_phononwind
