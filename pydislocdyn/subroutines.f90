! Author: Daniel N. Blaschke
! Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
! Date: July 23, 2018 - July 21, 2026

!>defines various constants to be used elsewhere in the code
module dislocdyn_parameters
  implicit none
  integer,parameter :: sel = selected_real_kind(10)
  integer,parameter :: selsm = selected_real_kind(6)  !< some memory-heavy subroutines use lower precision in favor of speed
  integer,parameter :: version = 20260721
  real(kind=sel), parameter :: rzero = 2.d0*tiny(0.)
  real(kind=sel), parameter :: hbar = 1.0545718d-34       !< reduced Planck constant
  real(kind=sel), parameter :: kB = 1.38064852d-23        !< Boltzmann constant
  real(kind=sel), parameter :: pi = (4.d0*atan(1.d0)) !< number Pi
  real(kind=sel), parameter :: pi2 = (4.d0*atan(1.d0))**2 !< Pi squared
end module dislocdyn_parameters

!>this module contains a number of helper functions
module dislocdyn_utilities
  implicit none
  interface operator(.cross.)
    module procedure cross
  end interface
  interface operator(.inv.)
    module procedure inv
  end interface
  public :: ompinfo, elbrak, elbrak1d, cross, operator(.cross.), trapz, cumtrapz, inv, operator(.inv.), linspace
  contains
    !>returns the number of threads used for OpenMP parallelization (or 0 if compiled without OpenMP support)
    subroutine ompinfo(nthreads)
    !$   Use omp_lib
    integer, intent(out) :: nthreads
    nthreads = 0
    !$   nthreads = omp_get_max_threads()
    !~ !$   print*, 'OpenMP: using ',nthreads,' of ',omp_get_num_procs(),' processors'
    !~ !$   print*, 'type "export OMP_NUM_THREADS=n" before running this prog. to change'
    return
    end subroutine ompinfo

    !> Compute the bracket (A,B) := A.Cmat.B, where Cmat is a tensor of 2nd order elastic constants.
    !> All three variables have an additional disloc. character dependence.
    SUBROUTINE elbrak(a,b,Cmat,Ntheta,Nphi,AB)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
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
              end do
            end do
          end do
        end do
      end do
      !$OMP END PARALLEL DO
      
    END SUBROUTINE elbrak

    !> Compute the bracket (A,B) := A.Cmat.B, where Cmat is a tensor of 2nd order elastic constants.
    pure SUBROUTINE elbrak1d(a,b,Cmat,Nphi,AB)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
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
            end do
          end do
        end do
      end do
      
    END SUBROUTINE elbrak1d

    !!**********************************************************************

    !> computes the cross product of two 3-dim vectors x and y
    pure function cross(x,y) result(z)
      use dislocdyn_parameters, only : sel
      implicit none
      real(sel), dimension(3), intent(in)  :: x, y
      real(sel), dimension(3)  :: z
      z(1) = x(2)*y(3) - x(3)*y(2)
      z(2) = x(3)*y(1) - x(1)*y(3)
      z(3) = x(1)*y(2) - x(2)*y(1)
    end function cross

    !!**********************************************************************

    !> integrate using the trapezoidal rule
    pure SUBROUTINE trapz(f,x,n,intf)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      INTEGER, INTENT(IN) :: n
      REAL(KIND=sel), INTENT(IN), DIMENSION(n)  :: f, x
      REAL(KIND=sel), INTENT(OUT) :: intf
     
      intf = sum(0.5d0*(f(2:n)+f(1:n-1))*(x(2:n)-x(1:n-1)))
    END SUBROUTINE trapz

    !!**********************************************************************

    !> cumulatively integrate using the trapezoidal rule
    pure SUBROUTINE cumtrapz(f,x,n,intf)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
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
      
    END SUBROUTINE cumtrapz

    !!**********************************************************************

    !> invert 3x3 matrix A
    pure function inv(A) result(invA)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      REAL(KIND=sel), INTENT(IN), DIMENSION(3,3)  :: A
      REAL(KIND=sel), DIMENSION(3,3)  :: invA
      REAL(KIND=sel) :: det !, eps(3,3,3)
    !~   INTEGER :: i, j, k, l, m, n
    !~   eps = 0.d0
    !~   eps(1,2,3) = 1.d0; eps(3,1,2) = 1.d0; eps(2,3,1) = 1.d0
    !~   eps(3,2,1) = -1.d0; eps(1,3,2) = -1.d0; eps(2,1,3) = -1.d0
      
      ! note: hard coding this loop speeds up things by factor 6/27
    !~   do i=1,3; do j=1,3; do k=1,3
    !~     det = det + eps(i,j,k)*A(1,i)*A(2,j)*A(3,k)
    !~   end do; end do; end do
      det = A(1,1)*A(2,2)*A(3,3) + A(1,3)*A(2,1)*A(3,2) + A(1,2)*A(2,3)*A(3,1) &
            - A(1,3)*A(2,2)*A(3,1) - A(1,1)*A(2,3)*A(3,2) - A(1,2)*A(2,1)*A(3,3)
      
      ! note: hard coding this loop speeds up things by factor 18/27**2
    !~   invA = 0.d0
    !~   do i=1,3; do j=1,3; do k=1,3
    !~     do l=1,3; do m=1,3; do n=1,3
    !~       invA(i,j) = invA(i,j) + 0.5d0*eps(j,k,l)*eps(i,m,n)*A(k,m)*A(l,n)
    !~     end do; end do; end do
    !~   end do; end do; end do
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
    END function inv

    !!**********************************************************************

    !> fortran implementation of np.linspace() for real numbers
    pure SUBROUTINE linspace(start,finish,num,output)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
    !-----------------------------------------------------------------------
      real(kind=sel), intent(in) :: start, finish
      integer, intent(in) :: num
      real(kind=sel), intent(out), dimension(num) :: output
    !--------- local vars --------------------------------------------------
      integer :: i
      real(kind=sel) :: step
      
      if (num==1) then
        output = start
      else
        step = (finish - start) / (num-1.d0)
        output = [(start + (i-1)*step, i=1,num)]
      end if
    END SUBROUTINE linspace

end module dislocdyn_utilities

!!**********************************************************************

module dislocdyn_subroutines
  implicit none
  public :: accscrew_xyintegrand, computeEtot, strohgeometry, computeuk, computeuij
  contains
    !> subroutine of python function computeuij_acc_screw(), see PyDislocDyn docs
    SUBROUTINE accscrew_xyintegrand(integrand,x,y,t,xpr,a,b,c,ct,abc,ca,xcomp)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      IMPLICIT NONE
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
      end if
      
    END SUBROUTINE accscrew_xyintegrand

    !!**********************************************************************

    !> Computes the self energy of a straight dislocation uij moving at velocity beta.
    SUBROUTINE computeEtot(uij, betaj, C2, Cv, phi, Ntheta, Nphi, Wtot)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : trapz
      IMPLICIT NONE
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
              end do
            end do
          end do
        end do
        call trapz(Wdensity,phi,Nphi,Wtot(th))
      end do
      !$OMP END PARALLEL DO
      
    END SUBROUTINE computeEtot

    !!**********************************************************************
    !> computes several arrays to be used in the computation of a dislocation displacement gradient field for crystals
    !> using the integral version of the Stroh method
    pure subroutine strohgeometry(b,n0,t,m0,M,N,Cv,theta,phi,ntheta,nphi)
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : operator(.cross.)
      implicit none
      integer, intent(in) :: ntheta, nphi
      real(sel), intent(in) :: b(3), n0(3)
      real(sel), intent(in) :: theta(ntheta), phi(nphi)
      real(sel), intent(out), dimension(3,ntheta) :: t, m0
      real(sel), intent(out), dimension(nphi,3,ntheta) :: M, N
      real(sel), intent(out), dimension(3,3,3,3,ntheta)  :: Cv
      real(sel), dimension(3) :: x
      integer i,j,k,th,ph
      Cv = 0.d0
      x = b .cross. n0
      do concurrent (th=1:ntheta)
        t(:,th) = cos(theta(th))*b + x*sin(theta(th))
        m0(:,th) = n0 .cross. t(:,th)
        do i=1,3
          do j=1,3
            do k=1,3
              Cv(k,j,j,i,th) = m0(k,th)*m0(i,th)
            end do
          end do
        end do
        do ph=1, nphi
          M(ph,:,th) = m0(:,th)*cos(phi(ph)) + n0*sin(phi(ph))
          N(ph,:,th) = n0*cos(phi(ph)) - m0(:,th)*sin(phi(ph))
        end do
      end do
    end subroutine strohgeometry

    !!**********************************************************************
    !> Computes the dislocation displacement field uk.
    SUBROUTINE computeuk(beta, C2, Cv, b, M, N, phi, r, Ntheta, Nphi, Nr, uk)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel, pi2
      use dislocdyn_utilities, only : elbrak1d, operator(.inv.), trapz, cumtrapz
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
        NNinv(ph,:,:) = .inv. NN(ph,:,:)
      end do
      do j=1,3; do k=1,3; do i=1,3; do ph=1,Nphi
        Sphi(ph,i,j) = Sphi(ph,i,j) - NNinv(ph,i,k)*NM(ph,k,j)
      end do; end do; end do; end do
      do j=1,3; do i=1,3; do ph=1,Nphi
        Bphi(ph,i,j) = MM(ph,i,j)
        do k=1,3
          Bphi(ph,i,j) = Bphi(ph,i,j) + MN(ph,i,k)*Sphi(ph,k,j)
        end do
      end do; end do; end do
        do i=1,3; do j=1,3
          call trapz(Sphi(:,j,i),phi,Nphi,S(j,i))
          call trapz(Bphi(:,j,i),phi,Nphi,BB(j,i))
        end do; end do
        Sb(:) = (0.25d0/pi2)*MATMUL(S(:,:),b)
        BBb(:) = (0.25d0/pi2)*MATMUL(BB(:,:),b)
      do i=1,3; do ph=1,Nphi
        tmpu(ph,i) = DOT_PRODUCT(NNinv(ph,i,:),BBb(:)) - DOT_PRODUCT(Sphi(ph,i,:),Sb(:))
      end do; end do
      uiphi = 0d0
      do i=1,3
        call cumtrapz(tmpu(:,i),phi,Nphi,uiphi(:,i))
      end do
      do i=1,3; do j=1,Nr
        uk(:,i,th,j) = uiphi(:,i) - Sb(i)*log(r(j)/r(1))
      end do; end do; end do
      !$OMP END PARALLEL DO

    END SUBROUTINE computeuk

    !!**********************************************************************
    !>Computes the dislocation displacement gradient field uij according to the integral method
    SUBROUTINE computeuij(beta, C2, Cv, b, M, N, phi, Ntheta, Nphi, uij)
    !-----------------------------------------------------------------------
      use dislocdyn_parameters, only : sel, pi2
      use dislocdyn_utilities, only : elbrak1d, operator(.inv.), trapz
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
        NNinv(ph,:,:) = .inv. NN(ph,:,:)
      end do
      do j=1,3; do k=1,3; do i=1,3; do ph=1,Nphi
        Sphi(ph,i,j) = Sphi(ph,i,j) - NNinv(ph,i,k)*NM(ph,k,j)
      end do; end do; end do; end do
      do j=1,3; do i=1,3; do ph=1,Nphi
        Bphi(ph,i,j) = MM(ph,i,j)
        do k=1,3
          Bphi(ph,i,j) = Bphi(ph,i,j) + MN(ph,i,k)*Sphi(ph,k,j)
        end do
      end do; end do; end do
        do i=1,3; do j=1,3
          call trapz(Sphi(:,j,i),phi,Nphi,S(j,i))
          call trapz(Bphi(:,j,i),phi,Nphi,BB(j,i))
        end do; end do
        Sb(:) = (0.25d0/pi2)*MATMUL(S(:,:),b)
        BBb(:) = (0.25d0/pi2)*MATMUL(BB(:,:),b)
      do j=1,3; do i=1,3; do ph=1,Nphi
        uij(ph,i,j,th) = uij(ph,i,j,th) - Sb(i)*M(ph,j,th) &
                        + N(ph,j,th)*(DOT_PRODUCT(NNinv(ph,i,:),BBb(:)) - DOT_PRODUCT(Sphi(ph,i,:),Sb(:)))
      end do; end do; end do; end do
      !$OMP END PARALLEL DO

    END SUBROUTINE computeuij

    !!**********************************************************************
    !>Subroutine for computing the limiting velocity of an edge dislocation with reflection symmetry;
    !>C2 must be provided in Cartesian coordinates, rotated to align with the dislocation, and normalized by e.g. c44
    !>in which case one must provide also norm=c44/rho
    pure function edgevlim_of_phi(phi,i,C2,norm) result(vlim)
      use dislocdyn_parameters, only : sel
      use dislocdyn_utilities, only : elbrak1d
      implicit none
      real(sel), intent(in) :: phi, C2(3,3,3,3), norm
      integer, intent(in) :: i
      real(sel) :: vlim
      ! local variables
      real(sel) :: M(1,3), MM(1,3,3), Q, R, tmpout, cosph
      integer :: j
      cosph = cos(phi)
      M(1,:) = [cosph,sin(phi),0.d0]
      call elbrak1d(M,M,C2,1,MM)
      Q = 0.d0
      do j=1,2
        Q = Q + MM(1,j,j)
      end do
      R = MM(1,1,1)*MM(1,2,2) - MM(1,1,2)*MM(1,2,1)
      tmpout = 0.5d0*Q + i*sqrt(0.25d0*Q**2-R)
      vlim = abs(sqrt(tmpout*norm)/cosph)
    end function edgevlim_of_phi 

    !!**********************************************************************
    !>Subroutine for computing limiting velocities of a dislocation;
    !>C2 must be provided in Cartesian coordinates and normalized by e.g. c44
    !>in which case one must provide also norm=c44/rho
    !>if C2 has been rotated to align with the dislocation, pass m0=[1,0,0] and n0=[0,1,0]
    pure function vlim_of_phi(phi,i,C2,norm,m0,n0) result(vlim)
      use dislocdyn_parameters, only : sel, pi
      use dislocdyn_utilities, only : elbrak1d
      implicit none
      real(sel), intent(in) :: phi, C2(3,3,3,3), norm, m0(3), n0(3)
      integer, intent(in) :: i
      real(sel) :: vlim
      ! local variables
      real(sel) :: M(1,3), MM(1,3,3), MM2(3,3), P, Q, R, a, d, gam, tmpout, cosph
      integer :: j
      cosph = cos(phi)
      M(1,:) = m0*cosph + n0*sin(phi)
      call elbrak1d(M,M,C2,1,MM)
      P = 0.d0
      do j=1,3
        P = P - MM(1,j,j)
      end do
      Q = 0.5d0*P**2
      MM2(:,:) = 0.5d0*matmul(MM(1,:,:),MM(1,:,:))
      do j=1,3
        Q = Q - MM2(j,j)
      end do
      R = -(MM(1,1,1)*MM(1,2,2)*MM(1,3,3) + MM(1,1,3)*MM(1,2,1)*MM(1,3,2)+MM(1,1,2)*MM(1,2,3)*MM(1,3,1) &
              -MM(1,1,3)*MM(1,2,2)*MM(1,3,1) - MM(1,1,1)*MM(1,2,3)*MM(1,3,2) - MM(1,1,2)*MM(1,2,1)*MM(1,3,3))
      a = Q - P**2/3.d0
      d = 2.d0*P**3/27.d0-Q*P/3.d0+R
      gam = -0.5*d/sqrt(-a**3/27.d0)
      gam = max(-1.d0,min(1.d0,gam))
      gam = acos(gam)
      tmpout = -P/3.d0 + 2.d0*sqrt(-a/3.d0)*cos((gam+2.d0*i*pi)/3.d0) ! i=3 is equivalent to i=0
      vlim = abs(sqrt(tmpout*norm)/cosph)
    end function vlim_of_phi 

end module dislocdyn_subroutines
