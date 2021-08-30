# Compute the line tension of a moving dislocation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Aug. 5, 2021
'''This module contains a class, StrohGeometry, to calculate the displacement field of a steady state dislocation
   as well as various other properties. See also the more general Dislocation class defined in linetension_calcs.py,
   which inherits from the StrohGeometry class defined here and the metal_props class defined in polycrystal_averaging.py. '''
#################################
import numpy as np
from scipy.integrate import cumtrapz, quad
nonumba=False
try:
    from numba import jit
except ImportError:
    nonumba=True
    from functools import partial
    def jit(func=None,forceobj=True,nopython=False):
        '''define a dummy decorator if numba is unavailable at runtime'''
        if func is None:
            return partial(jit, forceobj=forceobj,nopython=nopython)
        return func
try:
    import subroutines as fsub
    usefortran = True
    ompthreads = fsub.ompinfo()
except ImportError:
    usefortran = False
    ompthreads = 0
    
def printthreadinfo(Ncores,ompthreads=ompthreads):
    '''print a message to screen informing whether joblib paralellization (Ncores) or OpenMP paralellization (ompthreads)
       or both are currently employed; also warn if imports of numba and/or subroutines failed.'''
    if Ncores > 1 and ompthreads == 0: # check if subroutines were compiled with OpenMP support
        print(f"using joblib parallelization with {Ncores} cores")
    elif Ncores > 1:
        print(f"Parallelization: joblib with {Ncores} cores and OpenMP with {ompthreads} threads")
    elif ompthreads > 0:
        print(f"using OpenMP parallelization with {ompthreads} threads")
    if nonumba: print("\nWARNING: cannot find just-in-time compiler 'numba', execution will be slower\n")
    if not usefortran:
        print("\nWARNING: module 'subroutines' not found, execution will be slower")
        print("run 'python -m numpy.f2py -c subroutines.f90 -m subroutines' to compile this module")
        print("OpenMP is also supported, e.g. with with gfortran: \n'python -m numpy.f2py --f90flags=-fopenmp -lgomp -c subroutines.f90 -m subroutines'\n")
    return None

### define the Kronecker delta
delta = np.diag((1,1,1))

def rotaround(v,s,c):
    '''Computes the rotation matrix with unit vector 'v' as the rotation axis and s,c are the sin/cos of the angle.'''
    if isinstance(v,np.ndarray) and v.dtype == np.dtype('O'):
        vx = np.zeros((3,3),dtype=object)
    else:
        vx = np.zeros((3,3))
    vx[0,1] = v[2]
    vx[1,0] = -v[2]
    vx[0,2] = -v[1]
    vx[2,0] = v[1]
    vx[1,2] = v[0]
    vx[2,1] = -v[0]
    out = delta +s*vx + np.dot(vx,vx)*(1-c)
    return out

class StrohGeometry:
    '''This class computes several arrays to be used in the computation of a dislocation displacement gradient field for crystals using the integral version of the Stroh method.
       Required input parameters are: the unit Burgers vector b, the slip plane normal n0, an array theta parametrizing the angle between dislocation line and Burgers vector,
       and the resolution Nphi of angles to be integrated over.
       Its initial attributes are: the velocity dependent shift Cv (to be added or subtracted from the tensor of 2nd order elastic constants) and
       the vectors M(theta,phi) and N(theta,phi) parametrizing the plane normal to the dislocation line, as well as the dislocation line direction t and unit vector m0 normal to n0 and t.
       Methods computeuij(), computeEtot(), and computeLT() call functions of the same name, storing results in attributes uij, Etot, and LT.
       Method computerot finally computes a rotation matrix that will align n0,t with Cartesian y,z directions.'''
    def __init__(self,b, n0, theta, Nphi):
        self.Ntheta = Ntheta = len(theta)
        self.theta = np.asarray(theta)
        self.phi = np.linspace(0,2*np.pi,Nphi)
        self.r = None
        self.b = np.asarray(b)
        bsq = np.dot(self.b,self.b)
        if bsq>1e-12 and abs(bsq-1)>1e-12:
            self.b = self.b/np.sqrt(bsq)
        self.n0 = np.asarray(n0)
        nsq = np.dot(self.n0,self.n0)
        if nsq>1e-12 and abs(nsq-1)>1e-12:
            self.n0 = self.n0/np.sqrt(nsq)
        self.t = np.zeros((Ntheta,3))
        self.m0 = np.zeros((Ntheta,3))
        self.Cv = np.zeros((3,3,3,3,Ntheta))
        self.M = np.zeros((3,Ntheta,Nphi))
        self.N = np.zeros((3,Ntheta,Nphi))
        
        self.t = np.outer(np.cos(self.theta),self.b) + np.outer(np.sin(self.theta),np.cross(self.b,self.n0))
        self.m0 = np.cross(self.n0,self.t)
        
        for i in range(3):
            self.M[i] = np.outer(self.m0[:,i],np.cos(self.phi)) + np.outer(np.repeat(self.n0[i],Ntheta),np.sin(self.phi))
            self.N[i] = np.outer(np.repeat(self.n0[i],Ntheta),np.cos(self.phi)) - np.outer(self.m0[:,i],np.sin(self.phi))
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        self.Cv[i,j,k,l] = self.m0[:,i]*delta[j,k]*self.m0[:,l]
        
        self.beta = 0
        self.C2_aligned=None
        self.C2norm = np.zeros((3,3,3,3)) # normalized
        self.sym = None ## keyword defining crystal symmetry, unknown until C2norm is set
        self.uij = np.zeros((3,3,Ntheta,Nphi))
        self.uij_aligned = np.zeros((3,3,Ntheta,Nphi))
        self.rot = np.zeros((Ntheta,3,3))
        self.Etot = np.zeros((Ntheta))
        self.LT = 0
        
    def __repr__(self):
        return f" b:\t {self.b}\n n0:\t {self.n0}\n beta:\t {self.beta}\n Ntheta:\t {self.Ntheta}"
        
    def computeuij(self, beta, C2=None, r=None, nogradient=False, debug=False):
        '''Compute the dislocation displacement gradient field according to the integral method (which in turn is based on the Stroh method).
           This function returns a 3x3xNthetaxNphi dimensional array.
           Required input parameters are the dislocation velocity beta and the 2nd order elastic constant tensor C2.
           If r is provided, the full displacement gradient (a 3x3xNthetaxNrxNphi dimensional array) is returned.
           If option nogradient is set to True, the displacement field (not its gradient) is returned: a 3xNthetaxNrxNphi dimensional array.
           In the latter two cases, the core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
        self.beta = beta
        if C2 is None:
            C2 = self.C2norm
        else:
            self.C2norm = C2
        if r is None:
            r = self.r
        else:
            self.r = r
        if usefortran and (r is None) and not nogradient and not debug:
            self.uij = np.moveaxis(fsub.computeuij(beta, C2, self.Cv, self.b, np.moveaxis(self.M,-1,0), np.moveaxis(self.N,-1,0), self.phi),0,-1)
        else:
            self.uij = computeuij(beta, C2, self.Cv, self.b, self.M, self.N, self.phi, r=r, nogradient=nogradient, debug=debug)
        if debug:
            self.Sb = self.uij['S.b']
            self.Bb = self.uij['B.b']
            self.NN = self.uij['NN']
            self.uij = self.uij['uij']
        
    def computeuij_acc(self,a,beta,burgers=None,rho=None,C2_aligned=None,phi=None,r=None,eta_kw=None,etapr_kw=None,t=None,shift=None,deltat=1e-3,fastapprox=False,beta_normalization=1):
        '''Computes the displacement gradient of an accelerating screw dislocation (based on  J. Mech. Phys. Solids 152 (2021) 104448, resp. arxiv.org/abs/2009.00167).
           For now, it is implemented only for slip systems with the required symmetry properties, that is the plane perpendicular to the dislocation line must be a reflection plane.
           In particular, a=acceleration, beta=v/c_A is a normalized velocity where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a).
           Keywords burgers and rho denote the Burgers vector magnitude and material density, respectively.
           C2_aligned is the tensor of SOECs in VOIGT notation rotated into coordinates aligned with the dislocation.
           r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core, i.e.
           x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi).
           Finally, more general dislocation motion can be defined via funcion eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
           likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
           Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
        self.beta = beta
        scrind = int((len(self.theta)-1)/2)
        if abs(self.theta[scrind]) > 1e-12:
            scrind=0
        if C2_aligned is None:
            C2_aligned = self.C2_aligned
        elif C2_aligned.shape==(6,6): ## check if we received C2_aligned only for screw rather than all characters
            C2_aligned=[C2_aligned]
            scrind=0
        if r is None:
            r = self.r
        else:
            self.r = r
        if burgers is None:
            burgers = self.burgers
        else:
            self.burgers = burgers
        if rho is None:
            rho = self.rho
        else:
            self.rho = rho
        if phi is None: phi=self.phi
        if r is None: r=burgers*np.linspace(0,1,250)
        test = np.abs(self.C2_aligned[scrind]/self.C2[3,3]) ## check for symmetry requirements
        if test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4] > 1e-12:
            raise ValueError("not implemented - slip plane is not a reflection plane")
        self.uij_acc_aligned = computeuij_acc(a,beta,burgers,C2_aligned[scrind],rho,phi,r,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift,deltat=deltat,fastapprox=fastapprox,beta_normalization=beta_normalization)
        
    def computerot(self,y = [0,1,0],z = [0,0,1]):
        '''Computes a rotation matrix that will align slip plane normal n0 with unit vector y, and line sense t with unit vector z.
           y, and z are optional arguments whose default values are unit vectors pointing in the y and z direction, respectively.'''
        cv = np.vdot(self.n0,y)
        if round(cv,15)==-1:
            rot1 = rotaround(z,0,-1)
        elif round(cv,15)==1:
            rot1 = delta
        else:
            v=np.cross(y,self.n0)
            sv = np.sqrt(np.vdot(v,v))
            rot1 = rotaround(v/sv,sv,cv)
        Ntheta = len(self.theta)
        rot = np.zeros((Ntheta,3,3))
        t=np.copy(self.t)
        for th in range(Ntheta):
            t[th] = np.dot(rot1,t[th])
        v = np.cross(z,t)
        for th in range(Ntheta):
            cv = np.vdot(t[th],z)
            if round(cv,15)==-1:
                rot[th] = np.dot(rotaround(y,0,-1),rot1)
            elif round(cv,15)==1:
                rot[th] = rot1
            else:
                sv=np.sqrt(np.vdot(v[th],v[th]))
                rot[th] = np.dot(rotaround(v[th]/sv,sv,cv),rot1)
        self.rot = rot
    
    def alignuij(self,accuracy=15):
        '''Rotates previously computed uij using rotation matrix rot (run computeuij and computerot methods first), and stores the result in attribute uij_aligned.'''
        n = self.uij.shape
        uijrotated = np.zeros(n)
        if len(n)==4:
            for th in range(len(self.theta)):
                uijrotated[:,:,th] = np.round(np.dot(self.rot[th],np.dot(self.rot[th],self.uij[:,:,th])),accuracy)
        else:
            for th in range(len(self.theta)):
                for ri in range(n[3]):
                    uijrotated[:,:,th,ri] = np.round(np.dot(self.rot[th],np.dot(self.rot[th],self.uij[:,:,th,ri])),accuracy)
        self.uij_aligned = uijrotated
        
    def computeEtot(self):
        '''Computes the self energy of a straight dislocation uij moving at velocity beta. (Requirement: run method .computeuij(beta,C2) first.)'''
        self.Etot = computeEtot(self.uij, self.beta, self.C2norm, self.Cv, self.phi)
        
    def computeLT(self):
        '''Computes the line tension prefactor of a straight dislocation by adding to its energy the second derivative of that energy w.r.t.
        the dislocation character theta. (Requirements: run methods computeuij(beta,C2) and computeEtot() first.)'''
        dtheta = abs(self.theta[1]-self.theta[0])
        self.LT = computeLT(self.Etot, dtheta)
        
        
#############################################################################

@jit
def ArrayDot(A,B):
    '''Dot product of matrices, except that each matrix entry is itself an array.'''
    Ashape = A.shape
    Bshape = B.shape
    nk = Ashape[0]
    nl = Bshape[1]
    no = max(Ashape[1],Bshape[0])
    # if numbers don't match, python loop will trigger an error, but jit-compiled will silently produce junk (tradeoff for speed)
    if len(Ashape)==2:
        N=1
    else:
        N=Ashape[-1]
    if len(Bshape)>2:
        N=max(N,Bshape[-1])
    AB = np.zeros((nk,nl,N))
    tmp = np.zeros((N))
    for k in range(nk):
        for l in range(nl):
            for o in range(no):
                np.add(np.multiply(A[k,o] , B[o,l] , tmp), AB[k,l], AB[k,l])
    return AB
    
if usefortran:
    ## gives faster results even for jit-compiled computeuij while forceobj=True there (see below)
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           All arguments are arrays, i.e. A and B have shape (3,Ntheta) where Ntheta is e.g. the number of character angles.'''
        return np.moveaxis(fsub.elbrak(np.moveaxis(A,-1,0),np.moveaxis(B,-1,0),elC),0,-1)
    def elbrak1d(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           This function is similar to elbrak(), but its arguments do not depend on the character angle, i.e. A, B have shape (3).'''
        return fsub.elbrak1d(A,B,elC)
else:
    @jit
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           All arguments are arrays, i.e. A and B have shape (3,Ntheta) where Ntheta is e.g. the number of character angles.'''
        Ntheta = len(A[0,:,0])
        Nphi = len(A[0,0])
        tmp = np.zeros((Nphi))
        AB = np.zeros((3,3,Ntheta,Nphi))
        for th in range(Ntheta):
            for l in range(3):
                for o in range(3):
                    for k in range(3):
                        for p in range(3):
                            # AB[l,o,th] += A[k,th]*elC[k,l,o,p,th]*B[p,th]
                            #### faster numba-jit code is generated if we write the above like this (equivalent in pure python):
                            np.add(AB[l,o,th] , np.multiply(np.multiply(A[k,th],elC[k,l,o,p,th],tmp),B[p,th],tmp) , AB[l,o,th])
        
        return AB
    @jit
    def elbrak1d(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           This function is similar to elbrak(), but its arguments do not depend on the character angle, i.e. A, B have shape (3).'''
        Nphi = len(A)
        AB = np.zeros((Nphi,3,3))
        for ph in range(Nphi):
            for l in range(3):
                for o in range(3):
                    for k in range(3):
                        for p in range(3):
                            AB[ph,l,o] += A[ph,k]*elC[k,l,o,p]*B[ph,p]
        return AB

@jit
def elbrak_alt(A,B,elC):
    '''Computes the contraction of matrices A, B with the tensor of 2nd order elastic constants elC.'''
    Ntheta = len(A[0,0,:,0])
    Nphi = len(A[0,0,0])
    tmp = np.zeros((Nphi))
    AB= np.zeros((Ntheta,Nphi))
 
    for th in range(Ntheta):
        for k in range(3):
            for l in range(3):
                for o in range(3):
                    for p in range(3):
                        np.add(AB[th] , np.multiply(np.multiply(A[k,l,th],elC[k,l,o,p],tmp),B[o,p,th],tmp) , AB[th])
        
    return AB

@jit
def heaviside(x):
    '''step function with convention heaviside(0)=1/2'''
    return (np.sign(x)+1)/2

@jit(nopython=True)
def accscrew_xintegrand(x,y,t,xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw):
    '''subroutine of computeuij_acc'''
    Rpr = np.sqrt((x-xpr)**2 - (x-xpr)*y*B/C + y**2/Ct)
    if eta_kw is None:
        eta = np.sqrt(2*xpr/a)
        etatilde = np.sign(x)*np.sqrt(2*abs(x)/a)*0.5*(1+xpr/x)
    else:
        eta = eta_kw(xpr)
        etatilde = eta_kw(x) + (xpr-x)*etapr_kw(x)
    tau = t - eta
    tau_min_R = np.sqrt(abs(tau**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct = heaviside(t - eta - Rpr/(cA*np.sqrt(ABC)) )
    tau2 = t - etatilde
    tau_min_R2 = np.sqrt(abs(tau2**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct2 = heaviside(t - etatilde - Rpr/(cA*np.sqrt(ABC)))
    xintegrand = stepfct*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R + tau**2*(ABC/Ct)/tau_min_R)
    return xintegrand - stepfct2*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R2 + tau2**2*(ABC/Ct)/tau_min_R2) ## subtract pole

@jit(nopython=True)
def accscrew_yintegrand(x,y,t,xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw):
    '''subroutine of computeuij_acc'''
    Rpr = np.sqrt((x-xpr)**2 - (x-xpr)*y*B/C + y**2/Ct)
    if eta_kw is None:
        eta = np.sqrt(2*xpr/a)
        etatilde = np.sign(x)*np.sqrt(2*abs(x)/a)*0.5*(1+xpr/x)
    else:
        eta = eta_kw(xpr)
        etatilde = eta_kw(x) + (xpr-x)*etapr_kw(x)
    tau = t - eta
    tau_min_R = np.sqrt(abs(tau**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct = heaviside(t - eta - Rpr/(cA*np.sqrt(ABC)) )
    tau2 = t - etatilde
    tau_min_R2 = np.sqrt(abs(tau2**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct2 = heaviside(t - etatilde - Rpr/(cA*np.sqrt(ABC)))
    yintegrand = stepfct*(1/Rpr**4)*((tau**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R - (x-xpr)**2*(tau_min_R))
    return  yintegrand - stepfct2*(1/Rpr**4)*((tau2**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R2 - (x-xpr)**2*tau_min_R2)

# @jit(nopython=True) ## cannot compile while using scipy.integrate.quad() inside this function
def computeuij_acc(a,beta,burgers,C2_aligned,rho,phi,r,eta_kw=None,etapr_kw=None,t=None,shift=None,deltat=1e-3,fastapprox=False,beta_normalization=1):
    '''For now, only pure screw is implemented for slip systems with the required symmetry properties.
       a=acceleration, beta=v/c_A where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a).
       C2_aligned is the tensor of SOECs in VOIGT notation rotated into coordinates aligned with the dislocation.
       Furthermore, x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi),
       i.e. r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core.
       Finally, more general dislocation motion can be defined via funcions eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
       likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
       Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
    A = C2_aligned[4,4]
    B = 2*C2_aligned[3,4]
    C = C2_aligned[3,3]
    cA = np.sqrt(A/rho)
    ABC = (1-B**2/(4*A*C))
    Ct = C/A
    if beta_normalization==1:
        v = beta*cA
    else:
        v = beta*beta_normalization
    if eta_kw is None:
        t = v/a
        shift = a*t**2/2  ## = v**2/(2*a) # distance covered by the disloc. when achieving target velocity
        # print("time we reach beta={}: t={}".format(beta,t))
    uxz = np.zeros((len(r),len(phi),2))
    uyz = np.zeros((len(r),len(phi),2))
    R = np.zeros((len(r),len(phi)))
    X = np.zeros((len(r),len(phi)))
    Y = np.zeros((len(r),len(phi)))
    uij = np.zeros((3,3,len(r),len(phi)))
    ### integrate.quad options (to trade-off accuracy for speed in the kinetic eqns.)
    quadepsabs=1.49e-04 ## absolute error tolerance; default: 1.49e-08
    quadepsrel=1.49e-04 ## relative error tolerance; default: 1.49e-08
    quadlimit=30 ## max no of subintervals; default: 50
    ###
    tv = t*np.array([1-deltat/2,1+deltat/2])
    for ri in range(len(r)):
        if abs(r[ri]) < 1e-25:
            r[ri]=1e-25
        for ph in range(len(phi)):
            x = r[ri]*np.cos(phi[ph]) + shift ### shift x to move with the dislocations
            y = r[ri]*np.sin(phi[ph])
            R[ri,ph] = np.sqrt(x**2 - x*y*B/C + y**2/Ct)
            X[ri,ph] = x
            Y[ri,ph] = y
            if not fastapprox: ## allow bypassing
                for ti in range(2):
                    uxz[ri,ph,ti] = quad(lambda xpr: accscrew_xintegrand(x,y,tv[ti],xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw), 0, np.inf, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)[0]
                    uyz[ri,ph,ti] = quad(lambda xpr: accscrew_yintegrand(x,y,tv[ti],xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw), 0, np.inf, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)[0]
    ## add static part and include burgers vector:
    if eta_kw is None:
        eta = np.sign(X)*np.sqrt(2*np.abs(X)/a)
        etapr = eta/(2*X)
        tau = t-0.5*eta
    else:
        eta = eta_kw(X)
        etapr = etapr_kw(X)
        tau = t-(eta-etapr*X)
    denom = tau*(tau-2*etapr*(X-Y*B/(2*C))) + (etapr*R)**2 - Y**2/(Ct*cA**2)
    heaviadd = heaviside(tau - R/(cA*np.sqrt(ABC)))
    rootadd = np.sqrt(np.abs(tau**2*ABC/Ct-R**2/(Ct*cA**2)))
    uxz_added = heaviadd*(Y/rootadd)\
        *(2*etapr*((X-Y*B/(2*C))/Ct)*(tau**2*ABC-(R**2)/(2*cA**2)) - tau*(tau**2-Y**2/(Ct*cA**2))*ABC/Ct)\
        /(R**2*(denom))
    uxz_static = Y*np.sqrt(ABC/Ct)/R**2
    # uxz_static = 0 ## for testing purposes
    # uxz_added = 0 ## for testing purposes
    uyz_added = (heaviadd/rootadd)\
        *(tau**2*(etapr)*ABC*(Y**2/Ct-X**2) + (X*etapr-tau)*(R**2/cA**2)*(X-Y*B/(2*C)) + X*tau*ABC*(tau**2-Y**2/(Ct*cA**2)))\
        /(R**2*Ct*(denom))
    uyz_static =  - X*np.sqrt(ABC/Ct)/R**2
    # uyz_static = 0 ## for testing purposes
    # uyz_added = 0 ## for testing purposes
    uij[2,0] = (burgers/(2*np.pi))*((uxz[:,:,1]-uxz[:,:,0])/deltat + uxz_static + uxz_added)
    uij[2,1] = (burgers/(2*np.pi))*((uyz[:,:,1]-uyz[:,:,0])/deltat + uyz_static + uyz_added)
    return uij

@jit(forceobj=True)  ## calls preventing nopython mode: np.dot with arrays >2D, np.moveaxis(), scipy.cumtrapz, np.linalg.inv with 3-D array arguments, and raise ValueError / debug option
def computeuij(beta, C2, Cv, b, M, N, phi, r=None, nogradient=False, debug=False):
    '''Compute the dislocation displacement gradient field according to the integral method (which in turn is based on the Stroh method).
       This function returns a 3x3xNthetaxNphi dimensional array (where the latter two dimensions encode the discretized dependence on theta and phi as explained below),
       which corresponds to the displacement gradient multiplied by the radius r (i.e. we only return the angular dependence).
       Required input parameters are: the dislocation velocity betaj, 2nd order elastic constant tensor C2, its velocity dependent shift Cv
       (and all three may be scaled by a common parameter like the mean shear modulus), the Burgers vector b, vectors M, N spanning the plane normal to the dislocation line,
       and the integration angle phi inside the plane spanned by M,N.
       If r is provided, the full displacement gradient (a 3x3xNthetaxNrxNphi dimensional array) is returned.
       If option nogradient is set to True, the displacement field (not its gradient) is returned: a 3xNthetaxNrxNphi dimensional array.
       In the latter two cases, the core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
    Ntheta = len(M[0,:,0])
    Nphi = len(phi)
    if r is not None:
        Nr = len(np.asarray(r))
    else:
        Nr = 0
    MM = np.zeros((3,3,Ntheta,Nphi))
    NN = np.zeros((3,3,Ntheta,Nphi))
    MN = np.zeros((3,3,Ntheta,Nphi))
    NM = np.zeros((3,3,Ntheta,Nphi))
    NNinv = np.zeros((3,3,Ntheta,Nphi))
    ## integrands without 1/2pi factors of B and S, former is later re-used for integrated B.b
    B = np.zeros((3,3,Ntheta,Nphi))
    S = np.zeros((3,3,Ntheta,Nphi))
    ## Sb = S.b (integrated over phi)
    Sb = np.empty((3,Ntheta))
    pi = np.pi
    
    tmp = np.empty((Nphi))
    bb = beta*beta
    tmpC = np.empty(Cv.shape)
    
    for th in range(Ntheta):
        for l in range(3):
            for o in range(3):
                for k in range(3):
                    for p in range(3):
                        tmpC[k,l,o,p,th] = C2[k,l,o,p] - bb*Cv[k,l,o,p,th]
                        
    MM = elbrak(M,M,tmpC)
    MN = elbrak(M,N,tmpC)
    NM = elbrak(N,M,tmpC)
    NN = elbrak(N,N,tmpC)
    NNinv = np.reshape(np.linalg.inv(np.reshape(NN,(3,3,(Ntheta)*Nphi)).T).T,(3,3,Ntheta,Nphi))
    
    for th in range(Ntheta):
        for k in range(3):
            for l in range(3):
                for o in range(3):
                    np.subtract(S[k,l,th] , np.multiply(NNinv[k,o,th] , NM[o,l,th] , tmp) , S[k,l,th])
                    
        
        for k in range(3):
            for l in range(3):
                np.add(B[k,l,th] , MM[k,l,th] , B[k,l,th])
                for o in range(3):
                    np.add(B[k,l,th] , np.multiply(MN[k,o,th] , S[o,l,th] , tmp) , B[k,l,th])
                    
    Sb = (1/(4*pi*pi))*np.dot(b,np.trapz(S,x=phi))
    B = (1/(4*pi*pi))*np.dot(b,np.trapz(B,x=phi))
    
    if nogradient:
        if Nr==0:
            raise ValueError("I need an array for r in conjunction with nogradient=True.")
        r0 = r[0] ## cutoff
        uiphi = np.zeros((3,Ntheta,Nphi))
        
        tmpu = np.zeros((3,Ntheta,Nphi))
        for th in range(Ntheta):
            for j in range(3):
                for p in range(3):
                    tmpu[j,th] += (NNinv[j,p,th]*B[p,th] - S[j,p,th]*Sb[p,th])
        uiphi = cumtrapz(tmpu,x=phi,initial=0)
        uij=np.moveaxis(np.reshape(np.outer(np.ones(Nr),uiphi)-np.outer(np.log(r/r0),np.outer(Sb,np.ones(Nphi))),(Nr,3,Ntheta,Nphi)),0,-2)
    else:
        uij = np.zeros((3,3,Ntheta,Nphi))
        for th in range(Ntheta):
            for k in range(3):
                for l in range(3):
                    uij[k,l,th] = uij[k,l,th] - Sb[k,th]*M[l,th]
                    for p in range(3):
                        uij[k,l,th] += N[l,th]*(NNinv[k,p,th]*B[p,th] - S[k,p,th]*Sb[p,th])
                        
        if Nr != 0:
            uij = np.moveaxis(np.reshape(np.outer(1/r,uij),(Nr,3,3,Ntheta,Nphi)),0,-2)
    if debug:
        uij = {'uij':uij,'S.b':Sb,'B.b':B,'NN':NN}
 
    return uij

def artan(x,y):
    '''returns a variation of np.arctan2(x,y): since numpys implementation jumps to negative values in 3rd and 4th quadrant, shift those by 2pi so that atan(tan(phi))=phi for phi=[0,2pi]'''
    out = np.arctan2(x,y)
    out += 2*np.pi*np.heaviside(-out,0)
    return out

############# isotropic case
def computeuij_iso(beta,ct_over_cl, theta, phi, r=None, nogradient=False):
    '''Compute the dislocation displacement gradient field in the isotropic limit.
       This function returns a 3x3xNthetaxNphi dimensional array (where the latter two dimensions encode the discretized dependence on theta and phi),
       which corresponds to the displacement gradient multiplied by the radius r over the magnitude of the Burgers vector (i.e. we only return the angular dependence).
       Required input parameters are: the dislocation velocity beta in units of transverse sound speed, the ratio of transverse to longitudinal sound speed,
       and two arrays encoding the discretized dependence on the angle theta between dislocation line and Burgers vector and the polar angle phi in the plane normal to the dislocation line.
       If r is provided, the full displacement gradient (a 3x3xNthetaxNrxNphi dimensional array) is returned.
       If option nogradient is set to True, the displacement field (not its gradient) is returned: a 3xNthetaxNrxNphi dimensional array.
       In the latter two cases, the core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
    if r is not None:
        Nr = len(np.asarray(r))
    else:
        Nr = 0
    Ntheta = len(theta)
    Nphi = len(phi)
    gamt = np.sqrt(1-beta**2) ## defined as 1/gamma
    gaml = np.sqrt(1-(ct_over_cl*beta)**2)
    uij = np.zeros((3,3,Ntheta,Nphi))
    pi = np.pi
    sinph = np.sin(phi)
    cosph = np.cos(phi)
    sinth = np.sin(theta)
    if nogradient:
        if Nr==0:
            raise ValueError("I need an array for r in conjunction with nogradient=True.")
        r0 = r[0] ## cutoff
        uk = np.zeros((3,Ntheta,Nr*Nphi))
        one = np.ones((Nr))
        if beta==0:
            crat2 = ct_over_cl**2
            atan = artan(sinph,cosph)
            ## edge parametrized by sin(theta)
            uk[0] = np.outer(np.sin(theta),np.outer(one,(atan + (1-crat2)*cosph*sinph)/(2*pi)))
            uk[1] = np.outer(np.sin(theta),np.outer(one,((1-crat2)*(sinph)**2)/(2*pi)) - np.outer((crat2*np.log(r**2/r0**2))/(4*pi),np.ones((Nphi))))
        else:
            x2 = np.outer(r**2,cosph**2)
            y2 = np.outer(r**2,sinph**2)
            gamt = np.sqrt(1-beta**2)  ## defined as 1/gamma
            gaml = np.sqrt(1-(ct_over_cl*beta)**2)
            atan = artan(sinph,(1/gamt)*cosph)
            atanL = artan(sinph,(1/gaml)*cosph)
            ## edge parametrized by sin(theta)
            uk[0] = np.outer(np.sin(theta),np.outer(one,(atanL - (1-beta**2/2)*atan)/(pi*beta**2)))
            uk[1] = np.outer(np.sin(theta),(gaml*np.log((x2 + y2*gaml**2)/r0**2) - (1/gamt)*(1-beta**2/2)*np.log((x2 + y2*gamt**2)/r0**2))/(2*pi*beta**2))
        ## screw parametrized by cos(theta)
        uk[2] = np.outer(np.cos(theta),np.outer(one,atan/(2*pi)))
        uij = np.reshape(uk,(3,Ntheta,Nr,Nphi))
    elif beta==0:
        crat2 = ct_over_cl**2
        denomT = 1
        ## edge parametrized by sin(theta)
        uij[0,0] = (-1/pi)*np.outer(sinth,sinph*((1-(crat2/2))*cosph**2+(crat2/2)*sinph**2))
        uij[0,1] = (1/pi)*np.outer(sinth,cosph*((1-(crat2/2))*cosph**2+(crat2/2)*sinph**2))
        uij[1,0] = (-1/pi)*np.outer(sinth,cosph*((1-(crat2/2))*sinph**2+(crat2/2)*cosph**2))
        uij[1,1] = (1/pi)*np.outer(sinth,sinph*((1-3*(crat2/2))*cosph**2-(crat2/2)*sinph**2))
        ## screw parametrized by cos(theta)
        uij[2,0] = (-1/(2*pi))*np.outer(np.cos(theta),sinph)
        uij[2,1] = (1/(2*pi))*np.outer(np.cos(theta),cosph)
    else:
        denomL = 1-(ct_over_cl*beta*sinph)**2
        denomT = 1-(beta*sinph)**2
        ## edge parametrized by sin(theta)
        uij[0,0] = (-1/(pi*beta**2))*np.outer(sinth,sinph*(gaml/denomL-(1-beta**2/2)*gamt/denomT))
        uij[0,1] = (1/(pi*beta**2))*np.outer(sinth,cosph*(gaml/denomL-(1-beta**2/2)*gamt/denomT))
        uij[1,0] = (1/(pi*beta**2))*np.outer(sinth,cosph*(gaml/denomL-(1-beta**2/2)/(denomT*gamt)))
        uij[1,1] = (1/(pi*beta**2))*np.outer(sinth,sinph*(gaml**3/denomL-(1-beta**2/2)*gamt/denomT))
        ## screw parametrized by cos(theta)
        uij[2,0] = (-1/(2*pi))*np.outer(np.cos(theta),sinph*gamt/denomT)
        uij[2,1] = (1/(2*pi))*np.outer(np.cos(theta),cosph*gamt/denomT)
    if Nr != 0 and not nogradient:
        uij = np.moveaxis(np.reshape(np.outer(1/r,uij),(Nr,3,3,Ntheta,Nphi)),0,-2)
    return uij

#### Fourier transform of uij, isotropic case
def fourieruij_iso(beta,ct_over_cl, theta, phi):
    '''Compute the dislocation displacement gradient field in the isotropic limit in Fourier space multiplied by iq/b (the radial coordinate over the magnitude of the Burgers vector),
       i.e. we only return the dependence on the (discretized) polar angle phi in Fourier space, and hence the result is a 3x3xNthetaxNphi dimensional array.
       Required input parameters are: the dislocation velocity beta in units of transverse sound speed, the ratio of transverse to longitudinal sound speed,
       and two arrays encoding the discretized dependence on the angle theta between dislocation line and Burgers vector and the polar angle phi.'''
    gamt = (1-beta**2) ## defined as 1/gamma**2
    gaml = (1-(ct_over_cl*beta)**2)
    uij = np.zeros((3,3,len(theta),len(phi)))
    sinph = np.sin(phi)
    cosph = np.cos(phi)
    sinth = np.sin(theta)
    denomL = 1-(ct_over_cl*beta*cosph)**2
    denomT = 1-(beta*cosph)**2
    if beta==0:
        crat2 = ct_over_cl**2
        ## edge parametrized by sin(theta)
        uij[0,0] = -np.outer(sinth,sinph*(sinph**2 + (2*crat2-1)*cosph**2))
        uij[0,1] = np.outer(sinth,cosph*(2*(1-crat2)*sinph**2 + 1))
        uij[1,0] = np.outer(sinth,cosph*(2*(1-crat2)*sinph**2 - 1))
        uij[1,1] = np.outer(sinth,sinph*(2*(1-crat2)*sinph**2 - 1))
    else:
        ## edge parametrized by sin(theta)
        uij[0,0] = (-2/(beta**2))*np.outer(sinth,sinph*(1/denomL-(1-beta**2/2)/denomT))
        uij[0,1] = (2/(beta**2))*np.outer(sinth,cosph*(gaml/denomL-(1-beta**2/2)*gamt/denomT))
        uij[1,0] = (2/(beta**2))*np.outer(sinth,cosph*(gaml/denomL-(1-beta**2/2)/denomT))
        uij[1,1] = (2/(beta**2))*np.outer(sinth,sinph*(gaml/denomL-(1-beta**2/2)/denomT))
    ## screw parametrized by cos(theta)
    uij[2,0] = -np.outer(np.cos(theta),sinph*1/denomT)
    uij[2,1] = np.outer(np.cos(theta),cosph*gamt/denomT)
    
    return uij

@jit
def fourieruij_sincos(r,phiX,q,ph):
    '''Subroutine for fourieruij() that computes q*sin(r*q*cos(phiX-ph)) integrated over r, resulting in an array of shape (len(q),len(phiX)*len(ph)).
       All input parameters must be 1-dim. arrays, although only the cutoffs r[0] and r[-1] are used since the integration over r is done analytically.'''
    phres = len(ph)
    qres = len(q)
    phiXres = len(phiX)
    cosphimph = np.reshape(np.cos(np.outer(np.ones((phres)),phiX)-np.outer(ph,np.ones((phiXres)))),(phres*phiXres))
    out = np.zeros((qres,phres*phiXres))
    for iq in range(qres):
        out[iq] = (np.cos(q[iq]*r[0]*cosphimph)-np.cos(q[iq]*r[-1]*cosphimph))/cosphimph
        
    return out

### compute fourier transform of uij for fixed velocity beta (i.e. uij is an array of shape (3,3,len(theta),len(phi)))
@jit
def fourieruij(uij,r,phiX,q,ph,sincos=None):
    '''Takes uij multiplied by r over the Burgers vector (which then is only phi dependent), and returns uijs Fourier transform multiplied q which then only depends on q through the cutoffs q*r[0] and q*r[-1].
       r, phi are the magnitude and angle in x-space, although of the former only the cutoffs r[0] and r[-1] are used since the integration over r is done analytically. q, ph are the magnitude and angle in Fourier space.
       If sincos=None, sincos is computed from fourieruij_sincos(). Optionally, however, it may be given as input explicitly.'''
    phres = len(ph)
    qres = len(q)
    phiXres = len(phiX)
    Ntheta = len(uij[0,0])
    ### computing just the non-vanishing purely imaginary part allows us to do everything with real numbers which is faster
    result = np.zeros((3,3,Ntheta,qres,phres))
    ph_ones = np.ones((phres))
    uij_array = np.zeros((3,3,phres*phiXres))
    if np.asarray(sincos).all() is None:
        sincos = fourieruij_sincos(r,phiX,q,ph)
    
    for th in range(Ntheta):
        integrand = np.zeros((3,3,qres,phres*phiXres))
        for i in range(3):
            for j in range(3):
                uij_array[i,j] = np.reshape(np.outer(ph_ones,uij[i,j,th]),(phres*phiXres))
                for iq in range(qres):
                    np.multiply(uij_array[i,j] , sincos[iq] , integrand[i,j,iq])
    
        result[:,:,th] = np.trapz(np.reshape(integrand,(3,3,qres,phres,phiXres)),x=phiX)
                
    return result

@jit
def fourieruij_nocut(uij,phiX,ph,regul=500,sincos=None):
    '''Takes uij multiplied by r over the Burgers vector (which then is only phi dependent), and returns uijs Fourier transform multiplied q which then only depends on q through the cutoffs q*r[0] and q*r[-1].
       This function, however, assumes these cutoffs are removed, i.e. taken to 0 and infinity, respectively. Hence, the result is q-independent and as such a smaller array of shape (3,3,len(theta),len(ph)) is returned,
       where ph is the angle in Fourier space.
       The expression q*sin(r*q*cos(phiX-ph)) is analytically integrated over r using the given regulator 'regul' resulting in (1-cos(regul*cos(phiX-ph)))/cos(regul*cos(phiX-ph)).
       regul is necessary in order to avoid division by (close to) zero from 1/cos(phiX-ph).
       Alternatively, the user may give an alternative form of this input via the 'sincos' variable, such as the output of fourieruij_sincos() averaged over q (assuming the cutoffs in r were chosen such that any q-dependence became neglegible).'''
    phres = len(ph)
    phiXres = len(phiX)
    ph2res = phres*phiXres
    Ntheta = len(uij[0,0])
    result = np.zeros((3,3,Ntheta,phres))
    ph_ones = np.ones((phres))
    uij_array = np.zeros((3,3,ph2res))
    if np.asarray(sincos).all() is None:
        cosphimph = np.reshape(np.cos(np.outer(np.ones((phres)),phiX)-np.outer(ph,np.ones((phiXres)))),(ph2res))
        sincos = (1-np.cos(regul*cosphimph))/cosphimph
    
    integrand = np.empty((3,3,ph2res))
    for th in range(Ntheta):
        for i in range(3):
            for j in range(3):
                uij_array[i,j] = np.reshape(np.outer(ph_ones,uij[i,j,th]),(ph2res))
                np.multiply(uij_array[i,j] , sincos , integrand[i,j])
    
        result[:,:,th] = np.trapz(np.reshape(integrand,(3,3,phres,phiXres)),x=phiX)
                
    return result


############# energy and line tension
if usefortran:
    def computeEtot(uij, betaj, C2, Cv, phi):
        '''Computes the self energy of a straight dislocation uij moving at velocity betaj.
        Additional required input parameters are the 2nd order elastic constant tensor C2 (for the strain energy) and its velocity dependent shift Cv (for the kinetic energy),
        as well as the integration angle phi inside the plane normal to the dislocation line.'''
        return fsub.computeetot(np.moveaxis(uij,-1,0),betaj,C2,Cv,phi)
else:
    @jit
    def computeEtot(uij, betaj, C2, Cv, phi):
        '''Computes the self energy of a straight dislocation uij moving at velocity betaj.
        Additional required input parameters are the 2nd order elastic constant tensor C2 (for the strain energy) and its velocity dependent shift Cv (for the kinetic energy),
        as well as the integration angle phi inside the plane normal to the dislocation line.'''
        Ntheta = len(uij[0,0,:,0])
        Nphi = len(phi)
        Wtot = np.zeros((Ntheta,Nphi))
    
        for th in range(Ntheta):
            for k in range(3):
                for l in range(3):
                    for o in range(3):
                        for p in range(3):
                            Wtot[th] += uij[l,k,th]*uij[o,p,th]*(C2[k,l,o,p] + betaj*betaj*Cv[k,l,o,p,th])
            
        return np.trapz(Wtot,x=phi)/2
    

def computeLT(Etot, dtheta):
    '''Computes the line tension prefactor of a straight dislocation by adding to its energy the second derivative of that energy w.r.t. the dislocation character theta.
    The latter needs to be discretized, i.e. Etot is an array in theta space. Additionally, the step size dtheta needs to be given as input.'''
    ddE = np.diff(Etot,2)/(dtheta*dtheta)
    return ddE + Etot[1:-1]
