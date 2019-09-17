# Compute the line tension of a moving dislocation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Sep. 17, 2019
#################################
from __future__ import division
from __future__ import print_function

# from sys import version_info
### make sure we are running a recent version of python
# assert version_info >= (3,5)
import numpy as np
from scipy.integrate import cumtrapz
nonumba=False
try:
    from numba import jit
except ImportError:
    print("WARNING: cannot find just-in-time compiler 'numba', execution will be slower\n")
    nonumba=True
    def jit(func):
        return func

### define the Kronecker delta
delta = np.diag((1,1,1))

def rotateinto(a,b):
    '''Computes the rotation matrix that rotates vector 'a' into vector 'b'.
       Except: if 'b' is a float, this function returns the rotation matrix for a rotation of angle 'b' around axis 'a'.'''
    old = np.asarray(a)
    na = np.sqrt(np.sum(old**2))
    if na>0. and abs(na-1)>1e-12:
        old = old/na
    if isinstance(b,float):
        v=-old
        sv = np.sin(b)
        cv = np.cos(b)
    else:
        new = np.asarray(b)
        nb = np.sqrt(np.sum(new**2))
        if nb>0. and abs(nb-1)>1e-12:
            new = new/nb
        v = np.cross(old,new)
        sv = np.sqrt(np.sum(v**2))
        cv = np.dot(old,new)
    vx = np.zeros((3,3))
    vx[0,1] = -v[2]
    vx[1,0] = v[2]
    vx[0,2] = v[1]
    vx[2,0] = -v[1]
    vx[1,2] = -v[0]
    vx[2,1] = v[0]
    if isinstance(b,float):
        out = delta +sv*vx + np.dot(vx,vx)*(1-cv)
    elif abs(sv)<1e-2:
        out = delta + vx + np.dot(vx,vx)/(1+cv)
    else:
        out = delta + vx + np.dot(vx,vx)*(1-cv)/sv**2
    return out

class StrohGeometry(object):
    '''This class computes several arrays to be used in the computation of a dislocation displacement gradient field for crystals using the integral version of the Stroh method.
       Required input parameters are: the Burgers vector b, the slip plane normal n0, an array theta parametrizing the angle between dislocation line and Burgers vector, 
       and the resolution Nphi of angles to be integrated over.
       Its initial attributes are: the velocity dependent shift Cv (to be added or subtracted from the tensor of 2nd order elastic constants) and
       the vectors M(theta,phi) and N(theta,phi) parametrizing the plane normal to the dislocation line, as well as the dislocation line direction t and unit vector m0 normal to n0 and t.
       Methods computeuij(), computeEtot(), and computeLT() call functions of the same name, storing results in attributes uij, Etot, and LT.
       Method computerot finally computes a rotation matrix that will align n0,t with Cartesian y,z directions.'''
    def __init__(self,b, n0, theta, Nphi):
        Ntheta = len(theta)
        self.theta = theta
        self.phi = np.linspace(0,2*np.pi,Nphi)
        self.b = b
        self.n0=n0
        self.t = np.zeros((Ntheta,3))
        self.m0 = np.zeros((Ntheta,3))
        self.Cv = np.zeros((3,3,3,3,Ntheta))
        self.M = np.zeros((3,Ntheta,Nphi))
        self.N = np.zeros((3,Ntheta,Nphi))
        
        self.t = np.outer(np.cos(theta),b) + np.outer(np.sin(theta),np.cross(b,n0))
        self.m0 = np.cross(n0,self.t)
        
        for i in range(3):
            self.M[i] = np.outer(self.m0[:,i],np.cos(self.phi)) + np.outer(np.repeat(n0[i],Ntheta),np.sin(self.phi))
            self.N[i] = np.outer(np.repeat(n0[i],Ntheta),np.cos(self.phi)) - np.outer(self.m0[:,i],np.sin(self.phi))
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        self.Cv[i,j,k,l] = self.m0[:,i]*delta[j,k]*self.m0[:,l]
        
        self.beta = 0
        self.C2 = np.zeros((3,3,3,3))
        self.uij = np.zeros((3,3,Ntheta,Nphi))
        self.uij_aligned = np.zeros((3,3,Ntheta,Nphi))
        self.rot = np.zeros((Ntheta,3,3))
        self.Etot = np.zeros((Ntheta))
        self.LT = 0
        
    def computeuij(self, beta, C2, r=None, nogradient=False):
        '''Compute the dislocation displacement gradient field according to the integral method (which in turn is based on the Stroh method).
           This function returns a 3x3xNthetaxNphi dimensional array.
           Required input parameters are the dislocation velocity beta and the 2nd order elastic constant tensor C2.
           If r is provided, the full displacement gradient (a 3x3xNthetaxNrxNphi dimensional array) is returned.
           If option nogradient is set to True, the displacement field (not its gradient) is returned: a 3xNthetaxNrxNphi dimensional array.
           In the latter two cases, the core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
        self.beta = beta
        self.C2 = C2
        self.uij = computeuij(beta, C2, self.Cv, self.b, self.M, self.N, self.phi, r=None, nogradient=False)
        
    def computerot(self,y = [0,1,0],z = [0,0,1]):
        '''Computes a rotation matrix that will align slip plane normal n0 with unit vector y, and line sense t with unit vector z.
           y, and z are optional arguments whose default values are unit vectors pointing in the y and z direction, respectively.)'''
        pi = np.pi
        if round(np.dot(self.n0,y),15)==-1:
            rot1 = rotateinto(z,pi)
        else:
            rot1 = rotateinto(self.n0,y)
        Ntheta = len(self.theta)
        rot = np.zeros((Ntheta,3,3))
        for th in range(Ntheta):
            newt = np.dot(rot1,self.t[th])
            if round(np.dot(newt,z),15)==-1:
                rot[th] = np.dot(rotateinto(y,pi),rot1)
            else:
                rot[th] = np.dot(rotateinto(newt,z),rot1)
        self.rot = rot
    
    def alignuij(self,accuracy=15):
        '''Rotates previously computed uij using rotation matrix rot (run computeuij and computerot methods first), and stores the result in attribute uij_aligned.'''
        n = self.uij.shape
        uijrotated = np.zeros(n)
        if len(n)==4:
            for th in range(len(self.theta)):
                uijrotated[:,:,th] = np.round(np.dot(self.rot[th],np.dot(self.rot[th],self.uij[:,:,th])),15)
        else:
            for th in range(len(self.theta)):
                for ri in range(n[3]):
                    uijrotated[:,:,th,ri] = np.round(np.dot(self.rot[th],np.dot(self.rot[th],self.uij[:,:,th,ri])),15)
        self.uij_aligned = uijrotated
        
    def computeEtot(self):
        '''Computes the self energy of a straight dislocation uij moving at velocity beta. (Requirement: run method .computeuij(beta,C2) first.)'''
        self.Etot = computeEtot(self.uij, self.beta, self.C2, self.Cv, self.phi)
        
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
    
try:
    import subroutines as fsub
    usefortran = True
except ImportError:
    print("WARNING: module 'subroutines' not found, execution will be slower")
    print("run 'f2py -c subroutines.f90 -m subroutines' to compile this module\n")
    usefortran = False
## numba version of elbrak() is just as fast as the fortran implementation, so we use the latter only as a fall-back if numba is missing
if nonumba and usefortran:
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors'''
        return np.moveaxis(fsub.elbrak(np.moveaxis(A,-1,0),np.moveaxis(B,-1,0),elC),0,-1)
else:
    @jit
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors'''
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
                        # AB[th] += A[k,l,th]*B[o,p,th]*elC[k,l,o,p]
                        #### faster numba-jit code is generated if we write the above like this (equivalent in pure python):
                        np.add(AB[th] , np.multiply(np.multiply(A[k,l,th],elC[k,l,o,p],tmp),B[o,p,th],tmp) , AB[th])
        
    return AB

@jit
def computeuij(beta, C2, Cv, b, M, N, phi, r=None, nogradient=False):
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
                    
    Sb = (1/(4*pi*pi))*np.tensordot(np.trapz(S,x=phi),b,axes = ([1],[0]))
    B = (1/(4*pi*pi))*np.tensordot(np.trapz(B,x=phi),b,axes = ([1],[0]))
    
    if nogradient==True:
        if Nr==0:
            raise ValueError("I need an array for r in conjunction with nogradient=True.")
        r0 = r[0] ## cutoff
        uiphi = np.zeros((3,Ntheta,Nphi))
        
        tmpu = np.zeros((3,Ntheta,Nphi))
        for th in range(Ntheta):
            # tmpu = np.zeros((3,Nphi))
            for j in range(3):
                for p in range(3):
                    tmpu[j,th] += (NNinv[j,p,th]*B[p,th] - S[j,p,th]*Sb[p,th])
        uiphi = cumtrapz(tmpu,x=phi,initial=0)
        # for ph in range(Nphi):
        #     uiphi[:,:,ph] = np.trapz(tmpu[:,:,:ph],x=phi[:ph])
        
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
 
    return uij


if np.version.version<'1.13':
    ## work around missing np.heaviside() in older versions of numpy
    @jit
    def artan(x,y):
        '''returns a variation of np.arctan2(x,y): since numpys implementation jumps to negative values in 3rd and 4th quadrant, shift those by 2pi so that atan(tan(phi))=phi for phi=[0,2pi]'''
        out = np.arctan2(x,y)
        sgn = np.sign(-out)
        out += 2*np.pi*sgn*(sgn+1)/2
        return out
else:
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
    if nogradient==True:
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
    if Nr != 0 and nogradient!=True:
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
    # rprime = np.outer(q,r)
    cosphimph = np.reshape(np.cos(np.outer(np.ones((phres)),phiX)-np.outer(ph,np.ones((phiXres)))),(phres*phiXres))
    out = np.zeros((qres,phres*phiXres))
    for iq in range(qres):
        # out[iq] = np.trapz(np.sin(np.outer(rprime[iq],cosphimph)),x=rprime[iq], axis=0)
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
    # result = np.zeros((3,3,Ntheta,qres,phres),dtype=complex)
    result = np.zeros((3,3,Ntheta,qres,phres))
    ph_ones = np.ones((phres))
    uij_array = np.zeros((3,3,phres*phiXres))
    if np.asarray(sincos).all()==None:
        sincos = fourieruij_sincos(r,phiX,q,ph)
    
    for th in range(Ntheta):
        integrand = np.zeros((3,3,qres,phres*phiXres))
        for i in range(3):
            for j in range(3):
                uij_array[i,j] = np.reshape(np.outer(ph_ones,uij[i,j,th]),(phres*phiXres))
                for iq in range(qres):
                    # integrand[i,j,iq] = uij_array[i,j]*sincos[iq]
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
    if np.asarray(sincos).all()==None:
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
    return (ddE + Etot[1:-1])

