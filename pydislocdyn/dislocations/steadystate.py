# Compute various properties of a moving dislocation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Sept. 22, 2025
'''This submodule contains a class, StrohGeometry, to calculate the displacement field of a steady state dislocation
   as well as various other properties. See also the more general Dislocation class defined in pydislocdyn.dislocations.general,
   which inherits from the StrohGeometry class defined here and the metal_props class defined in pydislocdyn.crystals.'''
#################################
import numpy as np
import sympy as sp
import pydislocdyn.dislocations.numba_subroutines as nsub
from ..utilities import usefortran, delta, rotaround, artan
if usefortran:
    from ..utilities import fsub
    fourieruij_sincos = fsub.fourieruij_sincos
    
    def fourieruij_nocut(uij,phiX,sincos,Ntheta,phres):
        '''Fourier transform of angular part of uij (needs result of subroutine fourieruij_sincos for sincos)'''
        return fsub.fourieruij_nocut(np.moveaxis(uij,-1,0),phiX,sincos,Ntheta,phres)
    
    def computeEtot(uij, betaj, C2, Cv, phi):
        '''Computes the self energy of a straight dislocation uij moving at velocity betaj.
        Additional required input parameters are the 2nd order elastic constant tensor C2 (for the strain energy) and its velocity dependent shift Cv (for the kinetic energy),
        as well as the integration angle phi inside the plane normal to the dislocation line.'''
        return fsub.computeetot(np.moveaxis(uij,-1,0),betaj,C2,Cv,phi)
    
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           All arguments are arrays, i.e. A and B have shape (3,Ntheta) where Ntheta is e.g. the number of character angles.'''
        return np.moveaxis(fsub.elbrak(np.moveaxis(A,-1,0),np.moveaxis(B,-1,0),elC),0,-1)
    
    def elbrak1d(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           This function is similar to elbrak(), but its arguments do not depend on the character angle, i.e. A, B have shape (3).'''
        return fsub.elbrak1d(A,B,elC)
    
    def computeuk(beta, C2, Cv, b, M, N, phi, r):
        '''Compute the dislocation displacement field uk according to the integral method.'''
        return np.moveaxis(fsub.computeuk(beta, C2, Cv, b, np.moveaxis(M,-1,0), np.moveaxis(N,-1,0), phi, r),0,-1)
        
    def computeuij(beta, C2, Cv, b, M, N, phi, r=None, debug=False):
        '''Compute the dislocation displacement gradient field uij according to the integral method. If vector r is omitted, return only the angular dependence.'''
        if debug:
            return nsub.computeuij(beta, C2, Cv, b, M, N, phi, r=r, debug=debug)
        out = np.moveaxis(fsub.computeuij(beta, C2, Cv, b, np.moveaxis(M,-1,0), np.moveaxis(N,-1,0), phi),0,-1)
        if r is not None:
            Ntheta = len(M[0,:,0])
            out = np.moveaxis(np.reshape(np.outer(1/r,out),(len(r),3,3,Ntheta,len(phi))),0,-2)
        return out
else:
    from pydislocdyn.dislocations.numba_subroutines import fourieruij_sincos, fourieruij_nocut, computeEtot, elbrak, elbrak1d, computeuij
    def computeuk(beta, C2, Cv, b, M, N, phi, r):
        '''Compute the dislocation displacement field uk according to the integral method. This wrapper will call computeuij() with keyword nogradient=True;
           see the docs of that function for further details.'''
        return computeuij(beta, C2, Cv, b, M, N, phi, r=r, nogradient=True)

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
        self.n0 = np.asarray(n0)
        self.t = np.zeros((Ntheta,3))
        self.m0 = np.zeros((Ntheta,3))
        self.Cv = np.zeros((3,3,3,3,Ntheta))
        self.M = np.zeros((3,Ntheta,Nphi))
        self.N = np.zeros((3,Ntheta,Nphi))
        self.Sb = self.Bb = self.NN = None ## only used for debugging

        self.beta = 0
        self.C2_aligned=None
        self.C2norm = np.zeros((3,3,3,3)) # normalized
        self.sym = None ## keyword defining crystal symmetry, unknown until C2norm is set
        self.uk = self.uk_aligned = None
        self.uij = self.uij_static = np.zeros((3,3,Ntheta,Nphi))
        self.uij_aligned = np.zeros((3,3,Ntheta,Nphi))
        self.uij_static_aligned = None
        self.uij_acc_screw_aligned = None
        self.uij_acc_edge_aligned = None
        self.rot = np.zeros((Ntheta,3,3))
        self.Etot = np.zeros((Ntheta))
        self.LT = 0
        
        bsq = np.dot(self.b,self.b)
        nsq = np.dot(self.n0,self.n0)
        if isinstance(bsq, sp.Expr) or isinstance(nsq, sp.Expr):
            self.b = np.array(sp.simplify(self.b/sp.sqrt(bsq)))
            self.n0 = np.array(sp.simplify(self.n0/sp.sqrt(nsq)))
            self.t = np.empty(self.t.shape,dtype=object)
            self.m0 = np.empty(self.t.shape,dtype=object)
            for i,th in enumerate(self.theta*sp.pi/np.pi):
                self.t[i] = sp.matrix2numpy(sp.simplify(sp.Matrix(self.b*sp.cos(th)) + sp.Matrix(self.b*sp.sin(th)).cross(sp.Matrix(self.n0)))).reshape((3))
                self.m0[i] = sp.matrix2numpy(sp.simplify(sp.Matrix(self.n0).cross(sp.Matrix(self.t[i])))).reshape((3))
            ## skip the rest: would need numbers, not sympy symbols
        else:
            if bsq>1e-12 and abs(bsq-1)>1e-12:
                self.b = self.b/np.sqrt(bsq)
            if nsq>1e-12 and abs(nsq-1)>1e-12:
                self.n0 = self.n0/np.sqrt(nsq)
            if np.abs(self.b @ self.n0)>1e-12:
                print(f"WARNING: the Burgers vector is not normal to the slip plane normal: \n{self.b @ self.n0=:.6f} != 0")
            
            self.t = np.outer(np.cos(self.theta),self.b) + np.outer(np.sin(self.theta),np.cross(self.b,self.n0))
            self.m0 = np.cross(self.n0,self.t)
            
            for i in range(3):
                self.M[i] = np.outer(self.m0[:,i],np.cos(self.phi)) + np.outer(np.repeat(self.n0[i],Ntheta),np.sin(self.phi))
                self.N[i] = np.outer(np.repeat(self.n0[i],Ntheta),np.cos(self.phi)) - np.outer(self.m0[:,i],np.sin(self.phi))
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            self.Cv[i,j,k,l] = self.m0[:,i]*delta[j,k]*self.m0[:,l]
        
    def __repr__(self):
        return f" b:\t {self.b}\n n0:\t {self.n0}\n beta:\t {self.beta}\n Ntheta:\t {self.Ntheta}"
        
    def findedgescrewindices(self,theta=None):
        '''Find the indices i where theta[i] is either 0 or +/-pi/2. If theta is omitted, assume theta=self.theta.'''
        if theta is None:
            theta=self.theta
        else: theta = np.asarray(theta)
        scrind = np.where(np.abs(theta)<1e-12)[0]
        out = [None,None]
        if len(scrind) == 1:
            out[0] = int(scrind)
        edgind = np.where(np.abs(theta-np.pi/2)<1e-12)[0]
        if len(edgind) == 1:
            out[1] = int(edgind)
        negedgind = np.where(np.abs(theta+np.pi/2)<1e-12)[0]
        if len(negedgind) == 1:
            out.append(int(negedgind))
        return out
        
    def computeuij(self, beta, C2=None, r=None, debug=False):
        '''Compute the dislocation displacement gradient field according to the integral method (which in turn is based on the Stroh method).
           This function returns a 3x3xNthetaxNphi dimensional array.
           Required input parameters are the dislocation velocity beta and the 2nd order elastic constant tensor C2.
           If r is provided, the full displacement gradient (a 3x3xNthetaxNrxNphi dimensional array) is returned.
           In the latter case, the core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
        self.beta = beta
        if C2 is None:
            C2 = self.C2norm
        else:
            self.C2norm = C2
        if r is not None:
            self.r = r
        self.uij = computeuij(beta, C2, self.Cv, self.b, self.M, self.N, self.phi, r=r, debug=debug)
        if abs(beta)<1e-15:
            self.uij_static = self.uij.copy()
        if debug:
            self.Sb = self.uij['S.b']
            self.Bb = self.uij['B.b']
            self.NN = self.uij['NN']
            self.uij = self.uij['uij']
        
    def computeuk(self, beta, C2=None, r=None):
        '''Compute the dislocation displacement field according to the integral method (which in turn is based on the Stroh method).
           This function returns a 3xNthetaxNrxNphi dimensional array.
           Required input parameters are the dislocation velocity beta and the 2nd order elastic constant tensor C2.
           The core cutoff is assumed to be the first element in array r, i.e. r0=r[0] (and hence r[0]=0 will give 1/0 errors).'''
        self.beta = beta
        if C2 is None:
            C2 = self.C2norm
        else:
            self.C2norm = C2
        if r is not None:
            self.r = r
        else:
            r = self.r
        if r is None:
            raise ValueError("I need an array for r.")
        self.uk = computeuk(beta, C2, self.Cv, self.b, self.M, self.N, self.phi, r)
        
    def computerot(self,y=[0,1,0],z=[0,0,1]):
        '''Computes a rotation matrix that will align slip plane normal n0 with unit vector y, and line sense t with unit vector z.
           y, and z are optional arguments whose default values are unit vectors pointing in the y and z direction, respectively.'''
        if self.n0.dtype == np.dtype('O') and y==[0,1,0] and z==[0,0,1]:
            self.rot = np.zeros((self.Ntheta,3,3),dtype='object')
            for th in range(self.Ntheta):
                self.rot[th] = np.array([np.cross(self.n0,self.t[th]),self.n0,self.t[th]])
            return
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

    def alignuk(self,accuracy=15):
        '''Rotates previously computed uk using rotation matrix rot (run computeuk and computerot methods first), and stores the result in attribute uk_aligned.'''
        n = self.uk.shape
        ukrotated = np.zeros(n)
        for th in range(len(self.theta)):
            for ri in range(n[2]):
                ukrotated[:,th,ri] = np.round(np.dot(self.rot[th],self.uk[:,th,ri]),accuracy)
        self.uk_aligned = ukrotated
        
    def computeEtot(self):
        '''Computes the self energy of a straight dislocation uij moving at velocity beta. (Requirement: run method .computeuij(beta,C2) first.)'''
        self.Etot = computeEtot(self.uij, self.beta, self.C2norm, self.Cv, self.phi)
        
    def computeLT(self):
        '''Computes the line tension prefactor of a straight dislocation by adding to its energy the second derivative of that energy w.r.t.
        the dislocation character theta. (Requirements: run methods computeuij(beta,C2) and computeEtot() first.)'''
        dtheta = abs(self.theta[1]-self.theta[0])
        self.LT = computeLT(self.Etot, dtheta)

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

def fourieruij_iso(beta,ct_over_cl, theta, phi):
    '''Compute the dislocation displacement gradient field in the isotropic limit in Fourier space multiplied by iq/b (the radial coordinate over the magnitude of the Burgers vector),
       i.e. we only return the dependence on the (discretized) polar angle phi in Fourier space, and hence the result is a 3x3xNthetaxNphi dimensional array.
       Required input parameters are: the dislocation velocity beta in units of transverse sound speed, the ratio of transverse to longitudinal sound speed,
       and two arrays encoding the discretized dependence on the angle theta between dislocation line and Burgers vector and the polar angle phi.'''
    gamt = 1-beta**2 ## defined as 1/gamma**2
    gaml = 1-(ct_over_cl*beta)**2
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

def computeLT(Etot, dtheta):
    '''Computes the line tension prefactor of a straight dislocation by adding to its energy the second derivative of that energy w.r.t. the dislocation character theta.
    The latter needs to be discretized, i.e. Etot is an array in theta space. Additionally, the step size dtheta needs to be given as input.'''
    ddE = np.diff(Etot,2)/(dtheta*dtheta)
    return ddE + Etot[1:-1]
