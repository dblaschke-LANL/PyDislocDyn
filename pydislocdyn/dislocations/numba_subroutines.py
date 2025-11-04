# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Sept. 15, 2025
'''This submodule contains various subroutines that are accelerated using just-in-time compiler numba.
   For the Fortran-implementation of these subroutines, see subroutines.f90.'''

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from ..utilities import nonumba, jit

if nonumba:
    trapz = trapezoid
    cumtrapz = cumulative_trapezoid
else:
    @jit(nopython=True)
    def trapz(y,x):
        '''integrate over the last axis using the trapezoidal rule (i.e. equivalent to numpy.trapz(y,x,axis=-1))'''
        theshape = y.shape
        n = theshape[-1]
        f = y.T
        outar = np.zeros(theshape).T
        for i in range(n-1):
            outar[i] = (0.5*(f[i+1]+f[i])*(x[i+1]-x[i]))
        return np.sum(outar.T,axis=-1)
    
    @jit(nopython=True)
    def cumtrapz(y,x,initial=0):
        '''Cumulatively integrate over the last axis using the trapezoidal rule (i.e. equivalent to scipy.integrate.cumtrapz(y,x,axis=-1,initial=0),
           but faster due to the use of the numba.jit compiler).'''
        theshape = y.shape
        n = theshape[-1]
        f = y.T
        outar = np.zeros(theshape).T
        tmp = np.zeros(theshape[:-1]).T
        for i in range(n-1):
            tmp += (0.5*(f[i+1]+f[i])*(x[i+1]-x[i]))
            outar[i+1] = tmp
        return outar.T

@jit(nopython=True)
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
                        np.add(AB[l,o,th], np.multiply(np.multiply(A[k,th], elC[k,l,o,p,th],tmp), B[p,th],tmp), AB[l,o,th])
    
    return AB

@jit(nopython=True)
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

@jit(forceobj=True)  ## calls preventing nopython mode: np.dot with arrays >2D, np.moveaxis(), np.linalg.inv with 3-D array arguments, and raise ValueError / debug option
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
                    np.subtract(S[k,l,th], np.multiply(NNinv[k,o,th], NM[o,l,th], tmp), S[k,l,th])
        
        for k in range(3):
            for l in range(3):
                np.add(B[k,l,th], MM[k,l,th], B[k,l,th])
                for o in range(3):
                    np.add(B[k,l,th], np.multiply(MN[k,o,th], S[o,l,th], tmp), B[k,l,th])
                    
    Sb = (1/(4*pi*pi))*np.dot(b,trapz(S,x=phi))
    B = (1/(4*pi*pi))*np.dot(b,trapz(B,x=phi))
    
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

@jit(nopython=True)
def fourieruij_sincos(ra,rb,phiX,q,ph):
    '''Subroutine for fourieruij() that computes q*sin(r*q*cos(phiX-ph)) integrated over r, resulting in an array of shape (len(q),len(phiX)*len(ph)).
       The latter is then averaged over q (assuming the cutoffs in r were chosen such that any q-dependence became negligible).
       All input parameters must be 1-dim. arrays, except for the cutoffs ra and rb (the integration over r is done analytically, so only need the endpoints).'''
    phres = len(ph)
    qres = len(q)
    phiXres = len(phiX)
    cosphimph = np.reshape(np.cos(np.outer(np.ones((phres)),phiX)-np.outer(ph,np.ones((phiXres)))),(phres*phiXres))
    out = np.zeros((phres*phiXres))
    for iq in range(qres):
        out += (np.cos(q[iq]*ra*cosphimph)-np.cos(q[iq]*rb*cosphimph))/cosphimph/qres
   
    return out

@jit(nopython=True)
def fourieruij_nocut(uij,phiX,sincos,Ntheta,phres):
    '''Takes uij multiplied by r over the Burgers vector (which then is only phi dependent), and returns uijs Fourier transform multiplied q which then only depends on q through the cutoffs q*r[0] and q*r[-1].
       This function, however, assumes these cutoffs are removed, i.e. taken to 0 and infinity, respectively. Hence, the result is q-independent and as such a smaller array of shape (3,3,len(theta),len(ph)) is returned,
       where ph is the angle in Fourier space.
       The expression q*sin(r*q*cos(phiX-ph)) analytically integrated over r needs to be passed via the 'sincos' input variable (see fourieruij_sincos()).'''
    phiXres = len(phiX)
    ph2res = phres*phiXres
    result = np.zeros((3,3,Ntheta,phres))
    ph_ones = np.ones((phres))
    uij_array = np.zeros((3,3,ph2res))
    integrand = np.empty((3,3,ph2res))
    for th in range(Ntheta):
        for i in range(3):
            for j in range(3):
                uij_array[i,j] = np.reshape(np.outer(ph_ones,uij[i,j,th]),(ph2res))
                np.multiply(uij_array[i,j], sincos, integrand[i,j])
    
        result[:,:,th] = trapz(np.reshape(integrand,(3,3,phres,phiXres)),x=phiX)
                
    return result

@jit(nopython=True)
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
        
    return trapz(Wtot,x=phi)/2

@jit(nopython=True)
def accscrew_xyintegrand(x,y,t,xpr,a,B,C,Ct,ABC,cA,xcomp):
    '''subroutine of computeuij_acc_screw'''
    Rpr = np.sqrt((x-xpr)**2 - (x-xpr)*y*B/C + y**2/Ct)
    eta = np.sqrt(2*xpr/a)
    etatilde = np.sign(x)*np.sqrt(2*abs(x)/a)*0.5*(1+xpr/x)
    tau = t - eta
    tau_min_R = np.sqrt(abs(tau**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct = (np.sign((t - eta - Rpr/(cA*np.sqrt(ABC))))+1)/2
    tau2 = t - etatilde
    tau_min_R2 = np.sqrt(abs(tau2**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct2 = (np.sign((t - etatilde - Rpr/(cA*np.sqrt(ABC))))+1)/2
    if xcomp:
        integrand = stepfct*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R + tau**2*(ABC/Ct)/tau_min_R)
        integrand -= stepfct2*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R2 + tau2**2*(ABC/Ct)/tau_min_R2) ## subtract pole
    else:
        integrand = stepfct*(1/Rpr**4)*((tau**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R - (x-xpr)**2*(tau_min_R))
        integrand -= stepfct2*(1/Rpr**4)*((tau2**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R2 - (x-xpr)**2*tau_min_R2)
    return integrand
