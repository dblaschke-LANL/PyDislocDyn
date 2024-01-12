# Compute the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Jan. 11, 2024
'''This module implements the calculation of a dislocation drag coefficient from phonon wind.
   Its three front-end functions are :
       elasticA3 ...... computes the coefficient A3 from the SOECs and TOECs
       dragcoeff_iso ....... computes the drag coefficient assuming an isotropic phonon spectrum.
       phonondrag ........ a high-level wrapper around dragcoeff_iso that takes an instance of the
                           Dislocation class (defined in linetension_calcs.py) as its first argument.
                           Users most likely will want to use this function instead of dragcoeff_iso().
   All other functions are subroutines of the latter.'''
#################################
import numpy as np
import pandas as pd
from elasticconstants import UnVoigt
from dislocations import fourieruij_sincos, fourieruij_nocut, fourieruij_iso
from linetension_calcs import Dislocation, Ncores
if Ncores>1:
    from joblib import Parallel, delayed
try:
    from numba import jit
except ImportError:
    from functools import partial
    def jit(func=None,forceobj=True,nopython=False):
        '''define a dummy decorator if numba is unavailable at runtime'''
        if func is None:
            return partial(jit, forceobj=forceobj,nopython=nopython)
        return func
try:
    import subroutines as fsub
    assert(fsub.version()>=20210303),"the subroutines module is outdated, please re-compile with f2py" ## make sure the compiled subroutines module is up to date
    usefortran = True
except ImportError:
    usefortran = False

delta = np.diag((1,1,1))
hbar = 1.0545718e-34
kB = 1.38064852e-23

@jit(nopython=True)
def phonon(T,omega,q):
    '''Compute a phonon distribution function. Requires input: temperature T in Kelvin, omega*q denotes the phonon energy over hbar.
       A scale omega must be given separately, so that q is roughly of order 1. Otherwise 'overflow encountered in exp' errors might occur.'''
    scale = hbar*omega/(kB*T)
    return 1/(np.exp(scale*q)-1)

@jit(nopython=True)
def elasticA3(C2, C3):
    '''Returns the tensor of elastic constants as it enters the interaction of dislocations with phonons. Required inputs are the tensors of SOEC and TOEC.'''
    A3 = C3.copy()
    for i in range(3):
        for ii in range(3):
            for j in range(3):
                for jj in range(3):
                    for k in range(3):
                        for kk in range(3):
                            A3[i,ii,j,jj,k,kk] += C2[i,ii,jj,kk]*delta[j,k] + C2[j,jj,ii,kk]*delta[i,k] + C2[ii,jj,k,kk]*delta[i,j]
    return A3

### this fct. needs dij for a fixed angle theta (no array in theta space)!
@jit(nopython=True)
def dragcoeff_iso_Bintegrand(prefactor,dij,poly):
    '''Subroutine of dragcoeff_iso().'''
    ## prefactor has shape (len(t),len(phi))
    result1 = np.zeros(prefactor.shape)
    prefacOnes = np.ones((len(prefactor)))
    for k in range(3):
        for kk in range(3):
            for n in range(3):
                for nn in range(3):
                    result1 -= np.outer(prefacOnes,dij[k,kk]*dij[n,nn])*poly[k,kk,n,nn]
                    
    return prefactor*result1

## dragcoeff_iso_computepoly() is currently the bottle neck, as it takes of the order of a few seconds to compute and is needed for every velocity, temperature and (in the anisotropic case) every dislocation character theta
## jit-compiling some of the subroutines helps somewhat
@jit(nopython=True)
def dragcoeff_iso_computepoly_A3qt2(qt,qtshift,A3,lentph):
    '''Subroutine of dragcoeff_iso_computepoly().'''
    A3qt2 = np.zeros((3,3,3,3,lentph))
    tmp = np.zeros((lentph))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for kk in range(3):
                    for ii in range(3):
                        for jj in range(3):
                            np.add(A3qt2[i,j,k,kk] , np.multiply(qt[ii] , np.multiply(qtshift[jj] , A3[i,ii,j,jj,k,kk] , tmp) , tmp) , A3qt2[i,j,k,kk])
                            
    return A3qt2

@jit(nopython=True)
def dragcoeff_iso_computepoly_part1(qt,delta1,A3qt2,lentph):
    '''Subroutine of dragcoeff_iso_computepoly().'''
    part1 = np.zeros((3,3,3,3,lentph))
    tmp = np.zeros((lentph))
    for k in range(3):
        for kk in range(3):
            for l in range(3):
                for j in range(3):
                    for i in range(3):
                        np.add(part1[l,j,k,kk] , np.multiply(np.subtract(delta1[l,i] , np.multiply(qt[l] , qt[i] , tmp) , tmp) , A3qt2[i,j,k,kk] , tmp) , part1[l,j,k,kk])
                            
    return part1

@jit(nopython=True)
def dragcoeff_iso_computepoly_part2(qtshift,delta2,mag,A3qt2,dphi1,lentph):
    '''Subroutine of dragcoeff_iso_computepoly().'''
    part2 = np.zeros((3,3,3,3,lentph))
    tmp = np.zeros((lentph))
    for n in range(3):
        for nn in range(3):
            for l in range(3):
                for j in range(3):
                    for m in range(3):
                        np.add(part2[l,j,n,nn] , np.multiply(np.subtract(delta2[j,m] , np.divide(np.multiply(qtshift[j] , qtshift[m] , tmp) , mag , tmp) , tmp) , A3qt2[l,m,n,nn] , tmp) , part2[l,j,n,nn])
    np.multiply(part2, dphi1, part2)
    return part2
    
@jit(nopython=True)
def dragcoeff_iso_computepoly_foldpart12(result_previous,part1,part2,lentph):
    '''Subroutine of dragcoeff_iso_computepoly().'''
    result = result_previous
    tmp = np.zeros((lentph))
    for k in range(3):
        for kk in range(3):
            for n in range(3):
                for nn in range(3):
                    for l in range(3):
                        for j in range(3):
                            np.add(result[k,kk,n,nn] , np.multiply(part1[l,j,k,kk] , part2[l,j,n,nn] , tmp) , result[k,kk,n,nn])
                            
    return result

def dragcoeff_iso_computepoly(A3, phi, qvec, qtilde, t, phi1, longitudinal=False):
    '''Subroutine of dragcoeff_iso(). Flag "longitudinal" may be "False" for purely transverse, or "True" for purely longitudinal or an integer 1 or 2 telling us which of the two phonons is longitudinal.
    If the latter mixed modes are considered, variable qtilde is the ratio of q/q1, and variable t is a function of q and other variables, and that is what needs to be passed to this function.'''
    lenph = len(phi)
    lenph1 = len(phi1)
    lent = len(qtilde) ## qtilde is a either Nt x Nphi dimensional array or just an Nt dimensional one (for the mixed phonon cases)
    lentph = lent*lenph
    dphi1 = phi1[1:] - phi1[:-1]
    result = np.zeros((3,3,3,3,lentph))
    ## initialize for jit
    A3qt2 = np.empty((3,3,3,3,lentph))
    part1 = np.empty((3,3,3,3,lentph))
    part2 = np.empty((3,3,3,3,lentph))
    ##
    delta1 = np.zeros((3,3))
    delta2 = np.zeros((3,3))
    qv = np.empty((3,lent,lenph))
    
    for i in range(3):
        if longitudinal is False:
            delta1[i,i] = 1
            delta2[i,i] = delta1[i,i]
        elif longitudinal=="1":
            delta2[i,i] = 1
        elif longitudinal=="2":
            delta1[i,i] = 1
            
    if len(t.shape)==1:
        mag = np.reshape(np.outer(np.ones((lent)),np.ones((lenph)))+qtilde**2-2*np.outer(t,np.ones((lenph)))*qtilde,(lentph))
        for i in range(3):
            qv[i] = qtilde*np.outer(np.ones((lent)),qvec[i])
        # pieces for qt:
        tcosphi = np.reshape(np.outer(t,np.cos(phi)),(lentph))
        sqrtsinphi = np.reshape(np.outer(np.sqrt(1-t**2),np.sin(phi)),(lentph))
        tsinphi = np.reshape(np.outer(t,np.sin(phi)),(lentph))
        sqrtcosphi = np.reshape(np.outer(np.sqrt(1-t**2),np.cos(phi)),(lentph))
        sqrtt = np.reshape(np.outer(np.sqrt(1-t**2),np.ones((lenph))),(lentph))
    else:
        mag = np.reshape(np.outer(np.ones((lent)),np.ones((lenph)))+np.outer(qtilde**2,np.ones((lenph)))-2*t*np.outer(qtilde,np.ones((lenph))),(lentph))
        for i in range(3):
            qv[i] = np.outer(qtilde,qvec[i])
        # pieces for qt:
        tcosphi = np.reshape(t*np.outer(np.ones((lent)),np.cos(phi)),(lentph))
        sqrtt = np.sqrt(abs(1-t**2))
        sqrtsinphi = np.reshape(sqrtt*np.outer(np.ones((lent)),np.sin(phi)),(lentph))
        tsinphi = np.reshape(t*np.outer(np.ones((lent)),np.sin(phi)),(lentph))
        sqrtcosphi = np.reshape(sqrtt*np.outer(np.ones((lent)),np.cos(phi)),(lentph))
        sqrtt = np.reshape(sqrtt,(lentph))
    
    qv = np.reshape(qv,(3,lentph))

    if usefortran:
        ### use fortran implementation:
        result = np.moveaxis(fsub.parathesum(tcosphi,sqrtsinphi,tsinphi,sqrtcosphi,sqrtt,qv.T,delta1,delta2,mag,A3,phi1[:-1],dphi1,lenph,lent,lenph1-1,lentph),0,4)
        lenph1 = 1 ## ensure we bypass the python loop below

    ### else use jit-compiled python implementation:
    for p1 in range(lenph1-1):
        dp1 = dphi1[p1]
        qt = np.empty((3,lentph))
        qt[0] = tcosphi - sqrtsinphi*np.cos(phi1[p1])
        qt[1] = tsinphi + sqrtcosphi*np.cos(phi1[p1])
        qt[2] = sqrtt*np.sin(phi1[p1])
        qtshift = qt - qv
        
        A3qt2 = dragcoeff_iso_computepoly_A3qt2(qt,qtshift,A3,lentph)
        part1 = dragcoeff_iso_computepoly_part1(qt,delta1,A3qt2,lentph)
        part2 = dragcoeff_iso_computepoly_part2(qtshift,delta2,mag,A3qt2,dp1,lentph)
        result = dragcoeff_iso_computepoly_foldpart12(result,part1,part2,lentph)
                                
    if np.sum(delta1 - delta2)==0:
        out = np.reshape(result,(3,3,3,3,lent,lenph))
    else:
        out = np.reshape(-result,(3,3,3,3,lent,lenph)) ## for mixed modes we need an additional minus because we write (delta-qq) above and where delta=0 we would actually need +qq
    return out

if usefortran:
    dragcoeff_iso_phonondistri = fsub.dragcoeff_iso_phonondistri
else:
    @jit(nopython=True)
    def dragcoeff_iso_phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi):
        '''Subroutine of dragcoeff_iso().'''
        distri = np.zeros((lenq1,lent,lenphi))
        for i in range(lenq1):
        ### we have q1^6 but from d^2 we have 1/q^2, so that is q1^4/qtilde^2, and multiplied by qtildexcosphi
            distri[i] = prefac*(phonon(T,c1qBZ,q1[i]) - phonon(T,c2qBZ,q1[i]*OneMinBtqcosph1))*q1h4[i]
        return distri

## r0cut implements a soft core cutoff following Alshits 1979, i.e. multiplying the dislocation field by (1-exp(r/r0)) which leads to 1/sqrt(1-q**2/r0**2) in Fourier space
## warning: only correct for an isotropic dislocation field, and note that shape of the cutoff is beta-dependent the way it is introduced (circle only at beta=0)! (TODO: generalize)
def dragcoeff_iso_computeprefactor(qBZ, cs, beta_list, burgers, q1, phi, qtilde, T, r0cut=None):
    '''Subroutine of dragcoeff_iso().'''
    lent = len(qtilde) ## qtilde is a either Nt x Nphi dimensional array or just an Nt dimensional one (for the mixed phonon cases)
    lenq1 = len(q1)
    lenphi = len(phi)
    beta = beta_list[0]
    beta_L = beta_list[1]
    csphi = np.abs(np.cos(phi))
    ct_over_cl = 1 ## only need this ratio for outgoing longitudinal mode
    ### then determine what the prefactor should be in the mixed case and implement that.
    if isinstance(cs, list):
        c1 = cs[0]
        c2 = cs[1]
        if c1>c2:
            beta2=beta ## assume beta is beta_T
            ct_over_cl = c2/c1
            beta1=beta2*ct_over_cl
            c_T = c2
        elif c1==c2:
            if beta_L is False:
                ## purely transverse
                c_T = c1
                beta1=beta
                beta2=beta
            else:
                ## purely longitudinal
                c_T = c1*beta_L/beta
                beta1=beta_L
                beta2=beta_L
        else:
            beta1=beta ## assume beta is beta_T
            beta2=beta1*c1/c2
            c_T = c1
        c1qBZ = c1*qBZ
        c2qBZ = c1qBZ ## c2 and beta do not appear in phonon distributions since they were eliminated by the energy-cons delta fct. relating omega_2 to omega_1-Omega_q:
        q1h4 = (qBZ*q1)**4 ## one power less due to not substituting q->t; but instead have new Jacobian from q -> qtilde=q/q1 leading to same power as below
        prefac = (1000*np.pi*hbar*qBZ*burgers**2*c_T**4/(4*beta1*c1**2*c2**2*(2*np.pi)**5))*(np.outer(1/qtilde,csphi))
        ## take beta1 here, i.e. c2 and beta2 do not appear since they were eliminated by the energy-conserving delta fct. relating omega_2 to omega_1-Omega_q:
        OneMinBtqcosph1 = np.outer(np.ones((lent)),np.ones((lenphi)))-beta1*np.outer(qtilde,csphi)
    else:
        c1qBZ = cs*qBZ
        c2qBZ = c1qBZ
        q1h4 = (qBZ*q1)**4
        if beta_L is not False:
            ## purely longitudinal case:
            ct_over_cl = beta_L/beta
            beta = beta_L
        ### multiply by 1000 to get the result in mPas instead of Pas; also multiply by Burgers vector squared since we scaled that out in dij
        prefac = (1000*np.pi*hbar*qBZ*burgers**2*ct_over_cl**4/(2*beta*(2*np.pi)**5))*(np.outer(np.ones((lent)),csphi/(np.ones((lenphi))-(beta*csphi)**2))/qtilde)
        OneMinBtqcosph1 = np.outer(np.ones((lent)),np.ones((lenphi)))-beta*qtilde*np.outer(np.ones((lent)),csphi)
    if r0cut is None:
        distri = dragcoeff_iso_phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi)
    else:
        if isinstance(cs, list):
            cut =np.ones((lenq1,lent,lenphi)) + (qBZ*r0cut)**2*np.reshape(np.outer(np.outer(q1**2,qtilde**2),np.ones((lenphi))),(lenq1,lent,lenphi))
        else:
            cut =np.ones((lenq1,lent,lenphi)) + (qBZ*r0cut)**2*np.reshape(np.outer(q1**2,qtilde**2),(lenq1,lent,lenphi))
        if usefortran:
            cut = np.moveaxis(cut,0,2)
        distri = dragcoeff_iso_phonondistri(prefac,T,c1qBZ,c2qBZ,q1,q1h4,OneMinBtqcosph1,lenq1,lent,lenphi)/(cut)
    if usefortran:
        distri = np.moveaxis(distri,2,0)
    
    ### if c1>c2, we need to further limit the integration range of q1 <= (c2/c1)/(1-beta1*qtilde*abs(cosphi)) (in addition to q1 <=1);
    ### we do this by applying a mask to set all according array elements to zero in 'distri' before we integrate
    if isinstance(cs, list):
        if c1>c2:
            q1mask = np.reshape(np.outer(q1,np.ones((lent*lenphi))),((lenq1,lent,lenphi)))
            q1limit = ct_over_cl/OneMinBtqcosph1
            q1mask = (q1mask<=q1limit)
            distri = distri*q1mask
        ## this is either q1max, or for c1>c2 limited to lower value which might lie <dq1 below actual limit;
        ## may estimate the latter case by dq1/2 and add to last interval and weight the end point by 2/3 compared to its neighbor; this amounts to doubling the endpoint
        distri[-1] = 2*distri[-1]
    
    ## we cut off q1=0 to prevent divisions by zero, so compensate by doubling first interval, and weighting end point 3 to 1 compared to its neighbor
    # however, edge point might introduce rounding errors due to divergence so better just keep underestimating by small value (hence multiply end point by 2 instead of 3)
    distri[0] = 2*distri[0]
    return np.trapz(distri,x = q1, axis=0)
               
### rho x ct^2  = c44, and B is divided by rho^2*ct^4 = c44^2;
### it is therefore convenient to divide A3 by c44 as it enters quadratically, and this is a requirement below, i.e. A3 must be rescaled by c44 to be dimensionless!

def dragcoeff_iso(dij, A3, qBZ, ct, cl, beta, burgers, T, modes='all', Nt=321, Nq1=400, Nphi1=50, Debye_series=False, target_accuracy=5e-3, maxrec=6, accurate_to_digit=1e-5, Nchunks=20, skip_theta=None, skip_theta_val=np.inf, r0cut=None, name='drag'):
    '''Computes the drag coefficient from phonon wind for an isotropic crystal. Required inputs are the dislocation displacement gradient (times magnitude q and rescaled by the Burgers vector) dij in Fourier space
       (being a 3x3xNthetaxNphi array where theta is the angle parametrizing the dislocation type and phi is the polar angle in Fourier space), the array of shifted 3rd order elastic constants A3 in units of the shear modulus mu,
       the radius of the Brillouin zone qBZ, the transverse and longitudinal sound speeds ct and cl, the velocity beta in units of ct (i.e. beta=v/ct), the magnitude of the Burgers vectors burgers, as well as the temperature T.
       The keyword 'modes' determines which phonon contributions should be included in the computation.
       Allowed values are: 'all' (default), 'TT' (only transverse phonons), 'LL' (only longitudinal), 'mix' (only the two mixed modes), 'LT' (incoming transverse, outgoing longitudinal phonon), and 'TL' (incoming longitudinal outgoing transverse phonon).
       Optionally, the default values for the resolution of integration variables t, q1, and phi1 may be changed. Note that Nt is automatically increased with larger beta depending on the computed phonon mode.
       The parameter 'Debye_series' may be set to True in order to use the 4 terms of the series representation of the Debye functions instead of computing the Debye integral over the phonon spectrum numerically.
       Note, however, that the series representation converges only for high enough temperature.
       Optional variable skip_theta = None (default) may be set in order to bypass the calculation for certain angles theta and instead set those entries to some predefined value=skip_theta_val, i.e. skip_theta must be a boolean mask of len(theta) where False=bypass.'''
    
    Ntheta = len(dij[0,0])
    theta_ind = np.arange((Ntheta)) # generate array of theta-indices for later use
    modes_allowed = ['all', 'TT', 'LL', 'LT', 'TL', 'mix'] ## define allowed keywords for modes
    if beta <0 or beta>1:
        raise ValueError(f"{beta=}, but must be between 0 and 1.")
        
    if Debye_series and r0cut is not None:
        print("Warning: r0cut is set, therefore ignoring 'Debye_series=True'.")
        
    if np.asarray(Nchunks).any() is None:
        Nchks = 1
        Nt_total = None
    else:
        Nchks = Nchunks
        ## make sure Nt_total-1 is divisible by 2*Nchunks, and that Nt_current is odd and >=5 (i.e. increase user provided Nt as necessary)
        Nt_k = int(abs(Nt - 1 - 4*Nchunks)/(2*Nchunks))
        Nt_total = 4*Nchunks + 2*Nt_k*Nchunks + 1
        if Nt_total < Nt:
            Nt_total += 2*Nchunks ## ensure we never reduce resolution
    
    def adaptive_t_chunks(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, target_accuracy=target_accuracy, maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta = skip_theta, r0cut=r0cut, Nt_total=None, Nchunks=Nchunks, mode='??'):
        if Nchunks is None:
            out = adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, target_accuracy=target_accuracy, maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta = skip_theta, r0cut=r0cut, Nt=Nt, mode=mode)
        else:
            out = 0
            Nt_current = int((Nt_total-1)/Nchunks+1)
            for kth in range(Nchunks):
                out += adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, target_accuracy=target_accuracy, maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta = skip_theta, r0cut=r0cut, chunks=(Nchunks,kth), Nt=Nt_current, mode=mode)
        return out
    
    def adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, target_accuracy=target_accuracy, maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta = skip_theta, r0cut=r0cut, chunks=None, Nt=Nt, mode='??'):
        if np.asarray(skip_theta).any() is None:
            dijtmp = dij
            A3tmp = A3
        elif A3[0,0,0,0,0,0].shape == ():
            dijtmp = dij[:,:,skip_theta]
            A3tmp = A3
        else:
            dijtmp = dij[:,:,skip_theta]
            A3tmp = A3[skip_theta]
        args = (dijtmp, A3tmp, qBZ, cs, beta, burgers, T)
        Ntheta = len(dijtmp[0,0])
        theta_ind = np.arange((Ntheta))
        Ntauto_old = int(Nt/2)+1
        out_old = dragcoeff_iso_onemode(*args,Nt=Ntauto_old, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, r0cut=r0cut, chunks=chunks)
        Ntauto = Ntauto_old-1 ## number of points to add (i.e. total is 2*Ntauto_old-1)
        ## refine previous result
        out = out_old/2 + dragcoeff_iso_onemode(*args, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, update='t', r0cut=r0cut, chunks=chunks)/2
        ## determine if target accuracy was achieved
        err_mask = (out!=0)
        out_error_all = np.abs(np.abs(out)-np.abs(out_old))[err_mask]
        if len(out_error_all)==0:
            out_error = 0
        else:
            out_norm = (np.abs(out))[err_mask]
            out_error = min(np.max(out_error_all/out_norm),np.max(target_accuracy*out_error_all/(accurate_to_digit/Nchks)))
        refnmts = 0
        for rec in range(maxrec):
            if out_error < target_accuracy:
                break
            out_old = np.copy(out)
            Ntauto_old = Ntauto+Ntauto_old ## total number of points so far
            Ntauto = 2*Ntauto ## number of points to add
            if A3tmp[0,0,0,0,0,0].shape == ():
                out = out_old/2 + dragcoeff_iso_onemode(*args, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, update='t', r0cut=r0cut, chunks=chunks)/2
            else:
                ## refine only needed angles theta if A3 is theta dependent to save computation time
                theta_mask = (out_error_all/out_norm>=target_accuracy)
                theta_refine = (theta_ind[err_mask])[theta_mask]
                out_newpoints = dragcoeff_iso_onemode((dijtmp[:,:,err_mask])[:,:,theta_mask], (A3tmp[err_mask])[theta_mask], qBZ, cs, beta, burgers, T, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, update='t', r0cut=r0cut, chunks=chunks)
                for th in range(len(theta_refine)):
                    thi = theta_refine[th] # retrieve theta-index to use below
                    out[thi] = out_old[thi]/2 + out_newpoints[th]/2
            err_mask = (out!=0)
            out_error_all = np.abs(np.abs(out)-np.abs(out_old))[err_mask]
            if len(out_error_all)==0:
                out_error = 0
            else:
                out_norm = (np.abs(out))[err_mask]
                out_error = min(np.max(out_error_all/out_norm),np.max(target_accuracy*out_error_all/(accurate_to_digit/Nchks)))
            refnmts = rec+1
        if refnmts==maxrec and out_error >= target_accuracy and maxrec>0:
            print(f"warning: max # recursions reached ({name}), {beta=:.4f}, {T=}, {mode=}, {chunks=}, est. error={100*out_error:.2f}%")
        
        return out
    
    if np.asarray(skip_theta).all() is not None:
        BTT = np.zeros((Ntheta))[skip_theta]
        BLL = np.zeros((Ntheta))[skip_theta]
        BTL = np.zeros((Ntheta))[skip_theta]
        BLT = np.zeros((Ntheta))[skip_theta]
    else:
        BTT = np.zeros((Ntheta))
        BLL = np.zeros((Ntheta))
        BTL = np.zeros((Ntheta))
        BLT = np.zeros((Ntheta))
    
    if modes=='all' or modes=='TT':
        if maxrec is None: ## bypass adaptive grid if requested by user
            Ntauto = int((1+beta)*Nt) ## we know we need more points at higher beta, so already start from higher value
            BTT = dragcoeff_iso_onemode(dij=dij, A3=A3, qBZ=qBZ, cs=ct, beta=beta, burgers=burgers, T=T, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut)
        else:
            BTT = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=ct, beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='TT')
        
    if modes=='all' or modes=='LL':
        if maxrec is None:
            Ntauto = int((1+beta/2)*Nt)
            BLL = dragcoeff_iso_onemode(dij=dij, A3=A3, qBZ=qBZ, cs=cl, beta=beta, burgers=burgers, T=T, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta*ct/cl, r0cut=r0cut)
        else:
            BLL = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=cl, beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta*ct/cl, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='LL')
        
    if modes in ('all', 'mix', 'TL'):
        if maxrec is None:
            Ntauto = int((1+beta/2)*Nt)
            BTL = dragcoeff_iso_onemode(dij=dij, A3=A3, qBZ=qBZ, cs=[ct,cl], beta=beta, burgers=burgers, T=T, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut)
        else:
            BTL = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=[ct,cl], beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='TL')
        
    if modes in ('all', 'mix', 'LT'):
        if maxrec is None:
            Ntauto = int((1+beta)*Nt)
            BLT = dragcoeff_iso_onemode(dij=dij, A3=A3, qBZ=qBZ, cs=[cl,ct], beta=beta, burgers=burgers, T=T, Nt=Ntauto, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut)
        else:
            BLT = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=[cl,ct], beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='LT')
    
    if np.asarray(skip_theta).any() is None:
        out = BTT + BLL + BTL + BLT
    else:
        out = skip_theta_val*np.ones((Ntheta))
        theta_calcd = theta_ind[skip_theta]
        for th in range(len(theta_calcd)):
            thi = theta_calcd[th] # retrieve theta-index to use below
            out[thi] = BTT[th] + BLL[th] + BTL[th] + BLT[th]
    
    if modes not in modes_allowed:
        raise ValueError(f"Error: invalid keyword {modes=}.")
    
    return out

if usefortran:
    integratetphi = fsub.integratetphi
    integrateqtildephi = fsub.integrateqtildephi
else:
    def integratetphi(B,beta,t,phi,updatet,kthchk):
        '''Subroutine of dragcoeff_iso().'''
        limit = beta*np.abs(np.cos(phi))
        # qtlimit = 1/(beta*np.abs(np.cos(phi))) ## mask not needed for this, as it is always automatically fulfilled in the present coordinates and with the limit above
        Bt = np.zeros((len(phi)))
        for p in range(len(phi)):
            Btmp = B[:,p]
            # qtmask = (qtilde[:,p]<qtlimit[p])
            # t1 = t[qtmask]
            # Btmp = Btmp[qtmask]
            # tmask = (t1>limit[p])
            # t1 = t1[tmask]
            tmask = (t>limit[p])
            t1 = t[tmask]
            Btmp = Btmp[tmask]
            ## tmin is moved to higher value for most angles phi and t[0] (after mask) may be <dt higher than tmin;
            ## thus might be missing an interval 0<=dt0<=dt which we approximate by dt0~dt/2 and use same value as the neighboring interval;
            ## also, on updatet runs, we are missing dt/2 intervals on both ends of a chunk since we're computing intermediate points;
            ## to compensate, multiply the end-intervals by 1.5 and weight the end points by 2/3 compared to their neighbors;
            ## this amounts to doubling the endpoints and letting trapz take care of the rest
            if len(Btmp!=0):
                if updatet:
                    Btmp[0] = 2*Btmp[0]
                    Btmp[-1] = 2*Btmp[-1]
                elif kthchk==0:
                    Btmp[0] = 2*Btmp[0]
            Bt[p] = np.trapz(Btmp,x = t1)
        return np.trapz(Bt,x = phi)

    def integrateqtildephi(B,beta1,qtilde,t,phi,updatet,kthchk,Nchunks):
        '''Subroutine of dragcoeff_iso().'''
        Bt = np.zeros((len(phi)))
        ## energy conservation tells us w1-Wq>0, and hence qtilde<c1/v*cosphi=1/beta1*cosphi;
        qtlimit = 1/(beta1*np.abs(np.cos(phi)))
        for p in range(len(phi)):
            tmask = (abs(t[:,p])<1)
            qt = qtilde[tmask]
            Btmp = B[:,p]
            Btmp = Btmp[tmask]
            qtmask = (qt<qtlimit[p])
            qt = qt[qtmask]
            Btmp = Btmp[qtmask]
            ## endpoints Btmp[0], Btmp[-1] may have moved inside due to mask
            ## also: if we're refining, we are missing dt/2 intervals on both ends of a chunk on updatet runs since we're computing intermediate points;
            ## to compensate, multiply the end-intervals by 1.5 and weight the end points by 2/3 compared to their neighbors;
            ## this amounts to doubling the endpoints and letting trapz take care of the rest
            if len(Btmp)!=0:
                if updatet or kthchk==0:
                    Btmp[0] = 2*Btmp[0]
                if updatet or kthchk==(Nchunks-1):
                    Btmp[-1] = 2*Btmp[-1]
            Bt[p] = np.trapz(Btmp,x = qt)
        return np.trapz(Bt,x = phi)

def computeprefactorHighT(qBZ, cs, beta_list, burgers, phi, qtilde,T):
    '''Subroutine of dragcoeff_iso_onemode(): approximation in the high temperature limit.'''
    lent = len(qtilde) ## qtilde is a Nt x Nphi dimensional array
    beta = beta_list[0]
    beta_L = beta_list[1]
    csphi = np.abs(np.cos(phi))
    if beta_L is not False:
        ## purely longitudinal case:
        ct_over_cl = beta_L/beta
        beta = beta_L
    else:
        ct_over_cl=1
    OnesTwoDim = np.outer(np.ones((lent)),np.ones((len(phi))))
    ### multiply by 1000 to get the result in mPas instead of Pas; also multiply by Burgers vector squared since we scaled that out in dij
    CsPhi = np.outer(np.ones((lent)),csphi)
    qcosphi = np.outer(np.ones((lent)),csphi/(np.ones((len(phi)))-(beta*csphi)**2))/qtilde
    distri = np.zeros((lent,len(phi)))
    betaqtildeCsPhi = beta*qtilde*CsPhi
    hbarcsqBZ_TkB = hbar*cs*qBZ/(T*kB)
    ### we have q1^6 but from d^2 we have 1/q^2, so that is q1^4/qtilde^2, and multiplied by qtildexcosphi
    distri = 1000*np.pi*burgers**2*ct_over_cl**4/(2*beta*(2*np.pi)**5)*qcosphi*(T*kB*qBZ**4/(2*cs))*(-(beta/2)*qtilde*CsPhi/(OnesTwoDim-betaqtildeCsPhi)\
            +(hbarcsqBZ_TkB**2/36)*(betaqtildeCsPhi)\
            -(hbarcsqBZ_TkB**4/(30*4*24))*(1-(1-betaqtildeCsPhi)**3)\
            +(hbarcsqBZ_TkB**6/(42*5*720))*(1-(1-betaqtildeCsPhi)**5))
    
    return distri

def dragcoeff_iso_onemode(dij, A3, qBZ, cs, beta, burgers, T, Nt=500, Nq1=400, Nphi1=50, Debye_series=False, beta_long=False, update=None, chunks=None, r0cut=None):
    '''Subroutine of dragcoeff_iso(): Computes one of the four modes (TT, LL, TL, LT where T=transverse, L=longitudinal) contributing to the drag coefficient from phonon wind for an isotropic crystal.
       Required inputs are the dislocation displacement gradient (times magnitude q and rescaled by the Burgers vector) dij in Fourier space,
       being a 3x3xNthetaxNphi array where theta is the angle parametrizing the dislocation type and phi is the polar angle in Fourier space. Additionally, the array of shifted 3rd order elastic constants A3 in units of the shear modulus mu,
       the radius of the Brillouin zone qBZ, the transverse and/or longitudinal sound speed cs, the velocity beta in units of transverse sound speed, the magnitude of the Burgers vectors burgers, as well as the temperature T.
       If the optional keyword beta_long=False (default) and if only one sound speed (the transverse one) is passed via keyword cs, only the transverse phonon modes are considered in the computation of phonon wind.
       If velocity in units of longitudinal sound speed is passed via the keyword "beta_long", cs is assumed to be the longitudinal sound speed, and in this case the phonon wind from scattering purely longitudinal phonons is computed.
       If cs=[c1,c2] is a list of two sound speeds, the smaller one is assumed to be transverse, and in this case a different code path is employed (i.e. a different set of variables is used), which is slower but works also for the mixed modes.
       In this case, the keyword "beta_long" is ignored. beta is always assumed to be velocity over transverse sound speed.
       Optionally, the default values for the resolution of integration variables t, q1, and phi1 may be changed. The parameter 'Debye_series' may be set to True in order to use the 4 terms of the series representation of the Debye functions
       instead of computing the Debye integral over the phonon spectrum numerically. Note, however, that the series representation converges only for high enough temperature.'''
    
    ### chunks = np.array(total#ofchunks=Nchk, #ofcurrentchunk=ithchk)
    ### if set, Nt is number of points to use on current chunk
    ### with this information we divide array t (or qtilde) into Nchk subintervals,
    ### subintervals always end on exactly one point which is shared across neighboring chunks (we can then easily refine only certain chunks), i.e. initial Nt for total range must be divisible by #chunks,
    ### i.e. Nt_total = 1 + #chunks*(Nt-1) where Nt is number of points in each chunk of initial run (we're sharing boundary points!), hence Nt_initial = (Nt_total -1)/#chunks + 1 and Nt_total = (int((Nt_userchoice)/#chunks) + 1) * #chunks + 1 (so that Nt_total>=Nt_userchoice)
    ### Nt will always be 2^#rec (Nt_initial-1) and we only need Nt to calculate dt below once we divide the interval t into #chunks subintervals (that alone determines lower and upper limit, which then automatically matches points of the initial run)
    if np.asarray(chunks).any() is None:
        Nt_total = None
        kthchk = 0
        Nchunks = 1
    else:
        Nchunks, kthchk = chunks
        Nt_total = 1 + Nchunks*(Nt-1)
    
    updatet=bool(update=='t')
    ### initialize arrays
    Nphi = len(dij[0,0,0])
    phi = np.linspace(0,2*np.pi,Nphi)
    phi1 = np.linspace(0,2*np.pi,Nphi1)
    q1 = np.linspace(0,1,Nq1) ## need to rescale by qBZ below!
    ##q1=0 is a divergence, so cut it off:
    q1 = q1[1:]
    qvec = np.array([np.cos(phi),np.sin(phi),np.zeros((len(phi)))])
    #####
    beta_L = beta_long
    longitud = beta_long
    if isinstance(cs, list):
        c1 = cs[0]
        c2 = cs[1]
        if c1>c2:
            longitud = "1"
            beta2=beta ## assume beta is beta_T
            beta1=beta2*c2/c1
            beta_L=beta1
            qt_min = abs(c1/c2-1)/(1+beta2)
            qt_max = (1+c1/c2)
            if Nt_total is None:
                if updatet:
                    dt = 1/(2*Nt)
                    qtilde = np.linspace(qt_min+dt,qt_max-dt,Nt)
                else:
                    qtilde = np.linspace(qt_min,qt_max,Nt)
            else:
                subqtmin = qt_min + (qt_max-qt_min)*kthchk/Nchunks
                subqtmax = qt_min + (qt_max-qt_min)*(1+kthchk)/Nchunks
                if updatet:
                    dt = (subqtmax-subqtmin)/(2*Nt)
                    qtilde = np.linspace(subqtmin+dt,subqtmax-dt,Nt)
                else:
                    qtilde = np.linspace(subqtmin,subqtmax,Nt)
        elif c1==c2: ## TODO: test this branch with adaptive grid patch (not used by default)!
            qt_max = (1+c1/c2)
            if beta_long is False:
                beta1=beta
                beta2=beta
            else:
                beta1=beta_L
                beta2=beta_L
            if Nt_total is None:
                if updatet:
                    dt = 1/(2*Nt)
                    qtilde = np.linspace(dt,qt_max-dt,Nt)
                else:
                    qtilde = np.linspace(0,qt_max,Nt)
            else:
                subqtmin = qt_max*kthchk/Nchunks
                subqtmax = qt_max*(1+kthchk)/Nchunks
                if updatet:
                    dt = (subqtmax-subqtmin)/(2*Nt)
                    qtilde = np.linspace(subqtmin+dt,subqtmax-dt,Nt)
                else:
                    qtilde = np.linspace(subqtmin,subqtmax,Nt)
        else:
            longitud = "2"
            beta1=beta ## assume beta is beta_T
            beta2=beta1*c1/c2
            beta_L=beta2
            qt_min = abs(1-c1/c2)/(1+beta2)
            qt_max = (1+c1/c2)
            if Nt_total is None:
                if updatet:
                    dt = 1/(2*Nt)
                    qtilde = np.linspace(qt_min+dt,qt_max-dt,Nt)
                else:
                    qtilde = np.linspace(qt_min,qt_max,Nt)
            else:
                subqtmin = qt_min + (qt_max-qt_min)*kthchk/Nchunks
                subqtmax = qt_min + (qt_max-qt_min)*(1+kthchk)/Nchunks
                if updatet:
                    dt = (subqtmax-subqtmin)/(2*Nt)
                    qtilde = np.linspace(subqtmin+dt,subqtmax-dt,Nt)
                else:
                    qtilde = np.linspace(subqtmin,subqtmax,Nt)
        ###
        t = np.outer((qtilde+(1-c1**2/c2**2)/qtilde)/2,np.ones((len(phi)))) + np.outer(np.ones((len(qtilde))),(c1*beta2/c2)*np.abs(np.cos(phi))) - np.outer(qtilde/2,(beta2*np.cos(phi))**2)
        ### when integrating t later, need to slice such that -1<=t<=1 is ensured;
        ### also notice that this restricts the range of qtilde, i.e.: abs(1-c1/c2)/(1+beta2) <= qtilde <= (1+c1/c2)/(1-beta2) for all c1, c2; hence the definitions above for qt_min and qt_max
        prefactor1 = dragcoeff_iso_computeprefactor(qBZ, cs, [beta, beta_L], burgers, q1, phi, qtilde, T, r0cut=r0cut)
    else:
        if beta_long is False:
            beta1=beta
        else:
            beta1=beta_L
        if Nt_total is None:
            # if updatet and Nt % 2 == 0:
            ## always need the range [dt,1-dt] for any refinement (i.e. endpoints are present only in the initial run), no matter if even or not
            if updatet:
                dt = 1/(2*Nt)
                t = np.linspace(dt,1-dt,Nt)
            else:
                t = np.linspace(0,1,Nt)
        else:
            tmin = kthchk/Nchunks
            tmax = (1+kthchk)/Nchunks
            if updatet:
                dt = (tmax-tmin)/(2*Nt)
                t = np.linspace(tmin+dt,tmax-dt,Nt)
            else:
                t = np.linspace(tmin,tmax,Nt)
        qtilde = 2*(np.outer(t,np.ones((len(phi)))) - beta1*np.outer(np.ones((len(t))),np.abs(np.cos(phi))))/np.outer(np.ones((len(t))),(1-(beta1*np.cos(phi))**2))
        qtilde[qtilde==0] = 1e-30 ## avoid 1/0

        if Debye_series and r0cut is None:
            prefactor1 = computeprefactorHighT(qBZ, cs, [beta, beta_L], burgers, phi, qtilde,T)
        else:
            prefactor1 = dragcoeff_iso_computeprefactor(qBZ, cs, [beta, beta_L], burgers, q1, phi, qtilde, T, r0cut=r0cut)
    
    Ntheta = len(dij[0,0])
    Bmix = np.empty((Ntheta,len(t),len(phi)))
    Bmixfinal = np.zeros((Ntheta))
    
    if A3[0,0,0,0,0,0].shape == ():
        poly = dragcoeff_iso_computepoly(A3, phi, qvec, qtilde, t, phi1, longitud)
        for th in range(Ntheta):
            if usefortran:
                Bmix[th] = fsub.dragintegrand(prefactor1.T,np.moveaxis(dij[:,:,th],2,0),np.moveaxis(poly,5,0)).T
            else:
                Bmix[th] = dragcoeff_iso_Bintegrand(prefactor1,dij[:,:,th],poly)
            if isinstance(cs, list):
                Bmixfinal[th] = integrateqtildephi(Bmix[th],beta1,qtilde,t,phi,updatet,kthchk,Nchunks)
            else:
                if beta_long is False:
                    Bmixfinal[th] = integratetphi(Bmix[th],beta,t,phi,updatet,kthchk)
                else:
                    Bmixfinal[th] = integratetphi(Bmix[th],beta_L,t,phi,updatet,kthchk)
    else:
        for th in range(Ntheta):
            poly = dragcoeff_iso_computepoly(A3[th], phi, qvec, qtilde, t, phi1, longitud)
            if usefortran:
                Bmix[th] = fsub.dragintegrand(prefactor1.T,np.moveaxis(dij[:,:,th],2,0),np.moveaxis(poly,5,0)).T
            else:
                Bmix[th] = dragcoeff_iso_Bintegrand(prefactor1,dij[:,:,th],poly)
            if isinstance(cs, list):
                Bmixfinal[th] = integrateqtildephi(Bmix[th],beta1,qtilde,t,phi,updatet,kthchk,Nchunks)
            else:
                if beta_long is False:
                    Bmixfinal[th] = integratetphi(Bmix[th],beta,t,phi,updatet,kthchk)
                else:
                    Bmixfinal[th] = integratetphi(Bmix[th],beta_L,t,phi,updatet,kthchk)
    
    return Bmixfinal

def phonondrag(disloc,beta,Nq=50,rmin=0,rmax=250,Nphi=50,skiptransonic=True,Ncores=Ncores,forceanis=False,**kwargs):
    '''Computes the drag coefficient from phonon wind for dislocation 'disloc' gliding at velocity v=beta*disloc.ct
       'disloc'' must be an instance of the Dislocation class; beta may be an array of (normalized) velocities.
       The drag coefficient is computed for every character angle in disloc.theta and is thus returned as an array
       of shape (len(beta),len(disloc.theta)). Additional keyword arguments may be passed to subroutine dragcoeff_iso()
       via kwargs. If disloc.sym==iso, we use the faster analytic expressions for the displacement gradient field unless
       option  'forceanis'=True.'''
    # if not isinstance(disloc,Dislocation):
    #     print(type(disloc),isinstance(disloc,Dislocation),Dislocation)
    #     raise ValueError("'disloc' must be an instance of the Dislocation class")
    if isinstance(beta, float) or isinstance(beta, int):
        beta = np.asarray([beta])
    else:
        beta = np.asarray(beta)
    disloc.C2norm = UnVoigt(disloc.C2/disloc.mu)
    C3 = UnVoigt(disloc.C3/disloc.mu)
    A3 = elasticA3(disloc.C2norm,C3)
    phi = np.linspace(0,2*np.pi,Nphi)
    if disloc.sym != 'iso' or forceanis:
        phiX = disloc.phi
        r = np.array([rmin*np.pi,rmax*np.pi])
        q = np.linspace(0,1,Nq)
        sincos_noq = np.average(fourieruij_sincos(r,phiX,q,phi)[3:-4],axis=0)
        A3rotated = np.zeros((disloc.Ntheta,3,3,3,3,3,3))
        disloc.computerot()
        rotmat = np.round(disloc.rot,15)
        for th in range(disloc.Ntheta):
            rotm = rotmat[th]
            A3rotated[th] = np.round(np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(A3,rotm.T)))))),12)
    else:
        A3rotated = A3
    if skiptransonic and (disloc.vcrit_all is None or len(disloc.vcrit_all[1])!=disloc.Ntheta or np.any(disloc.vcrit_all[0]!=disloc.theta)):
        disloc.computevcrit(set_screwedge=False)
        if np.isnan(disloc.vcrit_all[1]).any():
            print(f"Warning: found NaN in vcrit for {disloc.name}, replacing with interpolated values.")
            fixnan = pd.Series(disloc.vcrit_all[1])
            disloc.vcrit_all[1] = fixnan.where(fixnan.notnull(),other=(fixnan.fillna(method='ffill')+fixnan.fillna(method='bfill'))/2).to_numpy()
    
    def maincomputations(bt):
        '''wrap all main computations into a single function definition to be run in a parallelized loop'''
        Bmix = np.zeros((disloc.Ntheta))
        if disloc.sym != 'iso' or forceanis:
            disloc.computeuij(beta=bt)
            disloc.alignuij()
            dij = fourieruij_nocut(disloc.uij_aligned,phiX,phi,sincos=sincos_noq)
        else:
            dij = fourieruij_iso(bt, disloc.ct_over_cl, disloc.theta, phi)
        if not skiptransonic:
            skip_theta = None
        else:
            skip_theta = bt < disloc.vcrit_all[1]/disloc.ct
        if np.all(skip_theta==False):
            Bmix = np.repeat(np.inf,disloc.Ntheta)
        else:
            Bmix = dragcoeff_iso(dij=dij, A3=A3rotated, qBZ=disloc.qBZ, ct=disloc.ct, cl=disloc.cl, beta=bt, burgers=disloc.burgers, T=disloc.T, skip_theta=skip_theta, name=disloc.name, **kwargs)        
        return Bmix
    if Ncores == 1 or len(beta)<2:
        Bmix = np.array([maincomputations(bt) for bt in beta])
    else:
        Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt) for bt in beta))
    return Bmix
