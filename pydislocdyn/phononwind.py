#!/usr/bin/env python3
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Feb. 23, 2026
'''This module implements the calculation of a dislocation drag coefficient from phonon wind.
   Its front-end functions are :
       elasticA3 ...... computes the coefficient A3 from the SOECs and TOECs
       dragcoeff_iso ....... computes the drag coefficient assuming an isotropic phonon spectrum.
       phonondrag ........ a high-level wrapper around dragcoeff_iso that takes an instance of the
                           Dislocation class as its first argument.
                           Users most likely will want to use this function instead of dragcoeff_iso().
      B_of_sigma.......derives drag coefficient B(stress) from B(velocity) by using a fitting funcion
                       for the latter (which is a required input, see mkfit_Bv)
   All other functions are subroutines of the latter.'''
#################################
import ast
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, root
import pandas as pd
from pydislocdyn.utilities import Ncores, usefortran, hbar, kB, str2bool, init_parser, \
    plt, fntsettings, AutoMinorLocator ## matplotlib stuff
from pydislocdyn.elasticconstants import UnVoigt
from pydislocdyn.dislocations import Dislocation, fourieruij_sincos, fourieruij_nocut, fourieruij_iso
if usefortran:
    from pydislocdyn.subroutines import elastica3, phononwind_xx, phononwind_xy
    def elasticA3(C2, C3):
        '''Returns the tensor of elastic constants as it enters the interaction of dislocations with phonons. Required inputs are the tensors of SOEC and TOEC.'''
        A3 = elastica3(C2,C3)
        return A3
else:
    from pydislocdyn._phononwind_numba import elasticA3, phononwind_xx, phononwind_xy
if Ncores>1:
    from joblib import Parallel, delayed

### rho x ct^2  = c44, and B is divided by rho^2*ct^4 = c44^2;
### it is therefore convenient to divide A3 by c44 as it enters quadratically, and this is a requirement below, i.e. A3 must be rescaled by c44 to be dimensionless!
def dragcoeff_iso(dij, A3, qBZ, ct, cl, beta, burgers, T, modes='all', Nt=321, Nq1=400, Nphi1=50, Debye_series=False, target_accuracy=5e-3, maxrec=6, accurate_to_digit=1e-5, Nchunks=1, skip_theta=None, skip_theta_val=np.inf, r0cut=-1, name='drag'):
    '''Computes the drag coefficient from phonon wind for an isotropic crystal. Required inputs are the dislocation displacement gradient (times magnitude q and rescaled by the Burgers vector) dij in Fourier space
       (being a 3x3xNthetaxNphi array where theta is the angle parametrizing the dislocation type and phi is the polar angle in Fourier space), the array of shifted 3rd order elastic constants A3 in units of the shear modulus mu,
       the radius of the Brillouin zone qBZ, the transverse and longitudinal sound speeds ct and cl, the velocity beta in units of ct (i.e. beta=v/ct), the magnitude of the Burgers vectors burgers, as well as the temperature T.
       The keyword 'modes' determines which phonon contributions should be included in the computation.
       Allowed values are: 'all' (default), 'TT' (only transverse phonons), 'LL' (only longitudinal), 'mix' (only the two mixed modes), 'LT' (incoming transverse, outgoing longitudinal phonon), and 'TL' (incoming longitudinal outgoing transverse phonon).
       Optionally, the default values for the resolution of integration variables t, q1, and phi1 may be changed. Note that Nt is automatically increased with larger beta depending on the computed phonon mode.
       The parameter 'Debye_series' may be set to True in order to use the first 4 terms of the series representation of the Debye functions instead of computing the Debye integral over the phonon spectrum numerically.
       Note, however, that the series representation converges only for high enough temperature.
       Optional variable skip_theta = None (default) may be set in order to bypass the calculation for certain angles theta and instead set those entries to some predefined value=skip_theta_val,
       i.e. skip_theta must be a boolean mask of len(theta) where False=bypass.
       If option r0cut>0 (turned off by default), a soft dislocation core cutoff is included following Alshits 1979, i.e. multiplying the dislocation field by (1-exp(r/r0)) which leads to 1/sqrt(1-q**2/r0**2) in Fourier space
       Note: this cutoff is intended only for an isotropic dislocation field at low gliding velocity, as the shape of the cutoff is beta-dependent the way it is introduced (i.e. a circle only at beta=0).''' ##(TODO: generalize)
    Ntheta = len(dij[0,0])
    theta_ind = np.arange((Ntheta)) # generate array of theta-indices for later use
    modes_allowed = ['all', 'TT', 'LL', 'LT', 'TL', 'mix'] ## define allowed keywords for modes
    debye_convergence = hbar*cl*qBZ/(np.pi*kB) # warn if T < half the convergence limit for the longitudinal modes
    if beta <0 or beta>1:
        raise ValueError(f"{beta=}, but must be between 0 and 1.")
    if Debye_series and r0cut>0:
        Debye_series = False # ensure consistent behavior between python and fortran implementations below
        print("Warning: r0cut is set, therefore ignoring 'Debye_series=True'.")
    elif Debye_series and T<debye_convergence:
        print(f"Warning: using the high temperature expansion of the Debye functions at temperature {T=}K; result will be inaccurate. Recommend running with 'Debye_series=False'.")
    iso = False
    if A3[0,0,0,0,0,0].shape == ():
        A3 = np.repeat(A3[None,:],1,axis=0)
        iso=True
    dij=np.moveaxis(dij, -1, 0) ## shape:  (i,j,Ntheta,Nphi) -> (Nphi,i,j,Ntheta)
    A3=np.moveaxis(A3,0,-1) ## shape:  (Ntheta,i,ii,j,jj,k,kk) -> (i,ii,j,jj,k,kk,Ntheta)
    Nchks = Nchunks
    ## make sure Nt_total-1 is divisible by 2*Nchunks, and that Nt_current is odd and >=5 (i.e. increase user provided Nt as necessary)
    Nt_k = int(abs(Nt - 1 - 4*Nchunks)/(2*Nchunks))
    Nt_total = 4*Nchunks + 2*Nt_k*Nchunks + 1
    if Nt_total < Nt:
        Nt_total += 2*Nchunks ## ensure we never reduce resolution
    
    def adaptive_t_chunks(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, target_accuracy=target_accuracy, \
                          maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta=skip_theta, r0cut=r0cut, Nt_total=Nt, Nchunks=Nchunks, mode='??'):
        if Nchunks==1:
            out = adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, target_accuracy=target_accuracy, \
                             maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta=skip_theta, r0cut=r0cut, Nt=Nt, mode=mode)
        else:
            out = 0
            Nt_current = int((Nt_total-1)/Nchunks+1)
            for kth in range(Nchunks):
                out += adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta_long, target_accuracy=target_accuracy, \
                                  maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta=skip_theta, r0cut=r0cut, chunks=(Nchunks,kth), Nt=Nt_current, mode=mode)
        return out
    
    def adaptive_t(dij, A3, qBZ, cs, beta, burgers, T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, target_accuracy=target_accuracy, \
                   maxrec=maxrec, accurate_to_digit=accurate_to_digit, skip_theta=skip_theta, r0cut=r0cut, chunks=(1,0), Nt=Nt, mode='??'):
        if skip_theta is None:
            dijtmp = dij
            A3tmp = A3
        elif iso:
            dijtmp = dij[:,:,:,skip_theta]
            A3tmp = A3
        else:
            dijtmp = dij[:,:,:,skip_theta]
            A3tmp = A3[:,:,:,:,:,:,skip_theta]
        args = (dijtmp, A3tmp, qBZ, cs, beta, burgers, T)
        Ntheta = len(dijtmp[0,0,0])
        theta_ind = np.arange((Ntheta))
        Ntauto_old = int(Nt/2)+1
        Nphi = len(dijtmp)
        if mode in ('TT','LL'):
            args = {'dij':dijtmp, 'a3':A3tmp, 'qbz':qBZ, 'ct':ct, 'cl':0, 'beta':beta, 'burgers':burgers, 'temp':T, 'r0cut':r0cut, 'debye':Debye_series}
            if beta_long is not False:
                args['cl'] = cl
            out_old = phononwind_xx(**args, lentheta=Ntheta, lent=Ntauto_old, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=chunks)
            Ntauto = Ntauto_old-1 ## number of points to add (i.e. total is 2*Ntauto_old-1); refine previous result
            out = out_old/2 + phononwind_xx(**args, lentheta=Ntheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)/2
        else:
            c1,c2 = cs
            args = {'dij':dijtmp, 'a3':A3tmp, 'qbz':qBZ, 'cx':c1, 'cy':c2, 'beta':beta, 'burgers':burgers, 'temp':T, 'r0cut':r0cut, 'debye':Debye_series}
            out_old = phononwind_xy(**args, lentheta=Ntheta, lent=Ntauto_old, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=chunks)
            Ntauto = Ntauto_old-1 ## number of points to add (i.e. total is 2*Ntauto_old-1); refine previous result
            out = out_old/2 + phononwind_xy(**args, lentheta=Ntheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)/2
        ## determine if target accuracy was achieved
        err_mask = out!=0
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
            if iso:
                if mode in ('TT','LL'):
                    out = out_old/2 + phononwind_xx(**args, lentheta=Ntheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)/2
                else:
                    out = out_old/2 + phononwind_xy(**args, lentheta=Ntheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)/2
            else:
                ## refine only needed angles theta if A3 is theta dependent to save computation time
                theta_mask = out_error_all/out_norm>=target_accuracy
                theta_refine = (theta_ind[err_mask])[theta_mask]
                if mode in ('TT','LL'):
                    newargs = {'dij':(dijtmp[:,:,:,err_mask])[:,:,:,theta_mask], 'a3':(A3tmp[:,:,:,:,:,:,err_mask])[:,:,:,:,:,:,theta_mask], 'qbz':qBZ, 'ct':ct, \
                               'beta':beta, 'burgers':burgers, 'temp':T, 'r0cut':r0cut, 'debye':Debye_series}
                    newargs['cl'] = args['cl']
                    newNtheta = len(newargs['a3'][0,0,0,0,0,0])
                    out_newpoints = phononwind_xx(**newargs, lentheta=newNtheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)
                else:
                    newargs = {'dij':(dijtmp[:,:,:,err_mask])[:,:,:,theta_mask], 'a3':(A3tmp[:,:,:,:,:,:,err_mask])[:,:,:,:,:,:,theta_mask], 'qbz':qBZ, 'cx':c1, 'cy':c2, \
                               'beta':beta, 'burgers':burgers, 'temp':T, 'r0cut':r0cut, 'debye':Debye_series}
                    newNtheta = len(newargs['a3'][0,0,0,0,0,0])
                    out_newpoints = phononwind_xy(**newargs, lentheta=newNtheta, lent=Ntauto, lenph=Nphi, lenq1=Nq1, lenph1=Nphi1, updatet=True, chunks=chunks)
                for th, thi in enumerate(theta_refine):
                    out[thi] = out_old[thi]/2 + out_newpoints[th]/2
            err_mask = out!=0
            out_error_all = np.abs(np.abs(out)-np.abs(out_old))[err_mask]
            if len(out_error_all)==0:
                out_error = 0
            else:
                out_norm = (np.abs(out))[err_mask]
                out_error = min(np.max(out_error_all/out_norm),np.max(target_accuracy*out_error_all/(accurate_to_digit/Nchks)))
            refnmts = rec+1
        if refnmts==maxrec and out_error > 2*target_accuracy and maxrec>0:
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
    
    if modes in ('all', 'TT'):
        if maxrec<0: ## bypass adaptive grid if requested by user
            Ntauto = int((1+beta)*Nt) ## we know we need more points at higher beta, so already start from higher value
            BTT = phononwind_xx(dij=dij, a3=A3, qbz=qBZ, ct=ct, cl=0, beta=beta, burgers=burgers, temp=T, lentheta=Ntheta, lent=Ntauto, lenph=len(dij), \
                                lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=(1,0), r0cut=r0cut, debye=Debye_series)
        else:
            BTT = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=ct, beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='TT')
        
    if modes in ('all', 'LL'):
        if maxrec<0:
            Ntauto = int((1+beta/2)*Nt)
            BLL = phononwind_xx(dij=dij, a3=A3, qbz=qBZ, ct=ct, cl=cl, beta=beta, burgers=burgers, temp=T, lentheta=Ntheta, lent=Ntauto, lenph=len(dij), \
                                lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=(1,0), r0cut=r0cut, debye=Debye_series)
        else:
            BLL = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=cl, beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=beta*ct/cl, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='LL')
        
    if modes in ('all', 'mix', 'TL'):
        if maxrec<0:
            Ntauto = int((1+beta/2)*Nt)
            BTL = phononwind_xy(dij=dij, a3=A3, qbz=qBZ, cx=ct, cy=cl, beta=beta, burgers=burgers, temp=T, lentheta=Ntheta, lent=Ntauto, lenph=len(dij), \
                                lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=(1,0), r0cut=r0cut, debye=Debye_series)
        else:
            BTL = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=[ct,cl], beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='TL')
        
    if modes in ('all', 'mix', 'LT'):
        if maxrec<0:
            Ntauto = int((1+beta)*Nt)
            BLT = phononwind_xy(dij=dij, a3=A3, qbz=qBZ, cx=cl, cy=ct, beta=beta, burgers=burgers, temp=T, lentheta=Ntheta, lent=Ntauto, lenph=len(dij), \
                                lenq1=Nq1, lenph1=Nphi1, updatet=False, chunks=(1,0), r0cut=r0cut, debye=Debye_series)
        else:
            BLT = adaptive_t_chunks(dij=dij, A3=A3, qBZ=qBZ, cs=[cl,ct], beta=beta, burgers=burgers, T=T, Nq1=Nq1, Nphi1=Nphi1, Debye_series=Debye_series, beta_long=False, r0cut=r0cut, Nt_total=Nt_total, Nchunks=Nchunks, mode='LT')
    
    if skip_theta is None:
        out = BTT + BLL + BTL + BLT
    else:
        out = skip_theta_val*np.ones((Ntheta))
        theta_calcd = theta_ind[skip_theta]
        for th, thi in enumerate(theta_calcd):
            out[thi] = BTT[th] + BLL[th] + BTL[th] + BLT[th]
    
    if modes not in modes_allowed:
        raise ValueError(f"Error: invalid keyword {modes=}.")
    
    return out

def phonondrag(disloc,beta,Nq=50,rmin=0,rmax=250,Nphi=50,skiptransonic=True,Ncores=Ncores,forceanis=False,pandas_out=True,**kwargs):
    '''Computes the drag coefficient from phonon wind in mPas for dislocation 'disloc' gliding at velocity v=beta*disloc.ct
       'disloc'' must be an instance of the Dislocation class; beta may be an array of (normalized) velocities.
       The drag coefficient is computed for every character angle in disloc.theta and is thus returned as an array
       of shape (len(beta),len(disloc.theta)). Additional keyword arguments may be passed to subroutine dragcoeff_iso()
       via kwargs. If disloc.sym==iso, we use the faster analytic expressions for the displacement gradient field unless
       option  'forceanis'=True. If option pandas_out=True (default), the resulting numpy array is converted to a pandas
       DataFrame including metadata (i.e. dislocaion gliding velocities and character angles.'''
    if not isinstance(disloc,Dislocation):
        print(type(disloc),isinstance(disloc,Dislocation),Dislocation)
        raise ValueError("'disloc' must be an instance of the Dislocation class")
    if isinstance(beta, (float, int)):
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
        sincos_noq = fourieruij_sincos(r[0],r[-1],phiX,q[3:-4],phi)
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
            dij = fourieruij_nocut(disloc.uij_aligned,phiX,sincos=sincos_noq,Ntheta=disloc.Ntheta,phres=Nphi)
        else:
            dij = fourieruij_iso(bt, disloc.ct_over_cl, disloc.theta, phi)
        if not skiptransonic:
            skip_theta = None
        else:
            skip_theta = bt < disloc.vcrit_all[1]/disloc.ct
        if np.all(skip_theta==False): ## ignore pylint on this one: 'is False' does not work as expected on arrays
            Bmix = np.repeat(np.inf,disloc.Ntheta)
        else:
            Bmix = dragcoeff_iso(dij=dij, A3=A3rotated, qBZ=disloc.qBZ, ct=disloc.ct, cl=disloc.cl, beta=bt, burgers=disloc.burgers, T=disloc.T, skip_theta=skip_theta, name=disloc.name, **kwargs)
        return Bmix
    if Ncores == 1 or len(beta)<2:
        Bmix = np.array([maincomputations(bt) for bt in beta])
    else:
        Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt) for bt in beta))
    if pandas_out:
        Bmix = pd.DataFrame(Bmix,index=beta,columns=disloc.theta/np.pi)
        # Bmix.attrs = {'name' : f"B({disloc.name}) [mPas]"} ## .attrs is an experimental pandas feature (might go away again) and .name is not supported
        Bmix.index.name = 'beta'
        Bmix.columns.name = 'theta/pi'
    return Bmix

### define various functions for fitting drag results and for computing B(sigma) from (fitted) B(v)
def fit_mix(x, c0, c1, c2, c4):
    '''Defines a fitting function appropriate for drag coefficient B(v).'''
    return c0 - c1*x + c2*(1/(1-x**2)**(1/2) - 1) + c4*(1/(1-x**2)**(3/2) - 1)

def mkfit_Bv(Y,Bdrag,scale_plot=1,Bmax_fit='auto'):
    '''Calculates fitting functions for B(v) for pure screw, pure edge, and an average over all dislocation characters.
       Required inputs are an instance of the Dislocation class Y, and the the drag coefficient Bdrag formatted as a Pandas DataFrame where index
       contains the normalized velocities beta= v/Y.ct and columns contains the character angles theta at velocity v for all character angles theta.'''
    if not isinstance(Y,Dislocation):
        raise ValueError("'Y' must be an instance of the Dislocation class")
    if not isinstance(Bdrag,pd.DataFrame):
        raise ValueError("'Bdrag' must be a Pandas DataFrame.")
    Broom = Bdrag.to_numpy()
    vel = Bdrag.index.to_numpy()*Y.ct
    theta = Bdrag.columns
    Y.scale_plot = max(scale_plot,int(Y.T/30)/10)
    if Bmax_fit=='auto':
        Bmax_fit = int(20*Y.T/300)/100 ## only fit up to Bmax_fit [mPas]
    if theta[0]==0.:
        Y.scrind = 0
    else:
        Y.scrind = int(len(theta)/2)
    Y.Baver = np.average(Broom,axis=-1)
    beta_edgecrit = (vel/Y.vcrit_edge)[vel<Y.vcrit_edge]
    beta_screwcrit = (vel/Y.vcrit_screw)[vel<Y.vcrit_screw]
    beta_avercrit = (vel/Y.vcrit_smallest)[vel<Y.vcrit_smallest]
    ### having cut everything beyond the critical velocities (where B diverges), we additionally remove very high values (which are likely inaccurate close to vcrit) to improve the fits everywhere else; adjust Bmax_fit to your needs!
    beta_edgecrit = beta_edgecrit[[j for j in range(len(beta_edgecrit)) if Broom[j,-1] <Bmax_fit + np.min(Broom[:,-1])]]
    beta_screwcrit = beta_screwcrit[[j for j in range(len(beta_screwcrit)) if Broom[j,Y.scrind]<Bmax_fit + np.min(Broom[:,Y.scrind])]]
    beta_avercrit = beta_avercrit[[j for j in range(len(beta_avercrit)) if Y.Baver[j]<Bmax_fit + np.min(Y.Baver)]]
    popt_edge, pcov_edge = curve_fit(fit_mix, beta_edgecrit[beta_edgecrit<0.995], (Broom[:len(beta_edgecrit)])[beta_edgecrit<0.995,-1], bounds=([0.9*Broom[0,-1],0.,-0.,-0.], [1.1*Broom[0,-1], 2*Broom[0,-1], 1., 1.]))
    popt_screw, pcov_screw = curve_fit(fit_mix, beta_screwcrit[beta_screwcrit<0.995], (Broom[:len(beta_screwcrit)])[beta_screwcrit<0.995,Y.scrind], bounds=([0.9*Broom[0,Y.scrind],0.,-0.,-0.], [1.1*Broom[0,Y.scrind], 2*Broom[0,Y.scrind], 1., 1.]))
    popt_aver, pcov_aver = curve_fit(fit_mix, beta_avercrit[beta_avercrit<0.995], (Y.Baver[:len(beta_avercrit)])[beta_avercrit<0.995], bounds=([0.9*Y.Baver[0],0.,-0.,-0.], [1.1*Y.Baver[0], 2*Y.Baver[0], 1., 1.]))
    return popt_edge, pcov_edge, popt_screw, pcov_screw, popt_aver, pcov_aver

def B_of_sigma(Y,popt,character,mkplot=True,B0fit='weighted',resolution=500,indirect=False,fit=fit_mix,sigma_max='auto'):
    '''Computes arrays sigma and B_of_sigma of length 'resolution', and returns a tuple (B0,vcrit,sigma,B_of_sigma) where B0 is either the minimum value, or B(v=0) if B0fit=='zero'
       or a weighted average of the two (B0fit='weighted',default) and vcrit is the critical velocity for character (='screw', 'edge', or else an average is computed).
       Required inputs are an instance of the Dislocation class Y, fitting parameters popt previously calculated using fitting function fit_mix() (use option 'fit' to change),
       and a keyword 'character = 'screw'|'edge'|'aver'.
       A plot of the results is saved to disk if mkplot=True (default).
       If option indirect=False sigma will be evenly spaced (default), whereas if indirect=True sigma will be calculated from an evenly spaced velocity array.
       The latter is also used as fall back behavior if the computation of v(sigma) fails to converge.
       Option 'sigma_max'' is the highest stress to be considered in the present calculation. '''
    if not isinstance(Y,Dislocation):
        raise ValueError("'Y' must be an instance of the Dislocation class")
    ftitle = f"{Y.name}, {character}"
    fname = f"B_of_sigma_{character}_{Y.name}.pdf"
    if character=='screw':
        vcrit = Y.vcrit_screw
    elif character=='edge':
        vcrit = Y.vcrit_edge
    else: ## 'aver' = default
        vcrit = Y.vcrit_smallest
        fname = f"B_of_sigma_{Y.name}.pdf"
        ftitle = fr"{Y.name}, averaged over $\vartheta$"
    burg = Y.burgers
    
    @np.vectorize
    def B(v):
        bt = abs(v/vcrit)
        if bt<1:
            out = 1e-3*fit(bt, *popt)
        else:
            out = np.inf
        return out
        
    @np.vectorize
    def vr(stress):
        '''Returns the velocity of a dislocation in the drag dominated regime as a function of stress.'''
        bsig = abs(burg*stress)
        def nonlinear_equation(v):
            return abs(bsig-abs(v)*B(v)) ## need abs() if we are to find v that minimizes this expression (and we know that minimum is 0)
        themin = minimize_scalar(nonlinear_equation,method='bounded',bounds=(0,1.01*vcrit))
        out = themin.x
        zero = abs(themin.fun)
        if not themin.success or zero>1e-8:
            # fall back to (slower) root (try x0 close to vcrit as this case typically happens for very large stress):
            themin = root(nonlinear_equation,x0=0.999*vcrit)
            out = themin.x[0]
            zero=themin.fun[0]
            if not themin.success or zero>1e-8:
                print(f"Warning: bad convergence for {Y.name}, {character}; got vr({stress=:.3e})={out[0]:.2f}; {vcrit=:.2f}, eq={zero:.6e}")
        return out
        
    @np.vectorize
    def sigma_eff(v):
        '''Compute what stress is needed to move dislocations at velocity v.'''
        return v*B(v)/burg
        
    @np.vectorize
    def Bstraight(sigma,Boffset=0):
        '''Returns the slope of B in the asymptotic regime.'''
        return Boffset+sigma*burg/vcrit
        
    @np.vectorize
    def Bsimple(sigma,B0):
        '''Simple functional approximation to B(sigma), follows from B(v)=B0/sqrt(1-(v/vcrit)**2).'''
        return B0*np.sqrt(1+(sigma*burg/(vcrit*B0))**2)
    
    if sigma_max=='auto':
        ## determine stress that will lead to velocity of 99% critical speed and stop plotting there, or at 1.5GPa (whichever is smaller)
        sigma_max = sigma_eff(0.99*vcrit)
        # print(f"{Y.name}, {character}: sigma(99%vcrit) = {sigma_max/1e6:.1f} MPa")
        if sigma_max<6e8 and B(0.99*vcrit)<1e-4: ## if B, sigma still small, increase to 99.9% vcrit
            sigma_max = sigma_eff(0.999*vcrit)
            # print(f"{Y.name}, {character}: sigma(99.9%vcrit) = {sigma_max/1e6:.1f} MPa")
        sigma_max = min(1.5e9,sigma_max)
    Boffset = max(float(B(vr(sigma_max))-Bstraight(sigma_max,0)), 0) ## don't allow negative values
    ## find min(B(v)) to use for B0 in Bsimple():
    B0 = round(np.min(B(np.linspace(0,0.8*vcrit,1000))),7)
    if B0fit == 'weighted':
        B0 = (B(0)+3*B0)/4 ## or use some weighted average between Bmin and B(0)
    elif B0fit == 'zero':
        B0 = B(0)
    # print(f"{Y.name}: Boffset={1e3*Boffset:.4f}mPas, B0={1e3*B0:.4f}mPas")
    
    sigma = np.linspace(0,sigma_max,resolution)
    if not indirect:
        B_of_sig = B(vr(sigma))
        Bmax = B_of_sig[-1]
    if indirect or (np.max(B_of_sig) < 1.01*B(0)):
        # print(f"\nWARNING: using fall back for v(sigma) for {Y.name}, {character}\n")
        v = vcrit*np.linspace(0,0.999,resolution)
        sigma = sigma_eff(v)
        B_of_sig = B(v)
        Bmax = B_of_sig[-1]
        if sigma[-1]>1.1*sigma_max:
            Bmax = 1.15*B_of_sig[sigma<sigma_max][-1]
    if mkplot:
        fig, ax = plt.subplots(1, 1, sharey=False, figsize=(3.,2.5))
        ax.set_xlabel(r'$\sigma$[MPa]',**fntsettings)
        ax.set_ylabel(r'$B$[mPas]',**fntsettings)
        ax.set_title(ftitle,**fntsettings)
        ax.axis((0,sigma_max/1e6,0,Bmax*1e3))
        ax.plot(sigma/1e6,Bsimple(sigma,B0)*1e3,':',color='gray',label=r"$\sqrt{B_0^2\!+\!\left(\frac{\sigma b}{v_\mathrm{c}}\right)^2}$, $B_0=$"+f"{1e6*B0:.1f}"+r"$\mu$Pas")
        ax.plot(sigma/1e6,Bstraight(sigma,Boffset)*1e3,':',color='green',label=r"$B_0+\frac{\sigma b}{v_\mathrm{c}}$, $B_0=$"+f"{1e6*Boffset:.1f}"+r"$\mu$Pas")
        ax.plot(sigma/1e6,B_of_sig*1e3,label=r"$B_\mathrm{fit}(v(\sigma))$")
        plt.xticks(**fntsettings)
        plt.yticks(**fntsettings)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper left',handlelength=1.1, frameon=False, shadow=False,fontsize=8)
        plt.savefig(fname,format='pdf',bbox_inches='tight')
        plt.close()
    return (B0,vcrit,sigma,B_of_sig)

def init_drag_parser(**kwargs):
    '''initializes an instance of an argparse.ArgumentParser() class with some default options, allowing additional arguments to be passed via **kwargs.'''
    parser = init_parser(**kwargs)
    parser.add_argument('-minb','--minb', type=float, default=0.01, help='smallest normalized gliding velocity (units of ct by default, see option -beta_reference)')
    parser.add_argument('-maxb','--maxb', type=float, default=0.99, help='largest normalized gliding velocity (units of ct by default, see option -beta_reference)')
    parser.add_argument('-Nbeta', '--Nbeta', type=int, default=99, help="""number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)""")
    parser.add_argument('-modes','--modes', type=str, default='all', help="""phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T,
                                  'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four)""")
    parser.add_argument('-use_exp_Lame','--use_exp_Lame', type=str2bool, default=True, help='''if using data from metal_data, choose between experimentally determined Lame and Murnaghan constants (default)
                                   or analytical averages of SOEC and TOEC (use_exp_Lame = False)''')
    parser.add_argument('-Nphi', '--Nphi', type=int, default=50, help="""resolution of polar angle in Fourier space; keep this an even number for higher accuracy (because we integrate over 
                pi-periodic expressions in some places and phi ranges from 0 to 2pi)""")
    parser.add_argument('-NT','--NT', type=int, default=1, help=""""EXPERIMENTAL FEATURE - number of temperatures between baseT and maxT
                        (WARNING: implementation of temperature dependence is incomplete!)""")
    parser.add_argument('-constantrho','--constantrho', type=str2bool, default=False, help='set to True to override thermal expansion coefficient and use alpha_a = 0 for T > baseT')
    parser.add_argument('-increaseTby','--increaseTby', type=float, default=300, help='so that maxT=baseT+increaseTby (units=Kelvin)')
    parser.add_argument('-beta_reference','--beta_reference', default='base', type=str, help="""define beta=v/ct, choosing ct at baseT ('base') or current T ('current') as we increase temperature""")
    parser.add_argument('-phononwind_opts','--phononwind_opts', type=ast.literal_eval, default={}, help="""pass additional options to the phononwind subroutine (formatted as a dictionary)""")
    parser.add_argument('-allplots','--allplots', action='store_true', help='set to show more B_of_sigma plots for each metal')
    return parser
