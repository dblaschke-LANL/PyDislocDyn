# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Aug. 22, 2019
#################################
from __future__ import division
from __future__ import print_function

import sys
### make sure we are running a recent version of python
# assert sys.version_info >= (3,5)
import numpy as np
from scipy.optimize import curve_fit
##################
import matplotlib as mpl
##### use pdflatex and specify font through preamble:
# mpl.use("pgf")
# pgf_with_pdflatex = {
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": [
#           r"\usepackage[utf8x]{inputenc}",
#           r"\usepackage[T1]{fontenc}",
#           r"\usepackage{fouriernc}",
#           r"\usepackage{amsmath}",
#           ]
# }
# mpl.rcParams.update(pgf_with_pdflatex)
##################
import matplotlib.pyplot as plt
# plt.rc('font',**{'family':'Liberation Mono','size':'11'})
# plt.rc('font',**{'family':'Liberation Sans Narrow','size':'11'})
plt.rc('font',**{'family':'Liberation Serif','size':'11'})
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import metal_data as data
from elasticconstants import elasticC2, elasticC3
import dislocations as dlc
from phononwind import elasticA3, dragcoeff_iso
try:
    from joblib import Parallel, delayed
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    ## Ncores=0 bypasses phononwind calculations entirely and only generates plots using data from a previous run
    Ncores = -2
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = 1 ## must be 1 (or 0) without joblib

### choose various resolutions and other parameters:
Ntheta = 21 # number of angles between burgers vector and dislocation line (minimum 2, i.e. pure edge and pure screw)
Nbeta = 90 # number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)
minb = 0.01
maxb = 0.90
NT = 1 # number of temperatures between roomT and maxT (WARNING: implementation of temperature dependence is incomplete!)
roomT = 300 # in Kelvin
maxT = 600
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
# modes = 'TT'
skip_plots=False ## set to True to skip generating plots from the results
# in Fourier space:
Nphi = 50 # computation time scales linearly with resolution in phi, phi1 and t (each); increase for higher accuracy
Nphi1 = 50
Nq1 = 400
Nt = 250
Nq = 50 # only used in Fourier trafo of disloc. field, don't need such high resolution if cutoffs are chosen carefully since the q-dependence drops out in that case
# in x-space (used in numerical Fourier trafo):
NphiX = 3000
# Nr = 500
# cutoffs for r to be used in numerical Fourier trafo (in units of pi/qBZ)
# rmin = 1/6
# rmax = 90
### rmin smaller converges nicely, rmax bigger initially converges but if it gets to large (several hundred) we start getting numerical artefacts due to rapid oscillations
rmin = 0
rmax = 250
# r_reg = 3500 ## dimensionless regulator for Fourier transform with removed cutoffs rmax/rmin
### and range & step sizes
theta = np.linspace(0,np.pi/2,Ntheta)  ## note: some slip systems (such as bcc defined below) are asymmetric wrt theta->-theta
beta = np.linspace(minb,maxb,Nbeta)
highT = np.linspace(roomT,maxT,NT)
phi = np.linspace(0,2*np.pi,Nphi)
phiX = np.linspace(0,2*np.pi,NphiX)
###

## general rotation matrices
def Rotx(th):
    return np.array([[1,0,0],[0,np.cos(th),np.sin(th)],[0,-np.sin(th),np.cos(th)]])
def Roty(th):
    return np.array([[np.cos(th),0,-np.sin(th)],[0,1,0],[np.sin(th),0,np.cos(th)]])
def Rotz(th):
    return np.array([[np.cos(th),np.sin(th),0],[-np.sin(th),np.cos(th),0],[0,0,1]])

#### isotropic input data:
ac = data.CRC_a
cc = data.CRC_c
rho = data.CRC_rho
# rho = data.CRC_rho_sc
# 2nd order elastic constants taken from the CRC handbook:
c11 = data.CRC_c11
c12 = data.CRC_c12
c44 = data.CRC_c44
c13 = data.CRC_c13
c33 = data.CRC_c33
c66 = data.CRC_c66
## TOEC from various refs.
c111 = data.c111
c112 = data.c112
c123 = data.c123
c144 = data.c144
c166 = data.c166
c456 = data.c456

c113 = data.c113
c133 = data.c133
c155 = data.c155
c222 = data.c222
c333 = data.c333
c344 = data.c344
c366 = data.c366

### check with isotropic constants:
# c11 = data.ISO_c11
# c12 = data.ISO_c12
# c44 = data.ISO_c44
# c111 = data.ISO_c111
# c112 = data.ISO_c112
# c123 = data.ISO_c123
# c144 = data.ISO_c144
# c166 = data.ISO_c166
# c456 = data.ISO_c456

mu = data.ISO_c44 ## effective shear modulus of polycrystal
lam = data.ISO_c12
mu['Mo'] = 125.0e9 ## use the 'improved' average for polycrystalline Mo since we don't have experimental data (see results of 'polycrystal_averaging.py')
lam['Mo'] = 176.4e9
mu['Zr'] = 36.0e9 ## use the Hill average for polycrystalline Zr since we don't have experimental data (see results of 'polycrystal_averaging.py')
lam['Zr'] = 71.3e9

qBZ = {}
ct = {} ## may need some "effective" transverse sound speed
cl = {}
ct_over_cl = {}
burgers = {} # magnitude of Burgers vector
bulk = {}

### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. at least TOEC)
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(c111.keys()).intersection(mu.keys())))
metal_cubic = data.fcc_metals.union(data.bcc_metals).intersection(metal)

### compute various numbers for these metals
for X in metal_cubic:
    qBZ[X] = ((6*np.pi**2)**(1/3))/ac[X]
###
for X in data.hcp_metals.intersection(metal):
    qBZ[X] = ((4*np.pi**2/(ac[X]*ac[X]*cc[X]*np.sqrt(3)))**(1/3))
for X in data.tetr_metals.intersection(metal):
    qBZ[X] = ((6*np.pi**2/(ac[X]*ac[X]*cc[X]))**(1/3))
###
for X in metal:
    ## want to scale everything by the average (polycrystalline) shear modulus and thereby calculate in dimensionless quantities
    ct[X] = np.sqrt(mu[X]/rho[X])
    cl[X] = np.sqrt((lam[X]+2*mu[X])/rho[X])
    bulk[X] = lam[X] + 2*mu[X]/3
    ct_over_cl[X] = ct[X]/cl[X]

### define Burgers (unit-)vectors and slip plane normals for all metals
b = {}
n0 = {}
linet = {}
velm0 = {}
####
for X in data.fcc_metals.intersection(metal):
    burgers[X] = ac[X]/np.sqrt(2)
    b[X] = np.array([1,1,0]/np.sqrt(2))
    n0[X] = -np.array([1,-1,1]/np.sqrt(3))

for X in data.bcc_metals.intersection(metal):
    burgers[X] = ac[X]*np.sqrt(3)/2
    b[X] = np.array([1,-1,1]/np.sqrt(3))
    n0[X] = np.array([1,1,0]/np.sqrt(2))

### slip directions for hcp are the [1,1,bar-2,0] directions; the SOEC are invariant under rotations about the z-axis
### caveat: TOEC are only invariant under rotations about the z-axis by angles of n*pi/3; measurement was done with x-axis aligned with one of the slip directions
### therefore, may choose b parallel to x-axis
hcpslip = 'basal' ## default
# hcpslip = 'prismatic' ## uncomment to use
# hcpslip = 'pyramidal' ## uncomment to use
for X in data.hcp_metals.intersection(metal):
    burgers[X] = ac[X]
    b[X] = np.array([-1,0,0])
    ## basal slip:
    n0[X] = np.array([0,0,1]) ## slip plane normal = normal to basal plane
    if hcpslip=='prismatic':
        ## prismatic slip:
        n0[X] = np.array([0,-1,0])
    elif hcpslip=='pyramidal':
        ## pyramidal slip:
        n0[X] = np.array([0,-ac[X],cc[X]])/np.sqrt(ac[X]**2+cc[X]**2)
    
## just one of many possible slip systems in tetragonal crystals such as Sn (see Jpn J Appl Phys 32:3214 for a list):
## we choose here the simplest one with the shortest burgers vector in Sn (i.e. energetically most favorable),
## slip plane normal may be parallel to either x or y as C2,C3 are invariant under rotations by pi/2 about the z axis
for X in data.tetr_metals.intersection(metal):
    burgers[X] = cc[X]
    b[X] = np.array([0,0,-1])
    n0[X] = np.array([0,1,0])
    
###
rotmat = {}
## for fcc: rotate thz=pi/4, then thx=-atan1/sqrt(2)), then thy=(pi/2-theta)
def fccrotation(theta):
    return np.round(np.dot(Roty(np.pi/2-theta),np.round(np.dot(Rotx(-np.arctan2(1,np.sqrt(2))),Rotz(np.pi/4)),15)),15)

## for bcc: rotate thz=-pi/4, then thy=(pi/2-theta) - atan1/sqrt(2))
def bccrotation(theta):
    return np.round(np.dot(Roty(np.pi/2-theta-np.arctan2(1,np.sqrt(2))),Rotz(-np.pi/4)),15)
    
### rotation needed for the basal slip system (default):
def hcprotation(theta):
    return np.round(np.dot(Roty(-np.pi/2-theta),Rotx(np.pi/2)),15)

### rotation needed for the prismatic slip
def hcprotation_pris(theta):
    return np.round(np.dot(Roty(-np.pi/2-theta),Rotx(np.pi)),15)
    
### rotation needed for the pyramidal slip (material dependent!!)
def hcprotation_pyr(theta,X):
    return np.round(np.dot(Roty(-np.pi/2-theta),Rotx(np.pi/2+np.arcsin(ac[X]/np.sqrt(ac[X]**2+cc[X]**2)))),15)
    
### rotation needed for the slip system {010}[001]:
def tetrrotation(theta):
    return np.round(Roty(np.pi-theta),15)

### thermal coefficients:
alpha_a = data.CRC_alpha_a  ## coefficient of linear thermal expansion at room temperature
## TODO: need to implement T dependence of alpha_a!

#########
if __name__ == '__main__':
    if len(sys.argv) > 1:
        ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
        metal = sys.argv[1].split()
    
    if Ncores == 0:
        print("skipping phonon wind calculations as requested")
    else:
        with open("beta.dat","w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta)))
    
        with open("theta.dat","w") as thetafile:
            thetafile.write('\n'.join(map("{:.6f}".format,theta)))
    
        with open("temperatures.dat","w") as Tfile:
            Tfile.write('\n'.join(map("{:.1f}".format,highT)))  
        
        print("Computing the drag coefficient from phonon wind ({} modes) for: ".format(modes),metal)
    
    ## compute rotation matrices for later use
    for X in data.fcc_metals.intersection(metal):
        rotmat[X] = np.zeros((len(theta),3,3))
        for th in range(len(theta)):
            rotmat[X][th] = fccrotation(theta[th])

    for X in data.bcc_metals.intersection(metal):
        rotmat[X] = np.zeros((len(theta),3,3))
        for th in range(len(theta)):
            rotmat[X][th] = bccrotation(theta[th])
            
    for X in data.hcp_metals.intersection(metal):
        rotmat[X] = np.zeros((len(theta),3,3))
        if hcpslip=='prismatic':
            for th in range(len(theta)):
                rotmat[X][th] = hcprotation_pris(theta[th])
        elif hcpslip=='pyramidal':
            for th in range(len(theta)):
                rotmat[X][th] = hcprotation_pyr(theta[th],X)
        else: ## default = basal
            for th in range(len(theta)):
                rotmat[X][th] = hcprotation(theta[th])
            
    for X in data.tetr_metals.intersection(metal):
        rotmat[X] = np.zeros((len(theta),3,3))
        for th in range(len(theta)):
            rotmat[X][th] = tetrrotation(theta[th])
    ###
    r = [rmin*np.pi,rmax*np.pi] ## qBZ drops out of product q*r, so can rescale both vectors making them dimensionless and independent of the metal
    q = np.linspace(0,1,Nq)
    ## needed for the Fourier transform of uij (but does not depend on beta or T, so we compute it only once here)
    # sincos = dlc.fourieruij_sincos(r,phiX,q,phi)
    ## for use with fourieruij_nocut(), which is faster than fourieruij() if cutoffs are chosen such that they are neglegible in the result:
    sincos_noq = np.average(dlc.fourieruij_sincos(r,phiX,q,phi)[3:-4],axis=0)
        
    A3rotated = {}
    C2 = {}
    Cv = {}
    M = {}
    N = {}
    for X in metal:
        geometry = dlc.StrohGeometry(b=b[X], n0=n0[X], theta=theta, phi=phiX)
        Cv[X] = geometry.Cv
        M[X] = geometry.M
        N[X] = geometry.N
        linet[X] = np.round(geometry.t,15)
        velm0[X] = np.round(geometry.m0,15)
               
        C2[X] = elasticC2(c11=c11[X], c12=c12[X], c44=c44[X], c13=c13[X], c33=c33[X], c66=c66[X])/mu[X]  ## this must be the same mu that was used to define the dimensionless velocity beta, as both enter dlc.computeuij() on equal footing below!
        C3 = elasticC3(c111=c111[X], c112=c112[X], c113=c113[X], c123=c123[X], c133=c133[X], c144=c144[X], c155=c155[X], c166=c166[X], c222=c222[X], c333=c333[X], c344=c344[X], c366=c366[X], c456=c456[X])/mu[X]
        A3 = elasticA3(C2[X],C3)
        A3rotated[X] = np.zeros((len(theta),3,3,3,3,3,3))
        for th in range(len(theta)):
            A3rotated[X][th] = np.round(np.einsum('ab,cd,ef,gh,ik,lm,bdfhkm',rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],A3),12)
        
    for X in metal:
        # r = np.exp(np.linspace(np.log(burgers[X]/5),np.log(100*burgers[X]),125))
        ## perhaps better: relate directly to qBZ which works for all crystal structures (rmin/rmax defined at the top of this file)
        # r = np.exp(np.linspace(np.log(rmin*np.pi/qBZ[X]),np.log(rmax*np.pi/qBZ[X]),Nr))
        # q = qBZ[X]*np.linspace(0,1,Nq)
    
        # wrap all main computations into a single function definition to be run in a parallelized loop below
        def maincomputations(bt,X,modes=modes):
            Bmix = np.zeros((len(theta),len(highT)))
                                    
            ### compute dislocation displacement gradient uij, then its Fourier transform dij:
            uij = dlc.computeuij(beta=bt, C2=C2[X], Cv=Cv[X], b=b[X], M=M[X], N=N[X], phi=phiX)
            # uij_iso = dlc.computeuij_iso(bt,ct_over_cl[X], theta, phiX)
            uijrotated = np.zeros(uij.shape)
            for th in range(len(theta)):
                uijrotated[:,:,th] = np.round(np.dot(rotmat[X][th],np.dot(rotmat[X][th],uij[:,:,th])),15)
            
            # dij = np.average(dlc.fourieruij(uijrotated,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
            # dij = dlc.fourieruij_nocut(uijrotated,phiX,phi,r_reg)
            dij = dlc.fourieruij_nocut(uijrotated,phiX,phi,sincos=sincos_noq)
            
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3rotated[X], qBZ=qBZ[X], ct=ct[X], cl=cl[X], beta=bt, burgers=burgers[X], T=roomT, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
            
            for Ti in range(len(highT)-1):
                T = highT[Ti+1]
                qBZT = qBZ[X]/(1 + alpha_a[X]*(T - roomT))
                muT = mu[X] ## TODO: need to implement T dependence of shear modulus!
                lamT = bulk[X] - 2*muT/3 ## TODO: need to implement T dependence of bulk modulus!
                rhoT = rho[X]/(1 + alpha_a[X]*(T - roomT))**3
                burgersT = burgers[X]*(1 + alpha_a[X]*(T - roomT))
                ctT = np.sqrt(muT/rhoT)
                ct_over_cl_T = np.sqrt(muT/(lamT+2*muT))
                clT = ctT/ct_over_cl_T
                ## beta, as it appears in the equations, is v/ctT, therefore:
                betaT = bt*ct[X]/ctT
                    
                ###### T dependence of elastic constants (TODO)
                c11T = c11[X]
                c12T = c12[X]
                c44T = c44[X]
                c13T = c13[X]
                c33T = c33[X]
                c66T = c66[X]
                c111T = c111[X]
                c112T = c112[X]
                c113T = c113[X]
                c123T = c123[X]
                c133T = c133[X]
                c144T = c144[X]
                c155T = c155[X]
                c166T = c166[X]
                c222T = c222[X]
                c333T = c333[X]
                c344T = c344[X]
                c366T = c366[X]
                c456T = c456[X]
                ###
                C2T = elasticC2(c11=c11T, c12=c12T, c44=c44T, c13=c13T, c33=c33T, c66=c66T)/muT
                C3T = elasticC3(c111=c111T, c112=c112T, c113=c113T, c123=c123T, c133=c133T, c144=c144T, c155=c155T, c166=c166T, c222=c222T, c333=c333T, c344=c344T, c366=c366T, c456=c456T)/muT
                A3T = elasticA3(C2T,C3T)
                A3Trotated = np.zeros((len(theta),3,3,3,3,3,3))
                for th in range(len(theta)):
                    A3Trotated[th] = np.round(np.einsum('ab,cd,ef,gh,ik,lm,bdfhkm',rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],A3T),12)
                ##########################
                uij = dlc.computeuij(beta=betaT, C2=C2T, Cv=Cv[X], b=b[X], M=M[X], N=N[X], phi=phiX)
                uijrotated = np.zeros(uij.shape)
                for th in range(len(theta)):
                    uijrotated[:,:,th] = np.round(np.dot(rotmat[X][th],np.dot(rotmat[X][th],uij[:,:,th])),15)
                                    
                ## rT*qT = r*q, so does not change anything
                # dij = np.average(dlc.fourieruij(uijrotated,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
                dij = dlc.fourieruij_nocut(uijrotated,phiX,phi,sincos=sincos_noq)
            
                Bmix[:,Ti+1] = dragcoeff_iso(dij=dij, A3=A3Trotated, qBZ=qBZT, ct=ctT, cl=clT, beta=betaT, burgers=burgersT, T=T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
                    
            return Bmix

        # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
        if Ncores == 1:
            Bmix = np.array([maincomputations(bt,X,modes) for bt in beta])
        elif Ncores == 0:
            pass
        else:
            Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt,X,modes) for bt in beta))


        # and write the results to disk (in various formats)
        if Ncores != 0:
            with open("drag_anis_{}.dat".format(X),"w") as Bfile:
                Bfile.write("### B(beta,theta) for {} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT)>1 and Ncores !=0:
            with open("drag_anis_T_{}.dat".format(X),"w") as Bfile:
                Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,theta[1:-1])) + '\tBedge[mPas]' + '\n')
                for bi in range(len(beta)):
                    for Ti in range(len(highT)):
                        Bfile.write("{:.1f}".format(highT[Ti]) +'\t' + "{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')
                
            with open("drag_anis_T_screw_{}.dat".format(X),"w") as Bscrewfile:
                for bi in range(len(beta)):
                    Bscrewfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,0])) + '\n')
            
            with open("drag_anis_T_edge_{}.dat".format(X),"w") as Bedgefile:
                for bi in range(len(beta)):
                    Bedgefile.write('\t'.join(map("{:.6f}".format,Bmix[bi,-1])) + '\n')
    
            for th in range(len(theta[1:-1])):
                with open("drag_anis_T_mix{0:.6f}_{1}.dat".format(theta[th+1],X),"w") as Bmixfile:
                    for bi in range(len(beta)):
                        Bmixfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,th+1])) + '\n')

    #############################################################################################################################

    if skip_plots:
        print("skipping plots as requested")
        sys.exit()
    
    ###### plot room temperature results:
    print("Creating plots")
    ## compute smallest critical velocity in ratio to the scaling velocity and plot only up to this velocity
    vcrit_smallest = {}
    for X in metal:
        vcrit_smallest[X] = min(np.sqrt(c44[X]/mu[X]),np.sqrt((c11[X]-c12[X])/(2*mu[X])))
    ## above is correct for fcc, bcc and some (but not all) hcp, i.e. depends on values of SOEC and which slip plane is considered;
    ## numerically determined values (rounded):
    vcrit_smallest['Sn'] = 0.818
    vcrit_smallest['Zn'] = 0.943 ## for basal slip
    # vcrit for pure screw/edge for default slip systems (incl. basal for hcp), numerically determined values (rounded):
    vcrit_screw = {'Ag': 0.973, 'Al': 1.005, 'Au': 0.996, 'Cd': 1.398, 'Cu': 0.976, 'Fe': 0.803, 'Mg': 0.982, 'Mo': 0.987, 'Nb': 0.955, 'Ni': 1.036, 'Sn': 1.092, 'Zn': 1.211, 'Zr': 0.990}
    vcrit_edge = {'Fe': 0.852, 'Mo': 1.033, 'Nb': 1.026, 'Sn': 1.092, 'Ti': 1.033}
    for X in data.fcc_metals.union(data.hcp_metals).intersection(metal):
        if X in ['Ti']:
            vcrit_screw[X] = vcrit_smallest[X]
        else:
            vcrit_edge[X] = vcrit_smallest[X] ## coincide for the fcc slip system considered above, and for most hcp-basal slip systems

    if hcpslip=='prismatic':
        for X in data.hcp_metals.intersection(metal):
            if X in ['Ti']:
                vcrit_screw[X] = vcrit_edge[X]
                vcrit_edge[X] = vcrit_smallest[X]
            else:
                vcrit_edge[X] = vcrit_screw[X]
                vcrit_screw[X] = vcrit_smallest[X]
        vcrit_screw['Zn'] = 0.945
        vcrit_smallest['Cd'] = 0.948
        vcrit_smallest['Zn'] = 0.724
    elif hcpslip=='pyramidal':
        vcrit_screw['Cd'] = 1.278
        vcrit_screw['Mg'] = 0.979
        vcrit_screw['Ti'] = 0.930
        vcrit_screw['Zn'] = 1.132
        vcrit_screw['Zr'] =0.976
        vcrit_edge['Ti'] = vcrit_smallest['Ti']
        vcrit_edge['Zn'] = 0.945
        vcrit_smallest['Cd'] = 0.975
        vcrit_smallest['Zn'] = 0.775
    
    ## load data from semi-isotropic calculation
    Broom = {}
    theta = {}
    Ntheta = {}
    for X in metal:
        ## for every X, Broom has shape (len(theta+1),Nbeta), first column is beta all others all B for various dislocation types theta in the range 0 to  pi/2
        with open("drag_anis_{}.dat".format(X),"r") as Bfile:
            lines = list(line.rstrip() for line in Bfile)
            ### first read theta from file (already known, but make this code independent from above)
            theta[X] = np.pi*np.asarray(lines[1].split()[1:],dtype='float')
            Ntheta[X] = len(theta[X])
            ### determine length of beta from file
            Nbeta = len(lines)-2
            ### read beta vs drag coeff from file:
            Broom[X] = np.zeros((Nbeta,Ntheta[X]+1))
            for j in range(Nbeta):
                Broom[X][j] = np.asarray(lines[j+2].split(),dtype='float')
            beta = Broom[X][:,0]
    
    ## plot B(beta=0.01 ) against theta for every metal:
    def mksmallbetaplot(X,ylab=True,xlab=True,bt=0):
        beta_trunc = [j for j in beta if j <=vcrit_smallest[X]]
        B_trunc = (Broom[X][:len(beta_trunc),1:])
        ymax = max(0.01,(int(100*max(B_trunc[bt]))+1)/100)
        ymin = (int(100*min(B_trunc[bt])))/100
        ## if range is too large, cut off top/bottom:
        if ymax-ymin>=0.05:
            ymax = ymax-0.006
            ymin = ymin+0.006
        plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2,5*np.pi/8,3*np.pi/4,7*np.pi/8,np.pi],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$", r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"),fontsize=fntsize)
        plt.axis((theta[X][0],theta[X][-1],ymin,ymax))
        plt.yticks(fontsize=fntsize)
        if xlab==True:
            plt.xlabel(r'$\vartheta$',fontsize=fntsize)
        if ylab==True:
            plt.ylabel(r'$B(\beta_\mathrm{t}=0.01)$',fontsize=fntsize)
            plt.ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        plt.plot(theta[X],B_trunc[bt])
    
    ### create colormesh-plots for every metal:
    clbar_frac=0.12
    clbar_pd=0.03
    wrat1=1-clbar_frac-clbar_pd
    wspc=(clbar_frac+clbar_pd)*100/wrat1
    def mkmeshplot(X,ylab=True,xlab=True,colbar=True,Bmin=None,Bmax=None):
        beta_trunc = [j for j in beta if j <=vcrit_smallest[X]]
        B_trunc = (Broom[X][:len(beta_trunc),1:]).T
        y_msh , x_msh = np.meshgrid(beta_trunc,theta[X]) ## plots against theta and beta
        if Bmin==None:
            Bmin = (int(1000*np.min(B_trunc)))/1000
        if Bmax==None:
            Bmax = Bmin+0.016
            ## tweak colorbar range defined above:
            if np.sum(B_trunc<=Bmax)/(Ntheta[X]*len(beta_trunc))<0.65:
                Bmax = Bmin+0.032 ## if more than 35% of the area is >Bmax, double the range
            elif np.sum(B_trunc>Bmax)/(Ntheta[X]*len(beta_trunc))<0.02:
                Bmax = Bmin+0.008 ## if less than 2% of the area is >Bmax, cut the range in half
        namestring = "{}".format(X)
        plt.xticks(fontsize=fntsize)
        plt.yticks(np.arange(10)/10,fontsize=fntsize)
        cbarlevels = list(np.linspace(Bmin,Bmax,9))
        if xlab==True:
            plt.xlabel(r'$\vartheta$',fontsize=fntsize)
        if ylab==True:
            plt.ylabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        plt.title(namestring,fontsize=fntsize)
        colmsh=plt.pcolormesh(x_msh,y_msh,B_trunc,vmin=Bmin, vmax=Bmax,cmap = plt.cm.cubehelix_r)
        colmsh.set_rasterized(True)
        if colbar==True:
            cbar = plt.colorbar(fraction=clbar_frac,pad=clbar_pd, ticks=cbarlevels)
            cbar.set_label(r'$B$[mPa$\,$s]', labelpad=-22, y=1.11, rotation=0, fontsize = fntsize)
            cbar.ax.tick_params(labelsize = fntsize)
        plt.contour(x_msh,y_msh,B_trunc, colors=('gray','gray','gray','white','white','white','white','white','white'), levels=cbarlevels, linewidths=[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9], linestyles=['dashdot','solid','dashed','dotted','dashdot','solid','dashed','dotted','dashdot'])
        
    for X in metal:
        fig = plt.figure(figsize=(4.5,3.6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,0.25]) 
        ax0 = fig.add_subplot(gs[0])
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.yaxis.set_minor_locator(AutoMinorLocator())
        plt.setp(ax0.get_xticklabels(), visible=False)
        mkmeshplot(X,ylab=True,xlab=False,colbar=True,Bmin=None,Bmax=None)
        ax1 = fig.add_subplot(gs[1] , sharex=ax0)
        ax1.set_yticks(np.arange(11)/100)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        mksmallbetaplot(X,ylab=True,xlab=True)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="{}%".format(wspc))
        if mpl.__version__ >= '2.0.0':
            cax.set_facecolor('none')
        for axis in ['top','bottom','left','right']:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.tight_layout(pad=0.3)
        plt.savefig("B_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=300)
        plt.close()
    
    ## define line styles for every metal in the same plot
    lnstyles = {'Al':'-', 'Cu':'--', 'Fe':':', 'Nb':'-.', 'Cd':'-', 'Mg':'--', 'Zn':':', 'Sn':'-.', 'Ni':'-.', 'Mo':'--', 'Ag':':', 'Au':'-.', 'Ti':'-', 'Zr':'-.'}
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe':'darkgreen', 'Nb':'firebrick', 'Zn':'purple', 'Sn':'black', 'Ag':'lightblue', 'Au':'goldenrod', 'Cd':'lightgreen', 'Mg':'lightsalmon', 'Mo':'magenta', 'Ni':'silver', 'Ti':'olive', 'Zr':'cyan'}
        
    ## define fitting fct.:
    def fit_mix(x, c0, c1, c2, c4):
        return c0 - c1*x + c2*(1/(1-x**2)**(1/2) - 1) + c4*(1/(1-x**2)**(3/2) - 1)

    beta_edgecrit = {}
    beta_screwcrit = {}
    beta_avercrit = {}
    Baver = {}
    popt_edge = {}
    pcov_edge = {}
    popt_screw = {}
    pcov_screw = {}
    popt_aver = {}
    pcov_aver = {}
    scrind = {}
    Bmax_fit = 0.20 ## only fit up to Bmax_fit [mPas]
    for X in metal:
        if X in data.bcc_metals and (np.all(theta[X]>=0) or np.all(theta[X]<=0)):
            print("warning: missing data for a range of dislocation character angles of bcc {}, average will be inaccurate!".format(X))
        if theta[X][0]==0.:
            scrind[X] = 0
        else:
            scrind[X] = int(Ntheta[X]/2)
        Baver[X] = np.average(Broom[X][:,1:],axis=-1)        
        beta_edgecrit[X] = (beta/vcrit_edge[X])[beta<vcrit_edge[X]]
        beta_screwcrit[X] = (beta/vcrit_screw[X])[beta<vcrit_screw[X]]
        beta_avercrit[X] =  (beta/vcrit_smallest[X])[beta<vcrit_smallest[X]]
        ### having cut everything beyond the critical velocities (where B diverges), we additionally remove very high values (which are likely inaccurate close to vcrit) to improve the fits everywhere else; adjust Bmax_fit to your needs!
        beta_edgecrit[X] = beta_edgecrit[X][[j for j in range(len(beta_edgecrit[X])) if Broom[X][j,Ntheta[X]] <Bmax_fit + np.min(Broom[X][:,Ntheta[X]])]]
        beta_screwcrit[X] = beta_screwcrit[X][[j for j in range(len(beta_screwcrit[X])) if Broom[X][j,scrind[X]+1]<Bmax_fit + np.min(Broom[X][:,scrind[X]+1])]]
        beta_avercrit[X] =  beta_avercrit[X][[j for j in range(len(beta_avercrit[X])) if Baver[X][j]<Bmax_fit + np.min(Baver[X])]]
        popt_edge[X], pcov_edge[X] = curve_fit(fit_mix, beta_edgecrit[X][beta_edgecrit[X]<0.995], (Broom[X][:len(beta_edgecrit[X])])[beta_edgecrit[X]<0.995,Ntheta[X]], bounds=([0.9*Broom[X][0,Ntheta[X]],0.,-0.,-0.], [1.1*Broom[X][0,Ntheta[X]], 2*Broom[X][0,Ntheta[X]], 1., 1.]))
        popt_screw[X], pcov_screw[X] = curve_fit(fit_mix, beta_screwcrit[X][beta_screwcrit[X]<0.995], (Broom[X][:len(beta_screwcrit[X])])[beta_screwcrit[X]<0.995,scrind[X]+1], bounds=([0.9*Broom[X][0,scrind[X]+1],0.,-0.,-0.], [1.1*Broom[X][0,scrind[X]+1], 2*Broom[X][0,scrind[X]+1], 1., 1.]))
        popt_aver[X], pcov_aver[X] = curve_fit(fit_mix, beta_avercrit[X][beta_avercrit[X]<0.995], (Baver[X][:len(beta_avercrit[X])])[beta_avercrit[X]<0.995], bounds=([0.9*Baver[X][0],0.,-0.,-0.], [1.1*Baver[X][0], 2*Baver[X][0], 1., 1.]))
    
    with open("drag_semi_iso_fit.txt","w") as fitfile:
        fitfile.write("Fitting functions for B[$\mu$Pas] at room temperature:\nEdge dislocations:\n")
        for X in metal:
            fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*(1/(1-x**2)**(1/2) - 1) + {3:.2f}*(1/(1-x**2)**(3/2) - 1)\n".format(*1e3*popt_edge[X]))
        fitfile.write("\nScrew dislocations:\n")
        for X in metal:
            fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*(1/(1-x**2)**(1/2) - 1) + {3:.2f}*(1/(1-x**2)**(3/2) - 1)\n".format(*1e3*popt_screw[X]))
        fitfile.write("\nAveraged over all characters:\n")
        for X in metal:
            fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*(1/(1-x**2)**(1/2) - 1) + {3:.2f}*(1/(1-x**2)**(3/2) - 1)\n".format(*1e3*popt_aver[X]))
        fitfile.write("\n\nwhere $x=v/v_c$ with:\n\n")
        fitfile.write(" & "+" & ".join((metal))+" \\\\\hline\hline")
        fitfile.write("\n $c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(" & "+"{:.0f}".format(ct[X]))
        fitfile.write(" \\\\\n $v_c^{\mathrm{e}}/c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(" & "+"{:.3f}".format(vcrit_edge[X]))
        fitfile.write("\n\\\\\hline\hline")
        fitfile.write("\n $v_c^{\mathrm{s}}/c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(" & "+"{:.3f}".format(vcrit_screw[X]))
        fitfile.write("\n\\\\\hline\hline")
        fitfile.write("\n $v_c^{\mathrm{av}}/c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(" & "+"{:.3f}".format(vcrit_smallest[X]))
        
    def mkfitplot(metal_list,filename,figtitle):
        fig, ax = plt.subplots(1, 1, figsize=(5.5,5.5))
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.set_xticks(np.arange(11)/10)
        ax.set_yticks(np.arange(12)/100)
        ax.axis((0,maxb,0,0.11)) ## define plot range
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        ax.set_ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        ax.set_title(figtitle,fontsize=fntsize)
        for X in metal_list:
            if filename=="edge":
                vcrit = vcrit_edge[X]
                popt = popt_edge[X]
                cutat = 1.001*vcrit ## vcrit is often rounded, so may want to plot one more point in some cases
                B = Broom[X][beta<cutat,Ntheta[X]]
            elif filename=="screw":
                vcrit = vcrit_screw[X]
                popt = popt_screw[X]
                cutat = 1.0*vcrit
                B = Broom[X][beta<cutat,scrind[X]+1]
            elif filename=="aver":
                vcrit = vcrit_smallest[X]
                popt = popt_aver[X]
                cutat = 1.007*vcrit
                B = Baver[X][beta<cutat]
            else:
                raise ValueError("keyword 'filename'={} undefined.".format(filename))
            ax.plot(beta[beta<cutat],B,color=metalcolors[X],label=X)
            beta_highres = np.linspace(0,vcrit,1000)
            ax.plot(beta_highres,fit_mix(beta_highres/vcrit,*popt),':',color='gray')
        ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig("B_{0}K_{1}+fits.pdf".format(roomT,filename),format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge","pure edge")
    mkfitplot(metal,"screw","pure screw")
    mkfitplot(metal,"aver","averaged over $\\vartheta$")
    