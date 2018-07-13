# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Los Alamos National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - June 25, 2018
#################################
from __future__ import division
from __future__ import print_function

import sys
### make sure we are running a recent version of python
# assert sys.version_info >= (3,5)
import numpy as np
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
from joblib import Parallel, delayed
## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
Ncores = -2

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
# SOEC taken from various refs.:
# c11 = data.THLPG_c11
# c12 = data.THLPG_c12
# c44 = data.THLPG_c44
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
# c13 = {}
# c33 = {}
# c66 = {}
# c113 = {}
# c133 = {}
# c155 = {}
# c222 = {}
# c333 = {}
# c344 = {}
# c366 = {}

mu = data.ISO_c44 ## effective shear modulus of polycrystal
lam = data.ISO_c12
mu['Mo'] = 125.0e9 ## use the 'improved' average for polycrystalline Mo since we don't have experimental data (see results of 'polycrystal_averaging.py')
lam['Mo'] = 176.4e9

qBZ = {}
ct = {} ## may need some "effective" transverse sound speed
cl = {}
ct_over_cl = {}
burgers = {} # magnitude of Burgers vector
bulk = {}

### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. at least TOEC)
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(c111.keys()).intersection(mu.keys())))
metal_cubic = data.fcc_metals.union(data.bcc_metals).intersection(metal)

### set "None" non-independent elastic constants
### and compute various numbers for these metals
for X in metal_cubic:
    c13[X] = None
    c33[X] = None
    c66[X] = None
    c113[X] = None
    c133[X] = None
    c155[X] = None
    c222[X] = None
    c333[X] = None
    c344[X] = None
    c366[X] = None
    qBZ[X] = ((6*np.pi**2)**(1/3))/ac[X]
###
for X in data.hcp_metals.intersection(metal):
    c66[X] = None
    c166[X] = None
    c366[X] = None
    c456[X] = None
    qBZ[X] = ((4*np.pi**2/(ac[X]*ac[X]*cc[X]*np.sqrt(3)))**(1/3))
for X in data.tetr_metals.intersection(metal):
    c222[X] = None
    qBZ[X] = ((6*np.pi**2/(ac[X]*ac[X]*cc[X]))**(1/3))
###
for X in metal:
    ## want to scale everything by the average (polycrystalline) shear modulus and thereby calculate in dimensionless quantities
    ct[X] = np.sqrt(mu[X]/rho[X])
    cl[X] = np.sqrt((lam[X]+2*mu[X])/rho[X])
    bulk[X] = lam[X] + 2*mu[X]/3
    ct_over_cl[X] = ct[X]/cl[X]
    # ct_over_cl[X] = np.sqrt(mu[X]/(c12[X]+2*c44[X])) ## for isotropic check

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
for X in data.hcp_metals.intersection(metal):
    burgers[X] = ac[X]
    b[X] = np.array([-1,0,0])
    n0[X] = np.array([0,0,1]) ## slip plane normal = normal to basal plane
    ### TODO: include also other slip systems
    
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
    
### rotation needed for the slip system {001}[100]:
def hcprotation(theta):
    return np.round(np.dot(Roty(-np.pi/2-theta),Rotx(np.pi/2)),15)
    ### TODO: include also other slip systems (above is for basal slip with b in -x direction)
    
### rotation needed for the slip system {010}[001]:
def tetrrotation(theta):
    return np.round(Roty(np.pi-theta),15)

### thermal coefficients:
alpha_a = data.CRC_alpha_a  ## coefficient of linear thermal expansion at room temperature
## TODO: need to implement T dependence of alpha_a!

#########
if __name__ == '__main__':
    with open("beta.dat","w") as betafile:
        betafile.write('\n'.join(map("{:.5f}".format,beta)))

    with open("theta.dat","w") as thetafile:
        thetafile.write('\n'.join(map("{:.6f}".format,theta)))          

    with open("temperatures.dat","w") as Tfile:
        Tfile.write('\n'.join(map("{:.1f}".format,highT)))  
            
    if len(sys.argv) > 1:
        ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
        metal = sys.argv[1].split()
        
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
        
    for X in metal:
        geometry = dlc.StrohGeometry(b=b[X], n0=n0[X], theta=theta, phi=phiX)
        Cv = geometry['Cv']
        M = geometry['M']
        N = geometry['N']
        linet[X] = np.round(geometry['t'],15)
        velm0[X] = np.round(geometry['m0'],15)
               
        C2 = elasticC2(c11=c11[X], c12=c12[X], c44=c44[X], c13=c13[X], c33=c33[X], c66=c66[X])/mu[X]  ## this must be the same mu that was used to define the dimensionless velocity beta, as both enter dlc.computeuij() on equal footing below!
        C3 = elasticC3(c111=c111[X], c112=c112[X], c113=c113[X], c123=c123[X], c133=c133[X], c144=c144[X], c155=c155[X], c166=c166[X], c222=c222[X], c333=c333[X], c344=c344[X], c366=c366[X], c456=c456[X])/mu[X]
        A3 = elasticA3(C2,C3)
        A3rotated = np.zeros((len(theta),3,3,3,3,3,3))
        for th in range(len(theta)):
            A3rotated[th] = np.round(np.einsum('ab,cd,ef,gh,ik,lm,bdfhkm',rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],rotmat[X][th],A3),12)
        
        # r = np.exp(np.linspace(np.log(burgers[X]/5),np.log(100*burgers[X]),125))
        ## perhaps better: relate directly to qBZ which works for all crystal structures (rmin/rmax defined at the top of this file)
        # r = np.exp(np.linspace(np.log(rmin*np.pi/qBZ[X]),np.log(rmax*np.pi/qBZ[X]),Nr))
        # q = qBZ[X]*np.linspace(0,1,Nq)
    
        # wrap all main computations into a single function definition to be run in a parallelized loop below
        def maincomputations(bt,modes=modes):
            Bmix = np.zeros((len(theta),len(highT)))
                                    
            ### compute dislocation displacement gradient uij, then its Fourier transform dij:
            uij = dlc.computeuij(beta=bt, C2=C2, Cv=Cv, b=b[X], M=M, N=N, phi=phiX)
            # uij_iso = dlc.computeuij_iso(bt,ct_over_cl[X], theta, phiX)
            uijrotated = np.zeros(uij.shape)
            for th in range(len(theta)):
                uijrotated[:,:,th] = np.round(np.dot(rotmat[X][th],np.dot(rotmat[X][th],uij[:,:,th])),15)
            
            # dij = np.average(dlc.fourieruij(uijrotated,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
            # dij = dlc.fourieruij_nocut(uijrotated,phiX,phi,r_reg)
            dij = dlc.fourieruij_nocut(uijrotated,phiX,phi,sincos=sincos_noq)
            
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3rotated, qBZ=qBZ[X], ct=ct[X], cl=cl[X], beta=bt, burgers=burgers[X], T=roomT, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
            
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
                uij = dlc.computeuij(beta=betaT, C2=C2T, Cv=Cv, b=b[X], M=M, N=N, phi=phiX)
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
            Bmix = np.array([maincomputations(bt,modes) for bt in beta])
        else:
            Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt,modes) for bt in beta))


        # and write the results to disk (in various formats)
        with open("drag_anis_{}.dat".format(X),"w") as Bfile:
            Bfile.write("### B(beta,theta) for {} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
            Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,theta/np.pi)) + '\n')
            for bi in range(len(beta)):
                Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT)>1:
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


    ###### plot room temperature results:
    ## compute smallest critical velocity in ratio to the scaling velocity and plot only up to this velocity
    vcrit_smallest = {}
    for X in metal:
        vcrit_smallest[X] = min(np.sqrt(c44[X]/mu[X]),np.sqrt((c11[X]-c12[X])/(2*mu[X])))
    ## above is correct for fcc, bcc and some (but not all) hcp, otherwise need to determine numerically as in:
    vcrit_smallest['Sn'] = 0.8181 ## rounded, determined numerically
    vcrit_smallest['Zn'] = 0.9431 ## rounded, determined numerically
    
    ## load data from semi-isotropic calculation
    Broom = {}
    for X in metal:
        ## for every X, Broom has shape (len(theta+1),Nbeta), first column is beta all others all B for various dislocation types theta in the range 0 to  pi/2
        Broom[X] = np.zeros((Nbeta,Ntheta+1))
        with open("drag_anis_{}.dat".format(X),"r") as Bfile:
            lines = list(line.rstrip() for line in Bfile)
            ### first read theta from file (already known, but make this code independent from above)
            theta = np.pi*np.asarray(lines[1].split()[1:],dtype='float')
            Ntheta = len(theta)
            ### determine length of beta from file
            Nbeta = len(lines)-2
            ### read beta vs drag coeff from file:
            Broom[X] = np.zeros((Nbeta,Ntheta+1))
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
        plt.axis((theta[0],theta[-1],ymin,ymax))
        plt.yticks(fontsize=fntsize)
        if xlab==True:
            plt.xlabel(r'$\vartheta$',fontsize=fntsize)
        if ylab==True:
            plt.ylabel(r'$B(\beta_\mathrm{t}=0.01)$',fontsize=fntsize)
            plt.ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        plt.plot(theta,B_trunc[bt])
    
    ### create colormesh-plots for every metal:
    clbar_frac=0.12
    clbar_pd=0.03
    wrat1=1-clbar_frac-clbar_pd
    wspc=(clbar_frac+clbar_pd)*100/wrat1
    def mkmeshplot(X,ylab=True,xlab=True,colbar=True,Bmin=None,Bmax=None):
        beta_trunc = [j for j in beta if j <=vcrit_smallest[X]]
        B_trunc = (Broom[X][:len(beta_trunc),1:]).T
        y_msh , x_msh = np.meshgrid(beta_trunc,theta) ## plots against theta and beta
        if Bmin==None:
            Bmin = (int(1000*np.min(B_trunc)))/1000
        if Bmax==None:
            Bmax = Bmin+0.016
            ## tweak colorbar range defined above:
            if np.sum(B_trunc<=Bmax)/(Ntheta*len(beta_trunc))<0.65:
                Bmax = Bmin+0.032 ## if more than 35% of the area is >Bmax, double the range
            elif np.sum(B_trunc>Bmax)/(Ntheta*len(beta_trunc))<0.02:
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
        cax.set_facecolor('none')
        for axis in ['top','bottom','left','right']:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.tight_layout(pad=0.3)
        plt.savefig("B_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=300)
        plt.close()
    
