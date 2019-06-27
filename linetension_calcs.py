# -*- coding: utf-8 -*-
# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - June 26, 2019
#################################
from __future__ import division
from __future__ import print_function

import sys
### make sure we are running a recent version of python
# assert sys.version_info >= (3,5)
#################################
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
plt.rc('font',**{'family':'Liberation Serif','size':'11'})
# plt.rc('font',**{'family':'Liberation Sans Narrow','size':'11'})
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import metal_data as data
try:
    from joblib import Parallel, delayed
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = -2
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = 1 ## must be 1 without joblib

### choose resolution of discretized parameters: theta is the angle between disloc. line and Burgers vector, beta is the dislocation velocity,
### and phi is an integration angle used in the integral method for computing dislocations
Ntheta = 600
Ntheta2 = 21 ## number of character angles for which to calculate critical velocities (set to None or 0 to bypass entirely)
Nbeta = 625
Nphi = 1000
### and range & step sizes
dtheta = np.pi/(Ntheta-2)
theta = np.linspace(-np.pi/2-dtheta,np.pi/2+dtheta,Ntheta+1)
beta = np.linspace(0,1,Nbeta+1)
phi = np.linspace(0,2*np.pi,Nphi)
#####

#### input data:
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals)))
metal = sorted(metal + ['ISO']) ### test isotropic limit

# 2nd order elastic constants taken from the CRC handbook:
c11 = data.CRC_c11
c12 = data.CRC_c12
c44 = data.CRC_c44
c13 = data.CRC_c13
c33 = data.CRC_c33
c66 = data.CRC_c66
## lattice constants needed for normalization of some slip plane normals
ac = data.CRC_a
cc = data.CRC_c

### define some isotropic constants to check isotropic limit:
c44['ISO'] = 1
nu = 1/3 ## define Poisson's ratio
c12['ISO'] = round(c44['ISO']*2*nu/(1-2*nu),2)
c11['ISO'] = c12['ISO']+2*c44['ISO']
c13['ISO'] = None
c33['ISO'] = None
c66['ISO'] = None

## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
mu = {}
for X in c11.keys():
    mu[X] = (c11[X]-c12[X]+2*c44[X])/4
    ### for hexagonal/tetragonal metals, this corresponds to the average shear modulus in the basal plane (which is the most convenient for the present calculations)
    
# we limit calculations to a velocity range close to what we actually need,
# i.e. the scaling factors are rounded up from the largest critical velocities that we encounter,
# namely the vcrit for screw disloc. for fcc metals and vcrit for edge disloc. for bcc metals
beta_scaled = {'Al':0.98*beta, 'Cu':0.85*beta,  'Ni':0.88*beta, 'Fe':0.87*beta, 'Nb':0.97*beta, 'Ta':0.95*beta, 'W':beta}
## for metals included in the list "metal" but not in "beta_scaled", use default scaling until the user provides a better scaling factor,
## i.e. ratio of smallest transverse soundspeed for pure screw in fcc and pure egde in bcc to effective one computed from average mu which we use below for all scalings
## (or 1 if that is the smaller number); in general, this automatic scaling will differ from the optimal user compute one only by a few percent

### ALTERNATIVE SCALING: use average shear modulus of polycrystal (uncomment block below to use this version):
# mu = data.ISO_c44
# mu['Mo'] = 125.0e9 ## use the 'improved' average for polycrystalline Mo, K since we don't have experimental data (see results of 'polycrystal_averaging.py')
# mu['K'] = 0.9e9
# mu['Be'] = 148.6e9 ## use Hill average for polycrystalline Be, In, Zr (see results of 'polycrystal_averaging.py')
# mu['In'] = 4.8e9
# mu['Zr'] = 36.0e9
# mu['ISO'] = 1
# metal = sorted(list(set(metal).intersection(mu.keys()))) ## update list of metals since we may not have mu for all elements
# beta_scaled = {} # reset
###########################################################

## for hexagonal/tetragonal crystals, we use scaling=1 for now
scaling = {}
autoscale = set(metal).difference(beta_scaled.keys())
for X in autoscale:
    beta_scaled[X] = beta

for X in autoscale.intersection(data.fcc_metals):
    scaling[X] = min(1,round(np.sqrt((5*c11[X]+c12[X]+8*c44[X]-np.sqrt(9*c11[X]**2+33*c12[X]**2+72*c12[X]*c44[X]+48*c44[X]**2-6*c11[X]*(c12[X]+4*c44[X])))/(12*mu[X])),2))
    beta_scaled[X] = beta*scaling[X]
        
for X in autoscale.intersection(data.bcc_metals):
    scaling[X] = min(1,round(np.sqrt((c11[X]-c12[X]+c44[X])/(3*mu[X])),2))
    beta_scaled[X] = beta*scaling[X]

### define Burgers (unit-)vectors and slip plane normals for all metals
bfcc = np.array([1,1,0]/np.sqrt(2))
n0fcc = -np.array([1,-1,1]/np.sqrt(3))

bbcc = np.array([1,-1,1]/np.sqrt(3))
n0bcc = np.array([1,1,0]/np.sqrt(2))

b = {}
n0 = {}
b['ISO'] = np.array([1,0,0]) ### take any direction in the isotropic limit
n0['ISO'] = np.array([0,1,0])
for X in data.fcc_metals.intersection(metal):
    b[X] = bfcc
    n0[X] = n0fcc

for X in data.bcc_metals.intersection(metal):
    b[X] = bbcc
    n0[X] = n0bcc

### slip directions for hcp are the [1,1,bar-2,0] directions; since the SOEC are invariant under rotations about the z-axis, we may align e.g. the x-axis with b:
### (comment: TOEC are only invariant under rotations about the z-axis by angles of n*pi/3; measurement is typically done with x-axis aligned with one of the slip directions,
###  so this choise is also consistent with future calculations involving TOEC)
hcpslip = 'basal' ## default
# hcpslip = 'prismatic' ## uncomment to use
# hcpslip = 'pyramidal' ## uncomment to use
for X in data.hcp_metals.intersection(metal):
    b[X] = np.array([-1,0,0]) ## any direction in the x-y plane (=slip plane) is possible, as hexagonal metals are isotropic in the basal plane, see H+L
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
for X in data.tetr_metals.intersection(metal):
    b[X] = np.array([0,0,-1])
    n0[X] = np.array([0,1,0])
####

## compute smallest critical velocity in ratio to the scaling velocity computed from the average shear modulus mu (see above):
## i.e. this corresponds to the smallest velocity leading to a divergence in the dislocation field at some character angle theta
vcrit_smallest = {}
for X in metal:
    ## happens to be the same formula for both fcc and bcc (but corresponding to pure edge for fcc and mixed with theta=arctan(sqrt(2)) for bcc)
    vcrit_smallest[X] = min(np.sqrt(c44[X]/mu[X]),np.sqrt((c11[X]-c12[X])/(2*mu[X])))
    ### also the correct value for some (but not all) hexagonal metals, i.e. depends on values of SOEC and which slip plane is considered
### ... but not for tetragonal, where vcrit needs to be determined numerically:
## numerically determined values:
vcrit_smallest['In'] = 0.549*np.sqrt(c44['In']/mu['In'])
vcrit_smallest['Sn'] = 0.749*np.sqrt(c44['Sn']/mu['Sn'])
vcrit_smallest['Zn'] = 0.998*np.sqrt(c44['Zn']/mu['Zn']) ## for basal slip
if hcpslip=='prismatic':
    vcrit_smallest['Cd'] = 0.932*np.sqrt(c44['Cd']/mu['Cd'])
    vcrit_smallest['Zn'] = 0.766*np.sqrt(c44['Zn']/mu['Zn'])
elif hcpslip=='pyramidal':
    vcrit_smallest['Cd'] = 0.959*np.sqrt(c44['Cd']/mu['Cd'])
    vcrit_smallest['Zn'] = 0.819*np.sqrt(c44['Zn']/mu['Zn'])

### import additional modules
from elasticconstants import elasticC2
import dislocations as dlc

## list of metals symmetric in +/-theta:
metal_symm = sorted(list(data.fcc_metals.union(data.hcp_metals).union(data.tetr_metals).intersection(metal))+['ISO'])

### start the calculations
if __name__ == '__main__':

    if len(sys.argv) > 1:
        ## only compute the metals the user has asked us to
        metal = sys.argv[1].split()

    print("Computing the line tension for: ",metal)

    with open("theta.dat","w") as thetafile:
        thetafile.write('\n'.join(map("{:.6f}".format,theta[1:-1])))        
                           
    C2 = {}
    for X in metal:  
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        C2[X] = elasticC2(c11=c11[X], c12=c12[X], c44=c44[X], c13=c13[X], c33=c33[X], c66=c66[X])/mu[X]
        
    # wrap all main computations into a single function definition to be run in a parallelized loop below
    def maincomputations(i):
        X = metal[i]
        with open("beta_{}.dat".format(X),"w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta_scaled[X])))
    
        geometry = dlc.StrohGeometry(b=b[X], n0=n0[X], theta=theta, phi=phi)
        Cv = geometry['Cv']
        M = geometry['M']
        N = geometry['N']
        
        ### compute dislocation displacement gradient uij and line tension LT
        def compute_lt(j):
            uij = dlc.computeuij(beta=beta_scaled[X][j], C2=C2[X], Cv=Cv, b=b[X], M=M, N=N, phi=phi)
            Etot = dlc.computeEtot(uij=uij, betaj=beta_scaled[X][j], C2=C2[X], Cv=Cv, phi=phi)
            return 4*np.pi*dlc.computeLT(Etot=Etot, dtheta=dtheta)
            
        LT = np.array([compute_lt(j) for j in range(Nbeta+1)])
        
        # write the results to disk (two different formats for convenience):
        with open("LT_{}.dat".format(X),"w") as LTfile:
            for j in range(Nbeta+1):
                LTfile.write('\t'.join(map("{:.6f}".format,LT[j])) + '\n')
        
        with open("LT_betatheta_{}.dat".format(X),"w") as LTbtfile:
            LTbtfile.write('beta\ttheta\tLT\n')
            for j in range(Nbeta+1):
                for th in range(len(theta[1:-1])):
                    LTbtfile.write("{:.4f}".format(beta_scaled[X][j]) +'\t' + "{:.4f}".format(theta[th+1]) + '\t' + "{:.3f}".format(LT[j,th]) + '\n')    
                            
        ### create plots:
        namestring = "{}".format(X)
        beta_trunc = [j for j in beta_scaled[X] if j <=vcrit_smallest[X]]
        if X in metal_symm:
            # plt.figure(figsize=(4.5,3.2))
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.2))
            LT_trunc = LT[:len(beta_trunc),int(Ntheta/2)-1:]
            y_msh , x_msh = np.meshgrid(theta[int(Ntheta/2):-1],beta_trunc)
            plt.yticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        else:
            # plt.figure(figsize=(4.5,4.5))
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,4.5))
            LT_trunc = LT[:len(beta_trunc)]
            y_msh , x_msh = np.meshgrid(theta[1:-1],beta_trunc)
            plt.yticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$-\pi/2$", r"$-3\pi/8$", r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        # plt.xticks(np.array([0, 0.2, 0.4, 0.6, 0.8]),fontsize=fntsize)
        plt.xticks(fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if X=='ISO':
            plt.xlabel(r'$\beta_\mathrm{t}=v/c_\mathrm{t}$',fontsize=fntsize)
            plt.title(r'Isotropic',fontsize=fntsize)
        else:
            plt.xlabel(r'$\beta_{\bar\mu}$',fontsize=fntsize)
            plt.title(namestring,fontsize=fntsize)
        plt.ylabel(r'$\vartheta$',rotation=0,fontsize=fntsize)
        colmsh = plt.pcolormesh(x_msh,y_msh,LT_trunc,vmin=-0.5, vmax=2,cmap = plt.cm.rainbow)
        plt.colorbar()
        plt.contour(x_msh,y_msh,LT_trunc, colors=('black','red','black','black','black','black'), levels=[-0.5,0,0.5,1,1.5,2], linewidths=[0.7,1.0,0.7,0.7,0.7,0.7], linestyles=['solid','solid','dashed','dashdot','dotted','solid'])
        colmsh.set_rasterized(True)
        plt.axhline(0, color='grey', linewidth=0.5, linestyle='dotted')
        plt.savefig("LT_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=450)
        # plt.savefig("LT_{}.png".format(X),format='png',bbox_inches='tight',dpi=450)
        plt.close()

        return 0
        
    # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
    if Ncores == 1:
        [maincomputations(i) for i in range(len(metal))]
    else:
        Parallel(n_jobs=Ncores)(delayed(maincomputations)(i) for i in range(len(metal)))

################################################         
    if Ntheta2==0 or Ntheta2==None:
        sys.exit()
    
    print("Computing critical velocities for: ",metal)
    import sympy as sp
    from scipy.optimize import fmin
    ### define some symbols and functions of symbols
    cc11 = sp.Symbol('cc11', real=True)
    cc12 = sp.Symbol('cc12', real=True)
    cc44 = sp.Symbol('cc44', real=True)
    cc13 = sp.Symbol('cc13', real=True)
    cc33 = sp.Symbol('cc33', real=True)
    cc66 = sp.Symbol('cc66', real=True)
    substitutions = {}
    for X in metal:
        ## substitute SOEC in units of GPa to minimize rounding errors from huge/tiny floats in intermediate steps
        substitutions[X] = {cc11:c11[X]/1e9, cc12:c12[X]/1e9, cc44:c44[X]/1e9}
        if c13[X]!=None:
            substitutions[X].update({cc13:c13[X]/1e9, cc33:c33[X]/1e9})
        if c66[X]!=None:
            substitutions[X].update({cc66:c66[X]/1e9})
        
    bt2 = sp.Symbol('bt2', real=True)
    phi = sp.Symbol('phi', real=True)
    C2cub = elasticC2(c11=cc11,c12=cc12,c44=cc44)
    C2hex = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33)
    C2tetr = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33,c66=cc66)
    
    ### setup slip-geometries for symbolic calculations:
    bfcc = np.array([1,1,0])/sp.sqrt(2)
    n0fcc = -np.array([1,-1,1])/sp.sqrt(3)
    
    bbcc = np.array([1,-1,1])/sp.sqrt(3)
    n0bcc = np.array([1,1,0])/sp.sqrt(2)
    
    #basal:
    bhcp = np.array([-1,0,0])
    n0hcp = np.array([0,0,1])
    #prismatic:
    n0pris = np.array([0,-1,0])
    #pyramidal:
    n0pyr = {}
    for X in data.hcp_metals.intersection(metal):
        n0pyr[X] = np.array([0,-ac[X],cc[X]])/sp.sqrt(ac[X]**2+cc[X]**2)
    
    btetr = np.array([0,0,-1])
    n0tetr = np.array([0,1,0])
        
    ## build symbolic expression of the determinant whose zeros we want to find (as a fct of velocity bt2 and polar angle phi)
    def build_det(N,C2,m0,cc44=cc44,bt2=bt2):
        NC2N = np.dot(N,np.dot(N,C2))
        thedot = np.dot(N,m0)
        for a in sp.preorder_traversal(thedot):
            if isinstance(a, sp.Float):
                thedot = thedot.subs(a, round(a, 12))
        thematrix = NC2N - bt2*cc44*(thedot**2)*np.diag((1,1,1))
        return sp.det(sp.Matrix(thematrix))
       
    def compute_bt2(thi,N,m0,C2,bt2):
        thedet = (build_det(N[thi],C2,m0[thi]))
        return sp.solve(thedet,bt2)
        
    def computevcrit(b,n0,C2,Ntheta,metal_list,symmetric=True,bt2=bt2,Ncores=Ncores):
        ## metal_list must only contain metals of the same symmetry and for the same slip system (although n0 may be a dictionary of slip plane normals)
        vcrit={}
        if symmetric:
            Theta = (sp.pi/2)*np.linspace(0,1,Ntheta)
        else:
            Theta = (sp.pi/2)*np.linspace(-1,1,Ntheta)
        if isinstance(n0,dict): ## check if slip plane normal depends on metal (like for pyramidal slip)
            Xloop=True
            t = {}
            m0 = {}
            N = {}
            bt2_curr = {}
            for X in metal_list:
                t[X] = np.zeros((Ntheta,3))
                m0[X] = np.zeros((Ntheta,3))
                N[X] = np.zeros((Ntheta,3),dtype=object)
                for thi in range(Ntheta):
                    t[X][thi] = sp.cos(Theta[thi])*b + sp.sin(Theta[thi])*np.cross(b,n0[X])
                    m0[X][thi] = np.cross(n0[X],t[X][thi])
                    N[X][thi] = n0[X]*sp.cos(phi) - m0[X][thi]*sp.sin(phi)
                bt2_curr[X] = np.zeros((Ntheta,3),dtype=object)
                if Ncores == 1:
                    for thi in range(Ntheta):
                        bt2_curr[X][thi] = compute_bt2(thi,N[X],m0[X],C2,bt2)
                else:
                    bt2_curr[X] = np.array(Parallel(n_jobs=Ncores)(delayed(compute_bt2)(thi,N[X],m0[X],C2,bt2) for thi in range(Ntheta)))
        else:
            Xloop=False
            t = np.zeros((Ntheta,3))
            m0 = np.zeros((Ntheta,3))
            N = np.zeros((Ntheta,3),dtype=object)
            for thi in range(Ntheta):
                t[thi] = sp.cos(Theta[thi])*b + sp.sin(Theta[thi])*np.cross(b,n0)
                m0[thi] = np.cross(n0,t[thi])
                N[thi] = n0*sp.cos(phi) - m0[thi]*sp.sin(phi)
            bt2_curr = np.zeros((Ntheta,3),dtype=object)
            if Ncores == 1:
                for thi in range(Ntheta):
                    bt2_curr[thi] = compute_bt2(thi,N,m0,C2,bt2)
            else:
                bt2_curr = np.array(Parallel(n_jobs=Ncores)(delayed(compute_bt2)(thi,N,m0,C2,bt2) for thi in range(Ntheta)))
        for X in metal_list:
            bt2_res = np.zeros((3,2),dtype=complex)
            vcrit[X] = np.zeros((Ntheta,3))
            for thi in range(Ntheta):
                if Xloop:
                    bt2_X = np.copy(bt2_curr[X][thi])
                else:
                    bt2_X = np.copy(bt2_curr[thi])
                for i in range(len(bt2_X)):
                    bt2_X[i] = (bt2_X[i].subs(substitutions[X]))
                for i in range(len(bt2_X)):
                    fphi = sp.lambdify((phi),bt2_X[i],modules=["math", "mpmath", "sympy"])
                    def f(x):
                        out = fphi(x)
                        return sp.re(out)
                    bt2_res[i,0] = fmin(f,0.001,disp=False)
                    bt2_res[i,1] = bt2_X[i].subs({phi:bt2_res[i,0]})
                mask = np.round(np.imag(bt2_res[:,1]),12)==0 ## only keep real solutions
                vcrit[X][thi] = np.sqrt(c44[X]/mu[X]*np.real(bt2_res[:,1][mask]))
        return vcrit
        
    vcrit = {}
    vcrit.update(computevcrit(bfcc,n0fcc,C2cub,Ntheta2,sorted(list(data.fcc_metals.intersection(metal))),symmetric=True))
    vcrit.update(computevcrit(bbcc,n0bcc,C2cub,2*Ntheta2-1,sorted(list(data.bcc_metals.intersection(metal))),symmetric=False))
    if hcpslip=='basal':
        vcrit.update(computevcrit(bhcp,n0hcp,C2hex,Ntheta2,sorted(list(data.hcp_metals.intersection(metal))),symmetric=True))
    elif hcpslip=='prismatic':
        vcrit.update(computevcrit(bhcp,n0pris,C2hex,Ntheta2,sorted(list(data.hcp_metals.intersection(metal))),symmetric=True))
    elif hcpslip=='pyramidal':
        vcrit.update(computevcrit(bhcp,n0pyr,C2hex,Ntheta2,sorted(list(data.hcp_metals.intersection(metal))),symmetric=True))
    vcrit.update(computevcrit(btetr,n0tetr,C2tetr,Ntheta2,sorted(list(data.tetr_metals.intersection(metal))),symmetric=True))

    ## write vcrit results to disk, then plot
    rho = data.CRC_rho ## need densities to convert vcrit to m/s
    with open("vcrit.dat","w") as vcritfile:
        vcritfile.write("theta/pi\t" + '\t'.join(map("{:.6f}".format,np.linspace(1/2,-1/2,2*Ntheta2-1))) + '\n')
        vcritfile.write("metal / vcrit[m/s] (3 solutions per angle)\n")
        for X in sorted(list(set(metal).intersection(vcrit.keys()))):
            for i in range(3):
                vcritfile.write("{}\t".format(X) + '\t'.join(map("{:.0f}".format,np.flipud(np.sqrt(mu[X]/rho[X])*vcrit[X][:,i]))) + '\n')
                
    def mkvcritplot(X,vcrit,Ntheta):
        fig, ax = plt.subplots(1, 1, figsize=(5.5,4.5))
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        if len(vcrit)==Ntheta:
            plt.xticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(0,np.pi/2,Ntheta)
        else:
            plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
        ax.axis((min(thetapoints),max(thetapoints),np.min(vcrit)*0.97,np.max(vcrit)*1.02)) ## define plot range
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(r'$\vartheta$',fontsize=fntsize)
        ax.set_ylabel(r'$v_c$[m/s]',fontsize=fntsize)
        ax.set_title("3 vcrit solutions for {}".format(X),fontsize=fntsize)
        if np.all(np.round(np.max(vcrit,axis=1),6)==round(np.max(vcrit,axis=1)[0],6)) and np.all(np.round(np.min(vcrit,axis=1),6)==round(np.min(vcrit,axis=1)[0],6)):
            vcrit = np.sort(vcrit)
        for i in range(3):
            ax.plot(thetapoints,vcrit[:,i])
        plt.savefig("vcrit_{}.pdf".format(X),format='pdf',bbox_inches='tight')
        plt.close()
    
    for X in sorted(list(set(metal).intersection(vcrit.keys()))):
        mkvcritplot(X,np.sqrt(mu[X]/rho[X])*vcrit[X],Ntheta2)
        