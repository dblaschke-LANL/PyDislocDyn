# -*- coding: utf-8 -*-
# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Los Alamos National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Sept. 25, 2018
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
fntsize=12
from matplotlib.ticker import AutoMinorLocator
##################
import metal_data as data
from joblib import Parallel, delayed
## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
Ncores = -2

### choose resolution of discretized parameters: theta is the angle between disloc. line and Burgers vector, beta is the dislocation velocity,
### and phi is an integration angle used in the integral method for computing dislocations
Ntheta = 600
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
# mu['Be'] = 148.6e9 ## use Hill average for polycrystalline Be, In (see results of 'polycrystal_averaging.py')
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
bhcp = np.array([-1,0,0]) ## any direction in the x-y plane (=slip plane) is possible, as hexagonal metals are isotropic in the basal plane, see H+L
n0hcp = np.array([0,0,1]) ## slip plane normal = normal to basal plane
for X in data.hcp_metals.intersection(metal):
    b[X] = bhcp
    n0[X] = n0hcp
    
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
vcrit_smallest['In'] = 0.54911*np.sqrt(c44['In']/mu['In'])
vcrit_smallest['Sn'] = 0.74938*np.sqrt(c44['Sn']/mu['Sn'])
vcrit_smallest['Zn'] = 0.99759*np.sqrt(c44['Zn']/mu['Zn']) ## for basal slip

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
        
    # wrap all main computations into a single function definition to be run in a parallelized loop below
    def maincomputations(i):
        X = metal[i]
        with open("beta_{}.dat".format(X),"w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta_scaled[X])))
                   
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        C2 = elasticC2(c11=c11[X], c12=c12[X], c44=c44[X], c13=c13[X], c33=c33[X], c66=c66[X])/mu[X]
    
        geometry = dlc.StrohGeometry(b=b[X], n0=n0[X], theta=theta, phi=phi)
        Cv = geometry['Cv']
        M = geometry['M']
        N = geometry['N']
        
        ### compute dislocation displacement gradient uij and line tension LT
        def compute_lt(j):
            uij = dlc.computeuij(beta=beta_scaled[X][j], C2=C2, Cv=Cv, b=b[X], M=M, N=N, phi=phi)
            Etot = dlc.computeEtot(uij=uij, betaj=beta_scaled[X][j], C2=C2, Cv=Cv, phi=phi)
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
