# Compute the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Los Alamos National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - May 22, 2018
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
plt.rc('font',**{'family':'Liberation Serif','size':'11'})
fntsize=11
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
##################
import metal_data as data
# import polycrystal_averaging as pca
from elasticconstants import elasticC2, elasticC3
from dislocations import fourieruij_iso
from phononwind import elasticA3, dragcoeff_iso
from joblib import Parallel, delayed
## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
Ncores = -2

### choose various resolutions and other parameters:
Ntheta = 3 # number of angles between burgers vector and dislocation line (minimum 2, i.e. pure edge and pure screw)
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
Nphi = 50
Nphi1 = 50
Nq1 = 400
Nt = 250
### and range & step sizes
theta = np.linspace(0,np.pi/2,Ntheta)
beta = np.linspace(minb,maxb,Nbeta)
highT = np.linspace(roomT,maxT,NT)
phi = np.linspace(0,2*np.pi,Nphi)

#### load input data:
ac = data.CRC_a
cc = data.CRC_c
rho = data.CRC_rho
# rho = data.CRC_rho_sc
#### EITHER LOAD ISOTROPIC DATA FROM EXTERNAL SOURCE
c12 = data.ISO_c12
c44 = data.ISO_c44
cMl = data.ISO_l
cMm = data.ISO_m
cMn = data.ISO_n
### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. at least TOEC for polycrystals)
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).intersection(cMl.keys())))

#### OR GENERATE IT BY AVERAGING OVER SINGLE CRYSTAL CONSTANTS:
# c12 = {}
# c44 = {}
# cMl = {}
# cMm = {}
# cMn = {}
# ### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. TOEC)
# metal = sorted(list(data.c111.keys()))
# metal_cubic = data.fcc_metals.union(data.bcc_metals).intersection(metal)
# ### compute average elastic constants for these metals:
# VoigtAverage = {}
# ReussAverage = {}
# HillAverage = {}
# ImprovedAv = {}
# print("computing averaged elastic constants ...")
# for X in metal:
#     pca.C2[X] = elasticC2(c11=pca.c11[X], c12=pca.c12[X], c44=pca.c44[X], c13=pca.c13[X], c33=pca.c33[X], c66=pca.c66[X])/1e9    
#     pca.S2[X] = pca.ec.elasticS2(pca.C2[X])
#     pca.C3[X] = elasticC3(c111=pca.c111[X], c112=pca.c112[X], c113=pca.c113[X], c123=pca.c123[X], c133=pca.c133[X], c144=pca.c144[X], c155=pca.c155[X], c166=pca.c166[X], c222=pca.c222[X], c333=pca.c333[X], c344=pca.c344[X], c366=pca.c366[X], c456=pca.c456[X])/1e9
#     pca.S3[X] = pca.ec.elasticS3(pca.S2[X],pca.C3[X])
#     VoigtAverage[X] = pca.voigt_average(pca.C2[X],pca.C3[X])    
#     ReussAverage[X] = pca.reuss_average(pca.S2[X],pca.S3[X]) 
#     HillAverage[X] = pca.hill_average(VoigtAverage[X], ReussAverage[X])
#     ### use Hill average for Lame constants for non-cubic metals, as we do not have a better scheme at the moment
#     c12[X] = float(HillAverage[X][pca.lam])*1e9
#     c44[X] = float(HillAverage[X][pca.mu])*1e9
#     ### use Hill average for Murnaghan constants, as we do not have a better scheme at the moment
#     cMl[X] = float(HillAverage[X][pca.Murl])*1e9
#     cMm[X] = float(HillAverage[X][pca.Murm])*1e9
#     cMn[X] = float(HillAverage[X][pca.Murn])*1e9
#     
# # replace Hill with improved averages for effective Lame constants of cubic metals:
# for X in metal_cubic:
#     ### don't waste time computing the "improved average" for the Murnaghan constants when we are going to use the Hill average
#     ImprovedAv[X] = pca.improved_average(X,pca.C2[X],None)
#     c12[X] = float(ImprovedAv[X][pca.lam])*1e9
#     c44[X] = float(ImprovedAv[X][pca.mu])*1e9
##################################################
ct_over_cl = {}
qBZ = {}
ct = {}
cl = {}
burgers = {}
bulk = {}

### compute various numbers for these metals
for X in metal:
    ct_over_cl[X] = np.sqrt(c44[X]/(c12[X]+2*c44[X]))
    qBZ[X] = ((6*np.pi**2)**(1/3))/ac[X]
    ct[X] = np.sqrt(c44[X]/rho[X])
    cl[X] =  ct[X]/ct_over_cl[X]
    bulk[X] = c12[X] + 2*c44[X]/3

for X in data.fcc_metals.intersection(metal):
    burgers[X] = ac[X]/np.sqrt(2)

for X in data.bcc_metals.intersection(metal):
    burgers[X] = ac[X]*np.sqrt(3)/2
    
for X in data.hcp_metals.intersection(metal):
    burgers[X] = ac[X]
    qBZ[X] = ((4*np.pi**2/(ac[X]*ac[X]*cc[X]*np.sqrt(3)))**(1/3))
    
for X in data.tetr_metals.intersection(metal):
    burgers[X] = cc[X]  # for one possible slip system
    qBZ[X] = ((6*np.pi**2/(ac[X]*ac[X]*cc[X]))**(1/3))
    
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
    
    for X in metal:
        
        A3 = elasticA3(elasticC2(c12=c12[X], c44=c44[X]), elasticC3(l=cMl[X], m=cMm[X], n=cMn[X]))/c44[X]  
        
        # wrap all main computations into a single function definition to be run in a parallelized loop below
        def maincomputations(bt,modes=modes):
            Bmix = np.zeros((len(theta),len(highT)))
                                    
            dij = fourieruij_iso(bt, ct_over_cl[X], theta, phi)
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3, qBZ=qBZ[X], ct=ct[X], cl=cl[X], beta=bt, burgers=burgers[X], T=roomT, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
            
            for Ti in range(len(highT)-1):
                T = highT[Ti+1]
                qBZT = qBZ[X]/(1 + alpha_a[X]*(T - roomT))
                c44T = c44[X] ## TODO: need to implement T dependence of shear modulus!
                c12T = bulk[X] - 2*c44T/3 ## TODO: need to implement T dependence of bulk modulus!
                rhoT = rho[X]/(1 + alpha_a[X]*(T - roomT))**3
                burgersT = burgers[X]*(1 + alpha_a[X]*(T - roomT))
                ctT = np.sqrt(c44T/rhoT)
                ct_over_cl_T = np.sqrt(c44T/(c12T+2*c44T))
                clT = ctT/ct_over_cl_T
                ## beta, as it appears in the equations, is v/ctT, therefore:
                betaT = bt*ct[X]/ctT
                
                dij = fourieruij_iso(betaT, ct_over_cl_T, theta, phi)
                ### TODO: need models for T dependence of Murnaghan constants!
                cMlT = cMl[X]
                cMmT = cMm[X]
                cMnT = cMn[X]
                ##
                A3T = elasticA3(elasticC2(c12=c12T, c44=c44T), elasticC3(l=cMlT, m=cMmT, n=cMnT))/c44T
                Bmix[:,Ti+1] = dragcoeff_iso(dij=dij, A3=A3T, qBZ=qBZT, ct=ctT, cl=clT, beta=betaT, burgers=burgersT, T=T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
                
            return Bmix

        # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
        if Ncores == 1:
            Bmix = np.array([maincomputations(bt,modes) for bt in beta])
        else:
            Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt,modes) for bt in beta))


        # and write the results to disk (in various formats)
        with open("drag_{}.dat".format(X),"w") as Bfile:
            # Bfile.write('beta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,theta[1:-1])) + '\tBedge[mPas]' + '\n')
            Bfile.write("### B(beta,theta) for {} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
            Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,theta/np.pi)) + '\n')
            for bi in range(len(beta)):
                Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')

        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT)>1:
            with open("drag_T_{}.dat".format(X),"w") as Bfile:
                Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,theta[1:-1])) + '\tBedge[mPas]' + '\n') 
                for bi in range(len(beta)):
                    for Ti in range(len(highT)):               
                        Bfile.write("{:.1f}".format(highT[Ti]) +'\t' + "{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')
                        
            with open("drag_T_screw_{}.dat".format(X),"w") as Bscrewfile:
                for bi in range(len(beta)):
                    Bscrewfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,0])) + '\n')
            
            with open("drag_T_edge_{}.dat".format(X),"w") as Bedgefile:
                for bi in range(len(beta)):
                    Bedgefile.write('\t'.join(map("{:.6f}".format,Bmix[bi,-1])) + '\n')
    
            for th in range(len(theta[1:-1])):
                with open("drag_T_mix{0:.6f}_{1}.dat".format(theta[th+1],X),"w") as Bmixfile:
                    for bi in range(len(beta)):
                        Bmixfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,th+1])) + '\n')
        
    ###### plot room temperature results:
    ## load data from isotropic calculation
    Broom = {}
    for X in metal:
        ## for every X, Broom has shape (len(theta+1),Nbeta), first column is beta all others all B for various dislocation types theta in the range 0 to  pi/2
        Broom[X] = np.zeros((Nbeta,Ntheta+1))
        with open("drag_{}.dat".format(X),"r") as Bfile:
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
            
    ## define line styles for every metal in the same plot
    lnstyles = {'Al':'-', 'Cu':'--', 'Fe':':', 'Nb':'-.'}
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe':'green', 'Nb':'red'}
    ## plot only pure screw and pure edge by default
    theta_indices = [0,Ntheta-1]
    for th in theta_indices:
        fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.2))
        titlestring = "$\\vartheta=$"+"{0:.4f}".format(theta[th])
        ax.axis((0,beta[-1],0,0.06))
        ax.set_xticks(np.arange(10)/10)
        ax.set_yticks(np.arange(7)/100)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax.set_xlabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        ax.set_ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        ax.set_title(titlestring,fontsize=fntsize)
        for X in metal:
            ax.plot(Broom[X][:,0],Broom[X][:,th+1],lnstyles[X], color=metalcolors[X], label=X)
        legend = ax.legend(loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.2, shadow=True)
        plt.savefig("B_iso_{:.4f}.pdf".format(theta[th]),format='pdf',bbox_inches='tight')
        plt.close()
        
