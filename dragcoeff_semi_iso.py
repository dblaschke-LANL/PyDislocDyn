# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Mar. 31, 2020
#################################
from __future__ import division
from __future__ import print_function

import sys
import os
### make sure we are running a recent version of python
# assert sys.version_info >= (3,6)
import numpy as np
from scipy.optimize import curve_fit, fmin
##################
import matplotlib as mpl
mpl.use('Agg', warn=False) # don't need X-window, allow running in a remote terminal session
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
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import metal_data as data
from elasticconstants import elasticC2, elasticC3, Voigt, UnVoigt
import polycrystal_averaging as pca
import dislocations as dlc
from phononwind import elasticA3, dragcoeff_iso
## work around for python 2:
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    ## Ncores=0 bypasses phononwind calculations entirely and only generates plots using data from a previous run
    Ncores = max(1,int(Ncpus/2)) ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
    # Ncores = -2
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = 1 ## must be 1 (or 0) without joblib

### choose various resolutions and other parameters:
Ntheta = 21 # number of angles between burgers vector and dislocation line (minimum 2, i.e. pure edge and pure screw)
Nbeta = 99 # number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)
minb = 0.01
maxb = 0.99
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
# modes = 'TT'
skip_plots=False ## set to True to skip generating plots from the results
use_exp_Lame=True ## if set to True, experimental values (where available) are taken for the Lame constants, isotropic phonon spectrum, and sound speeds
## missing values (such as Mo, Zr, or all if use_exp_Lame=False) are supplemented by Hill averages, or for cubic crystals the 'improved average' (see 'polycrystal_averaging.py')
use_iso=False ## set to True to calculate using isotropic elastic constants from metal_data (no effect if input files are used)
## choose among predefined slip systems when using metal_data.py (see that file for details)
bccslip = '110' ## allowed values: '110' (default), '112', '123', 'all' (for all three)
hcpslip = 'basal' ## allowed values: 'basal', 'prismatic', 'pyramidal', 'all' (for all three)
#####
NT = 1 # number of temperatures between baseT and maxT (WARNING: implementation of temperature dependence is incomplete!)
constantrho = False ## set to True to override thermal expansion coefficient and use alpha_a = 0 for T > baseT
increaseTby = 300 # so that maxT=baseT+increaseTby (default baseT=300 Kelvin, but may be overwritten by an input file below)
beta_reference = 'base'  ## define beta=v/ct, choosing ct at baseT ('base') or current T ('current') as we increase temperature
#####
# in Fourier space:
Nphi = 50
Nphi1 = 50
Nq1 = 400
Nt = 321 # base value, grid is adaptive in Nt
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
### and range & step sizes
theta = np.linspace(0,np.pi/2,Ntheta)  ## note: some slip systems (such as bcc defined below) are asymmetric wrt theta->-theta, in that case uncomment line below:
# theta = np.linspace(-np.pi/2,np.pi/2,Ntheta)
beta = np.linspace(minb,maxb,Nbeta)
phi = np.linspace(0,2*np.pi,Nphi)
phiX = np.linspace(0,2*np.pi,NphiX)

### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. at least TOEC)
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(data.c123.keys())))

isokeywd='omit' ## writeinputfile(..., iso='omit') will bypass writing ISO_c44 values to the input files and missing Lame constants will always be auto-generated by averaging
if use_exp_Lame:
    isokeywd=False
if use_iso:
    metal = sorted(list(set(metal).intersection(data.ISO_c44.keys()).intersection(data.ISO_l.keys())))
    isokeywd=True

#########
if __name__ == '__main__':
    Y={}
    inputdata = {}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        try:
            for i in range(len(args)):
                inputdata[i]=pca.readinputfile(args[i])
                X = inputdata[i].name
                metal_list.append(X)
                Y[X] = inputdata[i]
            use_metaldata=False
            metal = metal_list
            print("success reading input files ",args)
        except FileNotFoundError:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = sys.argv[1].split()
        
    bcc_metals = data.bcc_metals.copy()
    hcp_metals = data.hcp_metals.copy()
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        for X in metal:
            if X in bcc_metals:
                if bccslip == 'all':
                    slipkw = ['110', '112', '123']
                else:
                    slipkw=[bccslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokeywd,bccslip=kw)
                    metal_list.append(X+kw)
                    bcc_metals.add(X+kw)
            elif X in hcp_metals:
                if hcpslip == 'all':
                    slipkw = ['basal','prismatic','pyramidal']
                else:
                    slipkw=[hcpslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokeywd,hcpslip=kw)
                    metal_list.append(X+kw)
                    hcp_metals.add(X+kw)
            else:
                data.writeinputfile(X,X,iso=isokeywd) # write temporary input files for requested X of metal_data
                metal_list.append(X)
        for X in metal_list:
            Y[X] = pca.readinputfile(X)
        os.chdir("..")
        metal = metal_list
    
    if Ncores == 0:
        print("skipping phonon wind calculations as requested")
    else:
        with open("beta.dat","w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta)))
    
        with open("theta.dat","w") as thetafile:
            thetafile.write('\n'.join(map("{:.6f}".format,theta)))
        
        print("Computing the drag coefficient from phonon wind ({} modes) for: ".format(modes),metal)
    
    ###
    r = np.array([rmin*np.pi,rmax*np.pi]) ## qBZ drops out of product q*r, so can rescale both vectors making them dimensionless and independent of the metal
    q = np.linspace(0,1,Nq)
    ## needed for the Fourier transform of uij (but does not depend on beta or T, so we compute it only once here)
    # sincos = dlc.fourieruij_sincos(r,phiX,q,phi)
    ## for use with fourieruij_nocut(), which is faster than fourieruij() if cutoffs are chosen such that they are neglegible in the result:
    sincos_noq = np.average(dlc.fourieruij_sincos(r,phiX,q,phi)[3:-4],axis=0)
        
    A3rotated = {}
    C2 = {}
    highT = {}
    dislocation = {}
    rotmat = {}
    linet = {}
    velm0 = {}
    for X in metal:
        highT[X] = np.linspace(Y[X].T,Y[X].T+increaseTby,NT)
        if constantrho==True:
            Y[X].alpha_a = 0
        ## only write temperature to files if we're computing temperatures other than baseT=Y[X].T
        if len(highT[X])>1 and Ncores !=0:
            with open("temperatures_{}.dat".format(X),"w") as Tfile:
                Tfile.write('\n'.join(map("{:.2f}".format,highT[X])))
        
        dislocation[X] = dlc.StrohGeometry(b=Y[X].b, n0=Y[X].n0, theta=theta, Nphi=NphiX)
        linet[X] = np.round(dislocation[X].t,15)
        velm0[X] = np.round(dislocation[X].m0,15)
        dislocation[X].computerot()
        rotmat[X] = np.round(dislocation[X].rot,15)
               
        C2[X] = UnVoigt(Y[X].C2/Y[X].mu)  ## this must be the same mu that was used to define the dimensionless velocity beta, as both enter dlc.computeuij() on equal footing below!
        C3 = UnVoigt(Y[X].C3/Y[X].mu)
        A3 = elasticA3(C2[X],C3)
        A3rotated[X] = np.zeros((len(theta),3,3,3,3,3,3))
        for th in range(len(theta)):
            rotm = rotmat[X][th]
            A3rotated[X][th] = np.round(np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(A3,rotm.T)))))),12)
        
    # wrap all main computations into a single function definition to be run in a parallelized loop below
    def maincomputations(bt,X,modes=modes):
        Bmix = np.zeros((len(theta),len(highT[X])))
        ### compute dislocation displacement gradient uij, then its Fourier transform dij:
        dislocation[X].computeuij(beta=bt, C2=C2[X])
        dislocation[X].alignuij()
        # uij_iso = dlc.computeuij_iso(bt,Y[X].ct_over_cl, theta, phiX)
        # r = np.exp(np.linspace(np.log(Y[X].burgers/5),np.log(100*Y[X].burgers),125))
        ## perhaps better: relate directly to qBZ which works for all crystal structures (rmin/rmax defined at the top of this file)
        # r = np.exp(np.linspace(np.log(rmin*np.pi/Y[X].qBZ),np.log(rmax*np.pi/Y[X].qBZ),Nr))
        # q = Y[X].qBZ*np.linspace(0,1,Nq)
        # dij = np.average(dlc.fourieruij(dislocation[X].uij_aligned,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
        dij = dlc.fourieruij_nocut(dislocation[X].uij_aligned,phiX,phi,sincos=sincos_noq)
        Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3rotated[X], qBZ=Y[X].qBZ, ct=Y[X].ct, cl=Y[X].cl, beta=bt, burgers=Y[X].burgers, T=Y[X].T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
        
        for Ti in range(len(highT[X])-1):
            T = highT[X][Ti+1]
            expansionratio = (1 + Y[X].alpha_a*(T - Y[X].T)) ## TODO: replace with values from eos!
            qBZT = Y[X].qBZ/expansionratio
            burgersT = Y[X].burgers*expansionratio
            rhoT = Y[X].rho/expansionratio**3
            muT = Y[X].mu ## TODO: need to implement T dependence of shear modulus!
            lamT = Y[X].bulk - 2*muT/3 ## TODO: need to implement T dependence of bulk modulus!
            ctT = np.sqrt(muT/rhoT)
            ct_over_cl_T = np.sqrt(muT/(lamT+2*muT))
            clT = ctT/ct_over_cl_T
            ## beta, as it appears in the equations, is v/ctT, therefore:
            if beta_reference == 'current':
                betaT = bt
            else:
                betaT = bt*Y[X].ct/ctT
            ###### T dependence of elastic constants (TODO)
            c11T = Y[X].c11
            c12T = Y[X].c12
            c44T = Y[X].c44
            c13T = Y[X].c13
            c33T = Y[X].c33
            c66T = Y[X].c66
            c111T = Y[X].c111
            c112T = Y[X].c112
            c113T = Y[X].c113
            c123T = Y[X].c123
            c133T = Y[X].c133
            c144T = Y[X].c144
            c155T = Y[X].c155
            c166T = Y[X].c166
            c222T = Y[X].c222
            c333T = Y[X].c333
            c344T = Y[X].c344
            c366T = Y[X].c366
            c456T = Y[X].c456
            ###
            C2T = elasticC2(c11=c11T, c12=c12T, c44=c44T, c13=c13T, c33=c33T, c66=c66T)/muT
            C3T = elasticC3(c111=c111T, c112=c112T, c113=c113T, c123=c123T, c133=c133T, c144=c144T, c155=c155T, c166=c166T, c222=c222T, c333=c333T, c344=c344T, c366=c366T, c456=c456T)/muT
            A3T = elasticA3(C2T,C3T)
            A3Trotated = np.zeros((len(theta),3,3,3,3,3,3))
            for th in range(len(theta)):
                rotm = rotmat[X][th]
                A3Trotated[th] = np.round(np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(A3T,rotm.T)))))),12)
            ##########################
            dislocation[X].computeuij(beta=betaT, C2=C2T)
            dislocation[X].alignuij()
            ## rT*qT = r*q, so does not change anything
            # dij = np.average(dlc.fourieruij(dislocation[X].uij_aligned,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
            dij = dlc.fourieruij_nocut(dislocation[X].uij_aligned,phiX,phi,sincos=sincos_noq)
            Bmix[:,Ti+1] = dragcoeff_iso(dij=dij, A3=A3Trotated, qBZ=qBZT, ct=ctT, cl=clT, beta=betaT, burgers=burgersT, T=T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
        
        return Bmix

    for X in metal:
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
        if len(highT[X])>1 and Ncores !=0:
            with open("drag_anis_T_{}.dat".format(X),"w") as Bfile:
                Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,theta[1:-1])) + '\tBedge[mPas]' + '\n')
                for bi in range(len(beta)):
                    for Ti in range(len(highT[X])):
                        Bfile.write("{:.1f}".format(highT[X][Ti]) +'\t' + "{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')
                
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
    ## compute smallest critical velocity in ratio (for those data provided in metal_data) to the scaling velocity and plot only up to this velocity
    vcrit_screw = {}
    vcrit_edge = {}
    vcrit_smallest = {}
    for X in metal:
        if Y[X].sym=='iso':
            vcrit_smallest[X] = 1
        else:
            vcrit_smallest[X] = min(np.sqrt(Y[X].c44/Y[X].mu),np.sqrt((Y[X].c11-Y[X].c12)/(2*Y[X].mu)))
    ## above is correct for fcc, bcc (except 123 planes) and some (but not all) hcp, i.e. depends on values of SOEC and which slip plane is considered;
    ## numerically determined values at T=300K for metals in metal_data.py (rounded):
    if use_metaldata:
        vcrit_smallest['Sn'] = 0.818
        vcrit_smallest['Znbasal'] = 0.943 ## for basal slip
        # vcrit for pure screw/edge for default slip systems (incl. basal for hcp), numerically determined values (rounded):
        vcrit_screw = {'Ag': 0.973, 'Al': 1.005, 'Au': 0.996, 'Cdbasal': 1.398, 'Cu': 0.976, 'Fe110': 0.803, 'Mgbasal': 0.982, 'Mo110': 0.987, 'Nb110': 0.955, 'Ni': 1.036, 'Sn': 1.092, 'Tiprismatic': 1.033, 'Znbasal': 1.211, 'Zrbasal': 0.990}
        vcrit_edge = {'Cdprismatic': 1.398, 'Fe110': 0.852, 'Mgprismatic': 0.982, 'Mo110': 1.033, 'Nb110': 1.026, 'Sn': 1.092, 'Tibasal': 1.033, 'Znprismatic': 1.211, 'Zrprismatic': 0.990}
        for X in data.fcc_metals.union(hcp_metals).intersection(metal):
            if X in ['Tibasal']:
                vcrit_screw[X] = vcrit_smallest[X]
            else:
                vcrit_edge[X] = vcrit_smallest[X] ## coincide for the fcc slip system considered above, and for most hcp-basal slip systems
    
    for X in metal:
        if X not in vcrit_screw.keys(): ## fall back to this:
            vcrit_screw[X] = vcrit_smallest[X]
        if X not in vcrit_edge.keys():
            vcrit_edge[X] = vcrit_smallest[X]

    if use_metaldata:
        if hcpslip=='prismatic' or hcpslip=='all':
            for X in hcp_metals.intersection(metal):
                if X in ['Tiprismatic']:
                    vcrit_edge[X] = vcrit_smallest[X]
                elif "prismatic" in X:
                    vcrit_screw[X] = vcrit_smallest[X]
            vcrit_screw['Znprismatic'] = 0.945
            vcrit_smallest['Cdprismatic'] = 0.948
            vcrit_smallest['Znprismatic'] = 0.724
        if hcpslip=='pyramidal' or hcpslip=='all':
            vcrit_screw['Cdpyramidal'] = 1.278
            vcrit_screw['Mgpyramidal'] = 0.979
            vcrit_screw['Tipyramidal'] = 0.930
            vcrit_screw['Znpyramidal'] = 1.132
            vcrit_screw['Zrpyramidal'] =0.976
            vcrit_edge['Tipyramidal'] = vcrit_smallest['Tipyramidal']
            vcrit_edge['Znpyramidal'] = 0.945
            vcrit_smallest['Cdpyramidal'] = 0.975
            vcrit_smallest['Znpyramidal'] = 0.775
            
        if bccslip=='112' or bccslip=='all':
            for X in bcc_metals.intersection(metal):
                if '112' in X:
                    vcrit_screw[X] = vcrit_smallest[X] ## coincide for the bcc slip system with 112 planes
            vcrit_edge['Fe112'] = 0.817
        elif bccslip=='123' or bccslip=='all':
            vcrit_smallest['Fe123'] = 0.735
            for X in bcc_metals.intersection(metal):
                if '123' in X:
                    vcrit_screw[X] = vcrit_smallest[X] ## close enough
            vcrit_edge['Fe123'] = 0.825
    
    ## overwrite any of these values with data from input file, if available:
    for X in metal:
        if Y[X].vcrit_smallest != None:
            vcrit_smallest[X] = Y[X].vcrit_smallest/Y[X].ct
        if Y[X].vcrit_screw != None:
            vcrit_screw[X] = Y[X].vcrit_screw/Y[X].ct
        if Y[X].vcrit_edge != None:
            vcrit_edge[X] = Y[X].vcrit_edge/Y[X].ct
        
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
        cax.set_facecolor('none')
        for axis in ['top','bottom','left','right']:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.tight_layout(pad=0.3)
        plt.savefig("B_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=300)
        plt.close()
    
    ## define line colors for every metal in the same plot
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
    scale_plot = 1 ## need to increase plot and fitting range for higher temperatures
    for X in metal:
        scale_plot = max(scale_plot,int(scale_plot*Y[X].T/30)/10)
        Bmax_fit = int(20*Y[X].T/300)/100 ## only fit up to Bmax_fit [mPas]
        if X in bcc_metals and (np.all(theta[X]>=0) or np.all(theta[X]<=0)):
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
            fitfile.write(" & "+"{:.0f}".format(Y[X].ct))
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
        ax.set_yticks(np.arange(12)*scale_plot/100)
        ax.axis((0,maxb,0,0.11*scale_plot)) ## define plot range
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
            if X in metalcolors.keys():
                ax.plot(beta[beta<cutat],B,color=metalcolors[X],label=X)
            else:
                ax.plot(beta[beta<cutat],B,label=X) ## fall back to automatic colors
            beta_highres = np.linspace(0,vcrit,1000)
            ax.plot(beta_highres,fit_mix(beta_highres/vcrit,*popt),':',color='gray')
        ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig("B_{0:.0f}K_{1}+fits.pdf".format(Y[X].T,filename),format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge","pure edge")
    mkfitplot(metal,"screw","pure screw")
    mkfitplot(metal,"aver","averaged over $\\vartheta$")
    
    
    ### finally, also plot Baver as a function of stress using the fits computed above
    B_of_sig = {}
    sigma = {}
    B0 = {}
    Boffset = {}
    for X in metal:
        vcrit = Y[X].ct*vcrit_smallest[X]
        popt = popt_aver[X]
        burg = Y[X].burgers
        
        @np.vectorize
        def B(v):
            bt = abs(v/vcrit)
            if bt<1:
                out = 1e-3*fit_mix(bt, *popt)
            else:
                out = np.inf
            return out
            
        @np.vectorize
        def vr(stress):
            '''returns the velocity of a dislocation in the drag dominated regime as a function of stress.'''
            bsig = abs(burg*stress)
            def nonlinear_equation(v):
                return abs(bsig-abs(v)*B(v)) ## need abs() if we are to find v that minimizes this expression (and we know that minimum is 0)
            out = float(fmin(nonlinear_equation,0.01*vcrit,disp=False))
            zero = abs(nonlinear_equation(out))
            if zero>1e-5 and zero/bsig>1e-2:
                print("Warning: bad convergence for vr(stress={}): eq={:.6f}, eq/(burg*sig)={:.6f}".format(stress,zero,zero/bsig))
            return out
            
        ### compute what stress is needed to move dislocations at velocity v:
        @np.vectorize
        def sigma_eff(v):
            return v*B(v)/burg
            
        ## slope of B in the asymptotic regime:
        @np.vectorize
        def Bstraight(sigma,Boffset=0):
            return Boffset+sigma*burg/vcrit
            
        ## simple functional approximation to B(sigma), follows from B(v)=B0/sqrt(1-(v/vcrit)**2):
        @np.vectorize
        def Bsimple(sigma,B0):
            return B0*np.sqrt(1+(sigma*burg/(vcrit*B0))**2)
            
        ## determine stress that will lead to velocity of 99% critical speed and stop plotting there, or at 1GPa (whichever is smaller)
        sigma_max = min(1e9,sigma_eff(0.99*vcrit))
        # print("{}: sigma(99%vcrit) = {:.1f} MPa".format(X,sigma_max/1e6))
        Boffset[X] = float(B(vr(sigma_max))-Bstraight(sigma_max,0))
        ## find min(B(v)) to use for B0 in Bsimple():
        B0[X] = round(np.min(B(np.linspace(0,0.8*vcrit,1000))),7)
        B0[X] = (B(0)+3*B0[X])/4 ## or use some weighted average between Bmin and B(0)
        # print("{}: Boffset={:.4f}mPas, B0={:.4f}mPas".format(X,1e3*Boffset[X],1e3*B0[X]))
        
        fig, ax = plt.subplots(1, 1, sharey=False, figsize=(3.,2.5))
        ax.set_xlabel(r'$\sigma$[MPa]',fontsize=fntsize)
        ax.set_ylabel(r'$B$[mPas]',fontsize=fntsize)
        ax.set_title("{}, ".format(X) + "averaged over $\\vartheta$",fontsize=fntsize)
        sigma[X] = np.linspace(0,sigma_max,500)
        B_of_sig[X] = B(vr(sigma[X]))
        ax.axis((0,sigma[X][-1]/1e6,0,B_of_sig[X][-1]*1e3))
        ax.plot(sigma[X]/1e6,Bsimple(sigma[X],B0[X])*1e3,':',color='gray',label="$\sqrt{B_0^2\!+\!\\left(\\frac{\sigma b}{v_\mathrm{c}}\\right)^2}$, $B_0=$"+"{:.1f}".format(1e6*B0[X])+"$\mu$Pas")
        ax.plot(sigma[X]/1e6,Bstraight(sigma[X],Boffset[X])*1e3,':',color='green',label="$B_0+\\frac{\sigma b}{v_\mathrm{c}}$, $B_0=$"+"{:.1f}".format(1e6*Boffset[X])+"$\mu$Pas")
        ax.plot(sigma[X]/1e6,B_of_sig[X]*1e3,label="$B_\mathrm{fit}(v(\sigma))$")
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc='upper left',handlelength=1.1, frameon=False, shadow=False,fontsize=8)
        plt.savefig("B_of_sigma_{}.pdf".format(X),format='pdf',bbox_inches='tight')
        plt.close()
        
        
    fig, ax = plt.subplots(1, 1, sharey=False, figsize=(5.5,5.5))
    ax.set_xlabel(r'$\sigma b/(v_\mathrm{c}B_0)$',fontsize=fntsize)
    ax.set_ylabel(r'$B/B_0$',fontsize=fntsize)
    ax.set_title("averaged over $\\vartheta$",fontsize=fntsize)
    sig_norm = np.linspace(0,3.5,500)
    ax.axis((0,sig_norm[-1],0.5,4.5))
    for X in metal:
        sig0 = Y[X].ct*vcrit_smallest[X]*B0[X]/Y[X].burgers
        ax.plot(sigma[X]/sig0,B_of_sig[X]/B0[X],label="{}, $B_0\!=\!{:.1f}\mu$Pas".format(X,1e6*B0[X]))
    ax.plot(sig_norm,np.sqrt(1+sig_norm**2),':',color='black',label="$\sqrt{1+\\left(\\frac{\sigma b}{v_\mathrm{c}B_0}\\right)^2}$")
    ax.plot(sig_norm,0.25 + sig_norm,':',color='green',label="$0.25+\\frac{\sigma b}{v_\mathrm{c}B_0}$")
    plt.xticks(fontsize=fntsize)
    plt.yticks(fontsize=fntsize)
    ax.legend(loc='best', ncol=2, columnspacing=0.8, handlelength=1.1, frameon=False, shadow=False,fontsize=fntsize-1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig("B_of_sigma_all.pdf",format='pdf',bbox_inches='tight')
    plt.close()
    