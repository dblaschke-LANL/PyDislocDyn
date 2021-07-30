# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - July 27, 2021
'''This script will calculate the drag coefficient from phonon wind for anisotropic crystals and generate nice plots;
   it is not meant to be used as a module.
   The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
   metals that are predefined in metal_data.py, falling back to all available if no argument is passed.'''
#################################
import sys
import os
import numpy as np
from scipy.optimize import curve_fit, fmin, fsolve
from scipy import ndimage
##################
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
##### use pdflatex and specify font through preamble:
# mpl.use("pgf")
# pgf_with_pdflatex = {
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{fouriernc} \usepackage{amsmath}"
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
## workaround for spyder's runfile() command when cwd is somewhere else:
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
import metal_data as data
from elasticconstants import elasticC2, elasticC3, Voigt, UnVoigt
import dislocations as dlc
from linetension_calcs import readinputfile, Dislocation
from phononwind import elasticA3, dragcoeff_iso
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    ## Ncores=0 bypasses phononwind calculations entirely and only generates plots using data from a previous run
    Ncores = max(1,int(Ncpus/max(2,dlc.ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    # Ncores = -2
except ImportError:
    Ncpus = 1
    Ncores = 1 ## must be 1 (or 0) without joblib
Kcores = max(Ncores,int(min(Ncpus/2,Ncores*dlc.ompthreads/2))) ## use this for parts of the code where openmp is not supported
if Ncores==0: Kcores=max(1,int(Ncpus/2)) # in case user has set Ncores=0 above to bypass phonon wind calcs

### choose various resolutions and other parameters:
Ntheta = 21 # number of dislocation character angles between 0 and pi/2 (minimum 2, i.e. pure edge and pure screw), if the range -pi/2--pi/2 is required the number of angles is increased to 2*Ntheta-1
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
computevcrit_for_speed = None ### Optional: Unless None or 0, this integer variable is the number of theta angles (i.e. np.linspace(theta[0],theta[-1],computevcrit_for_speed)) for which we calculate vcrit explicitly, missing values will be interpolated to match len(theta);
### if provided, drag computations will be skipped for velocities bt>vcrit/ct on a per theta-angle basis
### Note: this may speed up calculations by avoiding slow converging drag calcs near divergences of the dislocation field, but not necessarily since computevcrit takes time as well
#####
NT = 1 # number of temperatures between baseT and maxT (WARNING: implementation of temperature dependence is incomplete!)
constantrho = False ## set to True to override thermal expansion coefficient and use alpha_a = 0 for T > baseT
increaseTby = 300 # so that maxT=baseT+increaseTby (default baseT=300 Kelvin, but may be overwritten by an input file below)
beta_reference = 'base'  ## define beta=v/ct, choosing ct at baseT ('base') or current T ('current') as we increase temperature
#####
# in Fourier space:
Nphi = 50 # keep this (and other Nphi below) an even number for higher accuracy (because we integrate over pi-periodic expressions in some places and phi ranges from 0 to 2pi)
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
### and range & step sizes (array of character angles theta is generated for every material independently below)
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
    dlc.printthreadinfo(Ncores,dlc.ompthreads)
    Y={}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        try:
            inputdata = [readinputfile(i, Nphi=NphiX, Ntheta=Ntheta) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = metal_list = list(Y.keys())
            use_metaldata=False
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
            if X in bcc_metals and '112' not in X:
                Y[X] = readinputfile(X, Nphi=NphiX, Ntheta=Ntheta, symmetric=False)
            else:
                Y[X] = readinputfile(X, Nphi=NphiX, Ntheta=Ntheta, symmetric=True)
        os.chdir("..")
        metal = metal_list
    
    if Ncores == 0:
        print("skipping phonon wind calculations as requested")
    else:
        with open("beta.dat","w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta)))
        for X in metal:
            with open(X+".log", "w") as logfile:
                logfile.write(Y[X].__repr__())
                logfile.write("\n\nbeta =v/ct:\n")
                logfile.write('\n'.join(map("{:.5f}".format,beta)))
                logfile.write("\n\ntheta:\n")
                logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
        
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
    rotmat = {}
    linet = {}
    velm0 = {}
    cache = [] ## store cached intermediate sympy results of computevcrit_stroh method to speed up subsequent calculations
    for X in metal:
        highT[X] = np.linspace(Y[X].T,Y[X].T+increaseTby,NT)
        if constantrho:
            Y[X].alpha_a = 0
        ## only write temperature to files if we're computing temperatures other than baseT=Y[X].T
        if len(highT[X])>1 and Ncores !=0:
            with open("temperatures_{}.dat".format(X),"w") as Tfile:
                Tfile.write('\n'.join(map("{:.2f}".format,highT[X])))
        
        linet[X] = np.round(Y[X].t,15)
        velm0[X] = np.round(Y[X].m0,15)
        Y[X].computerot()
        rotmat[X] = np.round(Y[X].rot,15)
               
        Y[X].C2norm = UnVoigt(Y[X].C2/Y[X].mu)  ## this must be the same mu that was used to define the dimensionless velocity beta, as both enter dlc.computeuij() on equal footing below!
        C3 = UnVoigt(Y[X].C3/Y[X].mu)
        A3 = elasticA3(Y[X].C2norm,C3)
        A3rotated[X] = np.zeros((Y[X].Ntheta,3,3,3,3,3,3))
        for th in range(Y[X].Ntheta):
            rotm = rotmat[X][th]
            A3rotated[X][th] = np.round(np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(A3,rotm.T)))))),12)
        
    if computevcrit_for_speed is not None and computevcrit_for_speed>0:
        print("Computing vcrit for {} character angles, as requested ...".format(computevcrit_for_speed))
        for X in metal:
            if Y[X].Ntheta==Ntheta:
                Y[X].theta_vcrit = np.linspace(Y[X].theta[0],Y[X].theta[-1],computevcrit_for_speed)
            else:
                Y[X].theta_vcrit = np.linspace(Y[X].theta[0],Y[X].theta[-1],2*computevcrit_for_speed-1)
            Y[X].computevcrit(Y[X].theta_vcrit,cache=cache,Ncores=Kcores)
            if computevcrit_for_speed != Ntheta:
                Y[X].vcrit_inter = ndimage.interpolation.zoom(Y[X].vcrit_all[1],Y[X].Ntheta/len(Y[X].theta_vcrit))
            else: Y[X].vcrit_inter = Y[X].vcrit_all[1]
        print("Done; proceeding ...")
    
    def maincomputations(bt,X,modes=modes):
        '''wrap all main computations into a single function definition to be run in a parallelized loop'''
        Bmix = np.zeros((Y[X].Ntheta,len(highT[X])))
        ### compute dislocation displacement gradient uij, then its Fourier transform dij:
        Y[X].computeuij(beta=bt)
        Y[X].alignuij()
        # uij_iso = dlc.computeuij_iso(bt,Y[X].ct_over_cl, theta, phiX)
        # r = np.exp(np.linspace(np.log(Y[X].burgers/5),np.log(100*Y[X].burgers),125))
        ## perhaps better: relate directly to qBZ which works for all crystal structures (rmin/rmax defined at the top of this file)
        # r = np.exp(np.linspace(np.log(rmin*np.pi/Y[X].qBZ),np.log(rmax*np.pi/Y[X].qBZ),Nr))
        # q = Y[X].qBZ*np.linspace(0,1,Nq)
        # dij = np.average(dlc.fourieruij(dislocation[X].uij_aligned,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
        dij = dlc.fourieruij_nocut(Y[X].uij_aligned,phiX,phi,sincos=sincos_noq)
        if computevcrit_for_speed is None or computevcrit_for_speed<=0:
            skip_theta = None
        else:
            skip_theta = bt < Y[X].vcrit_inter/Y[X].ct
        if np.all(skip_theta==False):
            Bmix[:,0] = np.repeat(np.inf,Y[X].Ntheta)
        else:
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3rotated[X], qBZ=Y[X].qBZ, ct=Y[X].ct, cl=Y[X].cl, beta=bt, burgers=Y[X].burgers, T=Y[X].T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1, skip_theta=skip_theta)
        
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
            ##
            cijT = Y[X].cij
            cijkT = Y[X].cijk
            ###
            C2T = elasticC2(c11=c11T, c12=c12T, c44=c44T, c13=c13T, c33=c33T, c66=c66T, cij=cijT)/muT
            C3T = elasticC3(c111=c111T, c112=c112T, c113=c113T, c123=c123T, c133=c133T, c144=c144T, c155=c155T, c166=c166T, c222=c222T, c333=c333T, c344=c344T, c366=c366T, c456=c456T, cijk=cijkT)/muT
            A3T = elasticA3(C2T,C3T)
            A3Trotated = np.zeros((Y[X].Ntheta,3,3,3,3,3,3))
            for th in range(Y[X].Ntheta):
                rotm = rotmat[X][th]
                A3Trotated[th] = np.round(np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(rotm,np.dot(A3T,rotm.T)))))),12)
            ##########################
            Y[X].computeuij(beta=betaT, C2=C2T) ## Y[X].C2norm will be overwritten with C2T here
            Y[X].alignuij()
            ## rT*qT = r*q, so does not change anything
            # dij = np.average(dlc.fourieruij(dislocation[X].uij_aligned,r,phiX,q,phi,sincos)[:,:,:,3:-4],axis=3)
            dij = dlc.fourieruij_nocut(Y[X].uij_aligned,phiX,phi,sincos=sincos_noq)
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
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,Y[X].theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT[X])>1 and Ncores !=0:
            with open("drag_anis_T_{}.dat".format(X),"w") as Bfile:
                Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,Y[X].theta[1:-1])) + '\tBedge[mPas]' + '\n')
                for bi in range(len(beta)):
                    for Ti in range(len(highT[X])):
                        Bfile.write("{:.1f}".format(highT[X][Ti]) +'\t' + "{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')
                
            with open("drag_anis_T_screw_{}.dat".format(X),"w") as Bscrewfile:
                for bi in range(len(beta)):
                    Bscrewfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,0])) + '\n')
            
            with open("drag_anis_T_edge_{}.dat".format(X),"w") as Bedgefile:
                for bi in range(len(beta)):
                    Bedgefile.write('\t'.join(map("{:.6f}".format,Bmix[bi,-1])) + '\n')
    
            for th in range(len(Y[X].theta[1:-1])):
                with open("drag_anis_T_mix{0:.6f}_{1}.dat".format(Y[X].theta[th+1],X),"w") as Bmixfile:
                    for bi in range(len(beta)):
                        Bmixfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,th+1])) + '\n')

    #############################################################################################################################

    ## compute smallest critical velocity in ratio (for those data provided in metal_data) to the scaling velocity and plot only up to this velocity
    vcrit_screw = {}
    vcrit_edge = {}
    vcrit_smallest = {}
    for X in metal:
        if Y[X].sym=='iso':
            vcrit_smallest[X] = 1
            Y[X].c11 = Y[X].C2[0,0] # not previously set, but needed below
        else:
            vcrit_smallest[X] = min(np.sqrt(Y[X].c44/Y[X].mu),np.sqrt((Y[X].c11-Y[X].c12)/(2*Y[X].mu)))
    ## above is correct for fcc, bcc (except 123 planes) and some (but not all) hcp, i.e. depends on values of SOEC and which slip plane is considered;
    ## numerically determined values at T=300K for metals in metal_data.py (rounded):
    if use_metaldata and not use_iso:
        if use_exp_Lame:
            vcrit_smallest['Sn'] = 0.818
            vcrit_smallest['Znbasal'] = 0.943 ## for basal slip
            vcrit_smallest['Fe123'] = 0.735 ## for 123 slip plane
        else:
            for X in set(metal).intersection({'Sn','Znbasal','Fe123'}):
                Y[X].findvcrit_smallest(cache=cache,Ncores=Kcores)
    if use_metaldata:
        ## use exact analytic results where we have them:
        for X in data.fcc_metals.intersection(metal):
            vcrit_screw[X] = np.sqrt(3*Y[X].c44*(Y[X].c11-Y[X].c12)/(2*Y[X].mu*(Y[X].c44+Y[X].c11-Y[X].c12)))
        for X in bcc_metals:
            if '112' in X:
                vcrit_edge[X] = Y[X].computevcrit_edge()
            elif ('110' in X or '123' in X):
                Y[X].computevcrit_stroh(2,symmetric=False,cache=cache,Ncores=Kcores)
                Y[X].vcrit_screw = np.min(Y[X].vcrit[0,1])
                Y[X].vcrit_edge = min(np.min(Y[X].vcrit[0,0]),np.min(Y[X].vcrit[0,2]))
        for X in hcp_metals:
            if 'prismatic' in X:
                vcrit_screw[X] = np.sqrt(Y[X].c44/Y[X].mu)
                vcrit_edge[X] = Y[X].computevcrit_edge() # often equals np.sqrt((Y[X].c11-Y[X].c12)/(2*Y[X].mu))
            elif 'basal' in X:
                vcrit_screw[X] = np.sqrt((Y[X].c11-Y[X].c12)/(2*Y[X].mu))
                vcrit_edge[X] = Y[X].computevcrit_edge() # often equals np.sqrt(Y[X].c44/Y[X].mu)
            elif 'pyramidal' in X:
                vcrit_screw[X] = np.sqrt(Y[X].c44*(Y[X].c11-Y[X].c12)*(3*Y[X].ac**2/4+Y[X].cc**2)/(2*(Y[X].c44*3*Y[X].ac**2/4+Y[X].cc**2*(Y[X].c11-Y[X].c12)/2)*Y[X].mu))
        for X in data.tetr_metals.intersection(metal):
            vcrit_screw[X] = np.sqrt(Y[X].c44/Y[X].mu)
            vcrit_edge[X] = Y[X].computevcrit_edge() # for Sn equals vcrit_screw[X]
    if use_metaldata:
        for X in metal:
            if X not in vcrit_screw.keys(): ## fall back to this (if use_metaldata, the values that have not been set yet will be used by this code)
                vcrit_screw[X] = vcrit_smallest[X] ## coincide for the bcc slip system with 112 planes
            if X not in vcrit_edge.keys():
                vcrit_edge[X] = vcrit_smallest[X] ## coincide for the fcc slip system considered above
    if use_metaldata and not use_iso:
        if use_exp_Lame and hcpslip in ('prismatic', 'pyramidal', 'all'):
            vcrit_smallest['Cdprismatic'] = vcrit_smallest['Cdpyramidal'] = 0.948
            vcrit_smallest['Znprismatic'] = vcrit_smallest['Znpyramidal'] = 0.724
        else:
            for X in set(metal).intersection({'Cdprismatic','Cdpyramidal','Znprismatic','Znpyramidal'}):
                Y[X].findvcrit_smallest(cache=cache,Ncores=Kcores)
    
    ## overwrite any of these values with data from input file, if available, or compute estimates on the fly:
    for X in metal:
        if not use_metaldata and Y[X].vcrit_screw is None:
            print("computing missing critical velocity for screw for ",X)
            Y[X].computevcrit_screw() ## only implemented for certain symmetry properties, no result otherwise
        if not use_metaldata and Y[X].vcrit_edge is None:
            print("computing missing critical velocity for edge for ",X)
            Y[X].computevcrit_edge() ## only implemented for certain symmetry properties, no result otherwise
        if not use_metaldata and (Y[X].vcrit_screw is None or Y[X].vcrit_edge is None):
            Y[X].computevcrit_stroh(2,symmetric=False,cache=cache,Ncores=Kcores) ## only compute vcrit if no values are provided in the input file
            if Y[X].vcrit_screw is None: Y[X].vcrit_screw = np.min(Y[X].vcrit[0,1])
            if Y[X].vcrit_edge is None: Y[X].vcrit_edge = min(np.min(Y[X].vcrit[0,0]),np.min(Y[X].vcrit[0,2]))
        ## compute sound wave speeds for sound waves propagating parallel to screw/edge dislocation glide for comparison:
        scrindm0 = int((len(velm0[X])-1)/2)
        if abs(Y[X].theta[scrindm0]) < 1e-12:
            Y[X].sound_screw = Y[X].computesound(velm0[X][scrindm0])
        else:
            Y[X].sound_screw = Y[X].computesound(velm0[X][0])
        Y[X].sound_edge = Y[X].computesound(velm0[X][-1])
        if not use_metaldata and Y[X].vcrit_smallest is None:
            print("computing missing smallest critical velocity for {} ...".format(X))
            Y[X].findvcrit_smallest(cache=cache,Ncores=Kcores)
            vcrit_smallest[X] = Y[X].vcrit_smallest/Y[X].ct
        ## need vcrit in ratio to ct:
        if Y[X].vcrit_smallest is not None:
            vcrit_smallest[X] = Y[X].vcrit_smallest/Y[X].ct
        if Y[X].vcrit_screw is not None:
            vcrit_screw[X] = Y[X].vcrit_screw/Y[X].ct
        if Y[X].vcrit_edge is not None:
            vcrit_edge[X] = Y[X].vcrit_edge/Y[X].ct
        
    if skip_plots:
        print("skipping plots as requested")
        sys.exit()
    
    ###### plot room temperature results:
    print("Creating plots")
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
        '''Plot the drag coefficient at low velocity over the character angle.'''
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
        if xlab:
            plt.xlabel(r'$\vartheta$',fontsize=fntsize)
        if ylab:
            plt.ylabel(r'$B(\beta_\mathrm{t}=0.01)$',fontsize=fntsize)
            plt.ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        plt.plot(theta[X],B_trunc[bt])
    
    ### create colormesh-plots for every metal:
    clbar_frac=0.12
    clbar_pd=0.03
    wrat1=1-clbar_frac-clbar_pd
    wspc=(clbar_frac+clbar_pd)*100/wrat1
    def mkmeshplot(X,ylab=True,xlab=True,colbar=True,Bmin=None,Bmax=None):
        '''Plot the drag coefficient over the character angle and the dislocation velocity.'''
        beta_trunc = [j for j in beta if j <=vcrit_smallest[X]]
        B_trunc = (Broom[X][:len(beta_trunc),1:]).T
        y_msh , x_msh = np.meshgrid(beta_trunc,theta[X]) ## plots against theta and beta
        if Bmin is None:
            Bmin = (int(1000*np.min(B_trunc)))/1000
        if Bmax is None:
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
        if xlab:
            plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2,5*np.pi/8,3*np.pi/4,7*np.pi/8,np.pi],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$", r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$"),fontsize=fntsize)
            plt.xlabel(r'$\vartheta$',fontsize=fntsize)
        if ylab:
            plt.ylabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        plt.title(namestring,fontsize=fntsize)
        colmsh=plt.pcolormesh(x_msh,y_msh,B_trunc,vmin=Bmin, vmax=Bmax,cmap = plt.cm.cubehelix_r,shading='gouraud')
        colmsh.set_rasterized(True)
        if colbar:
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
        cax = divider.append_axes("right", size="{}%".format(wspc), pad=0)
        cax.set_facecolor('none')
        for axis in ['top','bottom','left','right']:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.tight_layout(pad=0.3)
        plt.savefig("B_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=300)
        plt.close()
    
    ## define line colors for every metal in the same plot
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe110':'darkgreen', 'Nb110':'firebrick', 'Znbasal':'purple', 'Sn':'black', 'Ag':'lightblue', 'Au':'goldenrod', 'Cdbasal':'lightgreen', 'Mgbasal':'lightsalmon', 'Mo110':'magenta', 'Ni':'silver', 'Tibasal':'olive', 'Zrbasal':'cyan'}
    metalcolors.update({'Fe112':'yellowgreen', 'Fe123':'olivedrab', 'Mo112':'deeppink', 'Mo123':'hotpink', 'Nb112':'red', 'Nb123':'darkred'})
    metalcolors.update({'Cdprismatic':'khaki', 'Cdpyramidal':'darkkhaki', 'Mgprismatic':'deepskyblue', 'Mgpyramidal':'royalblue', 'Tiprismatic':'orange', 'Tipyramidal':'yellow', 'Znprismatic':'gray', 'Znpyramidal':'slateblue', 'Zrprismatic':'darkcyan', 'Zrpyramidal':'darkturquoise'})
    
    def fit_mix(x, c0, c1, c2, c4):
        '''define a fitting fct.'''
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
        scale_plot = max(scale_plot,int(Y[X].T/30)/10)
        Bmax_fit = int(20*Y[X].T/300)/100 ## only fit up to Bmax_fit [mPas]
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
        fitfile.write(" & "+" & ".join((metal))+r" \\\hline\hline")
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
        '''Plot the dislocation drag over velocity and show the fitting function.'''
        if len(metal_list)<5:
            fig, ax = plt.subplots(1, 1, figsize=(4.,4.))
            ncols=2
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5.5,5.5))
            ncols=3
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
            with np.errstate(divide='ignore'):
                ax.plot(beta_highres,fit_mix(beta_highres/vcrit,*popt),':',color='gray')
        ax.legend(loc='upper left', ncol=ncols, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig("B_{}+fits.pdf".format(filename),format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge","pure edge")
    mkfitplot(metal,"screw","pure screw")
    mkfitplot(metal,"aver","averaged over $\\vartheta$")
    
    
    ### finally, also plot B as a function of stress using the fits computed above
    def B_of_sigma(X,character,mkplot=True,B0fit='weighted',resolution=500,indirect=False):
        '''Computes arrays sigma and B_of_sigma of length 'resolution', and returns a tuple (B0,vcrit,sigma,B_of_sigma) where B0 is either the minimum value, or B(v=0) if B0fit=='zero'
           or a weighted average of the two (B0fit='weighted',default) and vcrit is the critical velocity for character (='screw', 'edge', or else an average is computed).
           A plot of the results is saved to disk if mkplot=True (default).
           If option indirect=False sigma will be evenly spaced (default), whereas if indirect=True sigma will be calculated from an evenly spaced velocity array.
           The latter is also used as fall back behavior if the computation of v(sigma) fails to converge.'''
        if character=='screw':
            vcrit = Y[X].ct*vcrit_screw[X]
            popt = popt_screw[X]
            fname = "B_of_sigma_screw_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "screw"
        elif character=='edge':
            vcrit = Y[X].ct*vcrit_edge[X]
            popt = popt_edge[X]
            fname = "B_of_sigma_edge_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "edge"
        else: ## 'aver' = default
            vcrit = Y[X].ct*vcrit_smallest[X]
            popt = popt_aver[X]
            fname = "B_of_sigma_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "averaged over $\\vartheta$"
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
                # print("Warning: bad convergence for vr(stress={}): eq={:.6f}, eq/(burg*sig)={:.6f}".format(stress,zero,zero/bsig))
                # fall back to (slower) fsolve:
                out = float(fsolve(nonlinear_equation,0.01*vcrit))
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
            
        ## determine stress that will lead to velocity of 99% critical speed and stop plotting there, or at 1.5GPa (whichever is smaller)
        sigma_max = sigma_eff(0.99*vcrit)
        # print("{}, {}: sigma(99%vcrit) = {:.1f} MPa".format(X,character,sigma_max/1e6))
        if sigma_max<6e8 and B(0.99*vcrit)<1e-4: ## if B, sigma still small, increase to 99.9% vcrit
            sigma_max = sigma_eff(0.999*vcrit)
            # print("{}, {}: sigma(99.9%vcrit) = {:.1f} MPa".format(X,character,sigma_max/1e6))
        sigma_max = min(1.5e9,sigma_max)
        Boffset = float(B(vr(sigma_max))-Bstraight(sigma_max,0))
        if Boffset < 0: Boffset=0 ## don't allow negative values
        ## find min(B(v)) to use for B0 in Bsimple():
        B0 = round(np.min(B(np.linspace(0,0.8*vcrit,1000))),7)
        if B0fit == 'weighted':
            B0 = (B(0)+3*B0)/4 ## or use some weighted average between Bmin and B(0)
        elif B0fit == 'zero':
            B0 = B(0)
        # print("{}: Boffset={:.4f}mPas, B0={:.4f}mPas".format(X,1e3*Boffset,1e3*B0))
        
        sigma = np.linspace(0,sigma_max,resolution)
        if not indirect:
            B_of_sig = B(vr(sigma))
            Bmax = B_of_sig[-1]
        if indirect or (np.max(B_of_sig) < 1.01*B(0)):
            # print("\nWARNING: using fall back for v(sigma) for {}, {}\n".format(X,character))
            v = vcrit*np.linspace(0,0.999,resolution)
            sigma = sigma_eff(v)
            B_of_sig = B(v)
            Bmax = B_of_sig[-1]
            if sigma[-1]>1.1*sigma_max:
                Bmax = 1.15*B_of_sig[sigma<sigma_max][-1]
        if mkplot:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(3.,2.5))
            ax.set_xlabel(r'$\sigma$[MPa]',fontsize=fntsize)
            ax.set_ylabel(r'$B$[mPas]',fontsize=fntsize)
            ax.set_title(ftitle,fontsize=fntsize)
            ax.axis((0,sigma_max/1e6,0,Bmax*1e3))
            ax.plot(sigma/1e6,Bsimple(sigma,B0)*1e3,':',color='gray',label=r"$\sqrt{B_0^2\!+\!\left(\frac{\sigma b}{v_\mathrm{c}}\right)^2}$, $B_0=$"+f"{1e6*B0:.1f}"+r"$\mu$Pas")
            ax.plot(sigma/1e6,Bstraight(sigma,Boffset)*1e3,':',color='green',label=r"$B_0+\frac{\sigma b}{v_\mathrm{c}}$, $B_0=$"+f"{1e6*Boffset:.1f}"+r"$\mu$Pas")
            ax.plot(sigma/1e6,B_of_sig*1e3,label=r"$B_\mathrm{fit}(v(\sigma))$")
            plt.xticks(fontsize=fntsize)
            plt.yticks(fontsize=fntsize)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.legend(loc='upper left',handlelength=1.1, frameon=False, shadow=False,fontsize=8)
            plt.savefig(fname,format='pdf',bbox_inches='tight')
            plt.close()
        return (B0,vcrit,sigma,B_of_sig)
        
    def plotall_B_of_sigma(character,ploteach=False):
        '''plot dislocation drag over stress'''
        B_of_sig = {}
        sigma = {}
        B0 = {}
        vc = {}
        for X in metal:
            Xc = X+character
            B0[Xc], vc[Xc], sigma[Xc], B_of_sig[Xc] = B_of_sigma(X,character,mkplot=ploteach,B0fit='weighted',indirect=False)
    
        if len(metal)<5:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.,4.))
            legendops={'loc':'best','ncol':1}
        else:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(5.,5.))
            legendops={'loc':'upper left','bbox_to_anchor':(1.01,1),'ncol':1}
        ax.set_xlabel(r'$\sigma b/(v_\mathrm{c}B_0)$',fontsize=fntsize)
        ax.set_ylabel(r'$B/B_0$',fontsize=fntsize)
        if character=='screw':
            ax.set_title("screw",fontsize=fntsize)
            fname = "B_of_sigma_all_screw.pdf"
        elif character=='edge':
            ax.set_title("edge",fontsize=fntsize)
            fname = "B_of_sigma_all_edge.pdf"
        else:
            ax.set_title("averaged over $\\vartheta$",fontsize=fntsize)
            fname = "B_of_sigma_all.pdf"
        sig_norm = np.linspace(0,3.5,500)
        ax.axis((0,sig_norm[-1],0.5,4.5))
        for X in metal:
            Xc = X+character
            sig0 = vc[Xc]*B0[Xc]/Y[X].burgers
            ax.plot(sigma[Xc]/sig0,B_of_sig[Xc]/B0[Xc],label=fr"{X}, $B_0\!=\!{1e6*B0[Xc]:.1f}\mu$Pas, "+r"$v_\mathrm{c}\!=\!"+f"{vc[Xc]/1e3:.2f}$km/s")
        ax.plot(sig_norm,np.sqrt(1+sig_norm**2),':',color='black',label=r"$\sqrt{1+\left(\frac{\sigma b}{v_\mathrm{c}B_0}\right)^2}$")
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.legend(**legendops, columnspacing=0.8, handlelength=1.1, frameon=False, shadow=False,fontsize=fntsize-1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.savefig(fname,format='pdf',bbox_inches='tight')
        plt.close()
        return (B_of_sig,sigma,B0,vc)
    
    if Kcores ==1: B_of_sig_results = [plotall_B_of_sigma(character) for character in ['aver', 'screw', 'edge']]
    else: B_of_sig_results = Parallel(max_nbytes=None, n_jobs=Kcores)(delayed(plotall_B_of_sigma)(character) for character in ['aver', 'screw', 'edge'])
    B_of_sig,sigma,B0,vc = B_of_sig_results[0]
    for i in [1,2]:
        B_of_sig.update(B_of_sig_results[i][0])
        sigma.update(B_of_sig_results[i][1])
        B0.update(B_of_sig_results[i][2])
        vc.update(B_of_sig_results[i][3])
