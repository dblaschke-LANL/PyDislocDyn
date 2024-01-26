#!/usr/bin/env python3
# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Jan. 16, 2024
'''This script will calculate the drag coefficient from phonon wind for anisotropic crystals and generate nice plots;
   it is not meant to be used as a module.
   The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
   metals that are predefined in metal_data.py, falling back to all available if no argument is passed.'''
#################################
import sys
import os
import ast
import shutil, lzma
import copy
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar, fsolve
##################
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
import matplotlib.pyplot as plt
##### use pdflatex and specify font through preamble:
# mpl.use("pgf")
# plt.rcParams.update({
#     "text.usetex": True, 
#     "text.latex.preamble": r"\usepackage{fouriernc}",
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "pgf.preamble": "\n".join([
#           r"\usepackage[utf8x]{inputenc}",
#           r"\usepackage[T1]{fontenc}",
#           r"\usepackage{fouriernc}",
#           r"\usepackage{amsmath}",
#     ]),
# })
##################
plt.rc('font',**{'family':'Liberation Serif','size':'11'})
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import pandas as pd
## workaround for spyder's runfile() command when cwd is somewhere else:
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
import metal_data as data
from dislocations import ompthreads, printthreadinfo
from linetension_calcs import readinputfile, Dislocation, read_2dresults, parse_options, str2bool, Ncores, Ncpus
from phononwind import phonondrag
if Ncores>1:
    from joblib import Parallel, delayed
Kcores = max(Ncores,int(min(Ncpus/2,Ncores*ompthreads/2))) ## use this for parts of the code where openmp is not supported

### choose various resolutions and other parameters:
Ntheta = 21 # number of dislocation character angles between 0 and pi/2 (minimum 2, i.e. pure edge and pure screw), if the range -pi/2--pi/2 is required the number of angles is increased to 2*Ntheta-1
Nbeta = 99 # number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)
minb = 0.01
maxb = 0.99
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
skip_plots=False ## set to True to skip generating plots from the results
use_exp_Lame=True ## if set to True, experimental values (where available) are taken for the Lame constants, isotropic phonon spectrum, and sound speeds
## missing values (such as Mo, Zr, or all if use_exp_Lame=False) are supplemented by Hill averages, or for cubic crystals the 'improved average' (see 'polycrystal_averaging.py')
use_iso=False ## set to True to calculate using isotropic elastic constants from metal_data (no effect if input files are used)
## choose among predefined slip systems when using metal_data.py (see that file for details)
bccslip = '110' ## allowed values: '110' (default), '112', '123', 'all' (for all three)
hcpslip = 'basal' ## allowed values: 'basal', 'prismatic', 'pyramidal', 'all' (for all three)
#####
skiptransonic = True ### if True (default) will skip phononwind calcs. for velocities above the lowest limiting velocity on a per character angle basis, filling in the blanks with np.inf
### Note: this will speed up calculations by avoiding slow converging drag calcs near divergences of the dislocation field
#####
NT = 1 # number of temperatures between baseT and maxT (WARNING: implementation of temperature dependence is incomplete!)
constantrho = False ## set to True to override thermal expansion coefficient and use alpha_a = 0 for T > baseT
increaseTby = 300 # so that maxT=baseT+increaseTby (default baseT=300 Kelvin, but may be overwritten by an input file below)
beta_reference = 'base'  ## define beta=v/ct, choosing ct at baseT ('base') or current T ('current') as we increase temperature
#####
# in Fourier space:
Nphi = 50 # keep this an even number for higher accuracy (because we integrate over pi-periodic expressions in some places and phi ranges from 0 to 2pi)
Nq = 50 # only used in Fourier trafo of disloc. field, don't need such high resolution if cutoffs are chosen carefully since the q-dependence drops out in that case
# in x-space (used in numerical Fourier trafo):
NphiX = 3000
# cutoffs for r to be used in numerical Fourier trafo (in units of pi/qBZ)
### rmin smaller converges nicely, rmax bigger initially converges but if it gets to large (several hundred) we start getting numerical artifacts due to rapid oscillations
rmin = 0
rmax = 250
## the following options can be set on the commandline with syntax --keyword=value:
phononwind_opts = {} ## pass additional options to dragcoeff_iso() of phononwind.py
OPTIONS = {"Ncores":int, "Ntheta":int, "Nbeta":int, "minb":float, "maxb":float, "modes":str, "skip_plots":str2bool, "use_exp_Lame":str2bool, "use_iso":str2bool,\
           "bccslip":str, "hcpslip":str, "skiptransonic":str2bool, "NT":int, "constantrho":str2bool, "increaseTby":float, "beta_reference":str,\
           "Nphi":int, "Nq":int, "NphiX":int, "rmin":float, "rmax":float, "phononwind_opts":ast.literal_eval}

### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. at least TOEC)
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(data.c123.keys())))

### define various functions for fitting drag results and for computing B(sigma) from (fitted) B(v)
def fit_mix(x, c0, c1, c2, c4):
    '''Defines a fitting function appropriate for drag coefficient B(v).'''
    return c0 - c1*x + c2*(1/(1-x**2)**(1/2) - 1) + c4*(1/(1-x**2)**(3/2) - 1)

def mkfit_Bv(Y,Bdrag,scale_plot=1,Bmax_fit='auto'):
    '''Calculates fitting functions for B(v) for pure screw, pure edge, and an average over all dislocation characters.
       Required inputs are an instance of the Dislocation class Y, and the the drag coefficient Bdrag formatted as a Pandas DataFrame where index
       contains the normalized velocities beta= v/Y.ct and columns contains the character angles theta at velocity v for all character angles theta.'''
    # if not isinstance(Y,Dislocation): ## this will fail if comparing pydislocdyn.linetension_calcs.Dislocation to linetension_calcs.Dislocation
    #     raise ValueError("'Y' must be an instance of the Dislocation class")
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
    beta_avercrit =  (vel/Y.vcrit_smallest)[vel<Y.vcrit_smallest]
    ### having cut everything beyond the critical velocities (where B diverges), we additionally remove very high values (which are likely inaccurate close to vcrit) to improve the fits everywhere else; adjust Bmax_fit to your needs!
    beta_edgecrit = beta_edgecrit[[j for j in range(len(beta_edgecrit)) if Broom[j,-1] <Bmax_fit + np.min(Broom[:,-1])]]
    beta_screwcrit = beta_screwcrit[[j for j in range(len(beta_screwcrit)) if Broom[j,Y.scrind]<Bmax_fit + np.min(Broom[:,Y.scrind])]]
    beta_avercrit =  beta_avercrit[[j for j in range(len(beta_avercrit)) if Y.Baver[j]<Bmax_fit + np.min(Y.Baver)]]
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
    # if not isinstance(Y,Dislocation): ## this will fail if comparing pydislocdyn.linetension_calcs.Dislocation to linetension_calcs.Dislocation
    #     raise ValueError("'Y' must be an instance of the Dislocation class")
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
        if not themin.success or (zero>1e-5 and zero/bsig>1e-2):
            # print(f"Warning: bad convergence for vr({stress=}): eq={zero:.6f}, eq/(burg*sig)={zero/bsig:.6f}")
            # fall back to (slower) fsolve:
            out = fsolve(nonlinear_equation,0.01*vcrit)[0]
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
    Boffset = max( float(B(vr(sigma_max))-Bstraight(sigma_max,0)) , 0) ## don't allow negative values
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

#########
if __name__ == '__main__':
    Y={}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args, kwargs = parse_options(sys.argv[1:],OPTIONS,globals())
        phononwind_opts.update(kwargs)
    phononwind_opts['modes']=modes
    printthreadinfo(Ncores,ompthreads)
    ### set range & step sizes (array of character angles theta is generated for every material independently below)
    beta = np.linspace(minb,maxb,Nbeta)
    phi = np.linspace(0,2*np.pi,Nphi)
    phiX = np.linspace(0,2*np.pi,NphiX)
    isokeywd='omit' ## writeinputfile(..., iso='omit') will bypass writing ISO_c44 values to the input files and missing Lame constants will always be auto-generated by averaging
    if use_exp_Lame:
        isokeywd=False
    if use_iso:
        metal = sorted(list(set(metal).intersection(data.ISO_c44.keys()).intersection(data.ISO_l.keys())))
        isokeywd=True
    if len(sys.argv) > 1 and len(args)>0:
        try:
            inputdata = [readinputfile(i, Nphi=NphiX, Ntheta=Ntheta) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = metal_list = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = args[0].split()
        
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
        with open("beta.dat","w", encoding="utf8") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta)))
        for X in metal:
            with open(X+".log", "w", encoding="utf8") as logfile:
                logfile.write(Y[X].__repr__())
                logfile.write("\n\nbeta =v/ct:\n")
                logfile.write('\n'.join(map("{:.5f}".format,beta)))
                logfile.write("\n\ntheta:\n")
                logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
        
        print(f"Computing the drag coefficient from phonon wind ({modes} modes) for: {metal}")
    
    ###
    highT = {}
    rotmat = {}
    for X in metal:
        highT[X] = np.linspace(Y[X].T,Y[X].T+increaseTby,NT)
        if constantrho:
            Y[X].alpha_a = 0
        ## only write temperature to files if we're computing temperatures other than baseT=Y[X].T
        if len(highT[X])>1 and Ncores !=0:
            with open(X+".log", "a", encoding="utf8") as logfile:
                logfile.write("\n\nT:\n")
                logfile.write('\n'.join(map("{:.2f}".format,highT[X])))
        
    for X in metal:
        if Ncores != 0:
            Bmix = np.zeros((len(beta),Y[X].Ntheta,len(highT[X])))
            Bmix[:,:,0] = phonondrag(Y[X], beta, Ncores=Ncores, skiptransonic=skiptransonic, pandas_out=False, **phononwind_opts)
            for Ti in range(len(highT[X])-1):
                Z = copy.copy(Y[X]) ## local copy we can modify for higher T
                Z.T = highT[X][Ti+1]
                expansionratio = (1 + Y[X].alpha_a*(Z.T - Y[X].T)) ## TODO: replace with values from eos!
                Z.qBZ = Y[X].qBZ/expansionratio
                Z.burgers = Y[X].burgers*expansionratio
                Z.rho = Y[X].rho/expansionratio**3
                Z.mu = Y[X].mu ## TODO: need to implement T dependence of shear modulus!
                Z.lam = Y[X].bulk - 2*Z.mu/3 ## TODO: need to implement T dependence of bulk modulus!
                Z.ct = np.sqrt(Z.mu/Z.rho)
                Z.ct_over_cl = np.sqrt(Z.mu/(Z.lam+2*Z.mu))
                Z.cl = Z.ct/Z.ct_over_cl
                ## beta, as it appears in the equations, is v/ctT, therefore:
                if beta_reference == 'current':
                    betaT = beta
                else:
                    betaT = beta*Y[X].ct/Z.ct
                ###### T dependence of elastic constants (TODO)
                Z.c11 = Y[X].c11
                Z.c12 = Y[X].c12
                Z.c44 = Y[X].c44
                Z.c13 = Y[X].c13
                Z.c33 = Y[X].c33
                Z.c66 = Y[X].c66
                Z.c111 = Y[X].c111
                Z.c112 = Y[X].c112
                Z.c113 = Y[X].c113
                Z.c123 = Y[X].c123
                Z.c133 = Y[X].c133
                Z.c144 = Y[X].c144
                Z.c155 = Y[X].c155
                Z.c166 = Y[X].c166
                Z.c222 = Y[X].c222
                Z.c333 = Y[X].c333
                Z.c344 = Y[X].c344
                Z.c366 = Y[X].c366
                Z.c456 = Y[X].c456
                ##
                Z.cij = Y[X].cij
                Z.cijk = Y[X].cijk
                ###
                Z.init_C2()
                Z.init_C3()
                Bmix[:,:,Ti+1] = phonondrag(Z, betaT, Ncores=Ncores, skiptransonic=skiptransonic, pandas_out=False, **phononwind_opts)
        
            # and write the results to disk
            if os.access(fname:=f"drag_anis_{X}.dat.xz", os.R_OK):
                shutil.move(fname,fname[:-3]+".bak.xz")
            with lzma.open(f"drag_anis_{X}.dat.xz","wt") as Bfile:
                Bfile.write(f"### B(beta,theta) for {X} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.\n")
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,Y[X].theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    Bfile.write(f"{beta[bi]:.4f}\t" + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT[X])>1 and Ncores !=0:
            if os.access(fname:=f"drag_anis_T_{X}.dat.xz", os.R_OK):
                shutil.move(fname,fname[:-3]+".bak.xz")
            with lzma.open(fname:=f"drag_anis_T_{X}.dat.xz","wt") as Bfile:
                Bfile.write(fr"### B(T,beta,theta[pi]) for {X} in units of mPas. Read this as pandas multiindex dataframe using .read_csv({fname},skiprows=1,index_col=[0,1],sep='\t')."+"\n")
                Bfile.write('temperature[K]\tbeta\t' + '\t'.join(map("{:.4f}".format,Y[X].theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    for Ti in range(len(highT[X])):
                        Bfile.write(f"{highT[X][Ti]:.1f}\t{beta[bi]:.4f}\t" + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')

    #############################################################################################################################

    ## compute smallest critical velocity in ratio (for those data provided in metal_data) to the scaling velocity and plot only up to this velocity
    velm0 = {}
    for X in metal:
        Y[X].computevcrit()
        Y[X].findvcrit_smallest()
        ## compute sound wave speeds for sound waves propagating parallel to screw/edge dislocation glide for comparison:
        velm0[X] = np.round(Y[X].m0,15)
        edgescrewind = Y[X].findedgescrewindices()
        Y[X].sound_screw = Y[X].computesound(velm0[X][edgescrewind[0]])
        Y[X].sound_edge = Y[X].computesound(velm0[X][edgescrewind[1]])
        
    if skip_plots:
        print("skipping plots as requested")
        sys.exit()
    
    ###### plot room temperature results:
    print("Creating plots")
    ## load data from semi-isotropic calculation
    for X in metal:
        Y[X].Broom = read_2dresults(f"drag_anis_{X}.dat")
        beta = Y[X].Broom.index.to_numpy()
        Y[X].beta_trunc = [j for j in beta if j <=Y[X].vcrit_smallest/Y[X].ct]
    
    ## plot B(beta=0.01 ) against theta for every metal:
    def mksmallbetaplot(X,ylab=True,xlab=True,bt=0,sinex=False):
        '''Plot the drag coefficient at low velocity over the character angle.'''
        B_trunc = (Y[X].Broom.iloc[:len(Y[X].beta_trunc)]).to_numpy()
        ymax = max(0.01,(int(100*max(B_trunc[bt]))+1)/100)
        ymin = (int(100*min(B_trunc[bt])))/100
        ## if range is too large, cut off top/bottom:
        if ymax-ymin>=0.05:
            ymax = ymax-0.006
            ymin = ymin+0.006
        xvals = Y[X].Broom.columns
        xtickvals = [-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2,5*np.pi/8,3*np.pi/4,7*np.pi/8,np.pi]
        if sinex:
            xlabel = r'$\sin^2\vartheta$'
            xvals = np.sin(xvals)**2
            xtickvals = np.sin([0,np.pi/6,np.pi/4,np.pi/3,np.pi/2])**2
            xticks = (r"$0$", r"$\frac{3}{4}$", r"$\frac{1}{2}$", r"$\frac{1}{4}$", r"$1$")
        else:
            xlabel = r'$\vartheta$'
            xticks = (r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$", r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$")
        plt.yticks(fontsize=fntsize)
        if xlab:
            plt.xticks(xtickvals,xticks,fontsize=fntsize)
            plt.xlabel(xlabel,fontsize=fntsize)
        if ylab:
            plt.ylabel(r'$B(\beta_\mathrm{t}=0.01)$',fontsize=fntsize)
            plt.ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        plt.axis((xvals[0],xvals[-1],ymin,ymax))
        plt.plot(xvals,B_trunc[bt])
    
    ### create colormesh-plots for every metal:
    clbar_frac=0.12
    clbar_pd=0.03
    wrat1=1-clbar_frac-clbar_pd
    wspc=(clbar_frac+clbar_pd)*100/wrat1
    def mkmeshplot(X,ylab=True,xlab=True,colbar=True,Bmin=None,Bmax=None,sinex=False):
        '''Plot the drag coefficient over the character angle and the dislocation velocity.'''
        B_trunc = (Y[X].Broom.iloc[:len(Y[X].beta_trunc)].to_numpy()).T
        y_msh , x_msh = np.meshgrid(Y[X].beta_trunc,Y[X].Broom.columns) ## plots against theta and beta
        if Bmin is None:
            Bmin = (int(1000*np.min(B_trunc)))/1000
        if Bmax is None:
            Bmax = Bmin+0.016
            ## tweak colorbar range defined above:
            if np.sum(B_trunc<=Bmax)/(len(Y[X].Broom.columns)*len(Y[X].beta_trunc))<0.65:
                Bmax = Bmin+0.032 ## if more than 35% of the area is >Bmax, double the range
            elif np.sum(B_trunc>Bmax)/(len(Y[X].Broom.columns)*len(Y[X].beta_trunc))<0.02:
                Bmax = Bmin+0.008 ## if less than 2% of the area is >Bmax, cut the range in half
        namestring = f"{X}"
        plt.xticks(fontsize=fntsize)
        plt.yticks(np.arange(10)/10,fontsize=fntsize)
        cbarlevels = list(np.linspace(Bmin,Bmax,9))
        xtickvals = [-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2,5*np.pi/8,3*np.pi/4,7*np.pi/8,np.pi]
        if sinex:
            xlabel = r'$\sin^2\vartheta$'
            x_msh = np.sin(x_msh)**2
            xtickvals = np.sin([0,np.pi/6,np.pi/4,np.pi/3,np.pi/2])**2
            xticks = (r"$0$", r"$\frac{3}{4}$", r"$\frac{1}{2}$", r"$\frac{1}{4}$", r"$1$")
        else:
            xlabel = r'$\vartheta$'
            xticks = (r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$", r"$\frac{5\pi}{8}$", r"$\frac{3\pi}{4}$", r"$\frac{7\pi}{8}$", r"$\pi$")
        if xlab:
            plt.xticks(xtickvals,xticks,fontsize=fntsize)
            plt.xlabel(xlabel,fontsize=fntsize)
        if ylab:
            plt.ylabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        plt.title(namestring,fontsize=fntsize)
        colmsh=plt.pcolormesh(x_msh,y_msh,B_trunc,vmin=Bmin, vmax=Bmax,cmap = plt.cm.cubehelix_r,shading='gouraud')
        colmsh.set_rasterized(True)
        if colbar:
            cbar = plt.colorbar(fraction=clbar_frac,pad=clbar_pd, ticks=cbarlevels)
            cbar.set_label(r'$B$[mPa$\,$s]', labelpad=-22, y=1.11, rotation=0, fontsize = fntsize)
            cbar.ax.tick_params(labelsize = fntsize)
        plt.contour(x_msh,y_msh,B_trunc, colors=('gray','gray','gray','white','white','white','white','white','white'), levels=cbarlevels, linewidths=0.9, linestyles=['dashdot','solid','dashed','dotted','dashdot','solid','dashed','dotted','dashdot'])
        
    def mkmeshbetaplot(X,sinex=False,**kwargs):
        '''Plot the drag coefficient over the character angle and the dislocation velocity as well as at low velocity over the character angle.'''
        fig = plt.figure(figsize=(4.5,3.6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1,0.25])
        ax0 = fig.add_subplot(gs[0])
        ax0.xaxis.set_minor_locator(AutoMinorLocator())
        ax0.yaxis.set_minor_locator(AutoMinorLocator())
        plt.setp(ax0.get_xticklabels(), visible=False)
        mkmeshplot(X,ylab=True,xlab=False,sinex=sinex,**kwargs)
        ax1 = fig.add_subplot(gs[1] , sharex=ax0)
        ax1.set_yticks(np.arange(11)/100)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        mksmallbetaplot(X,ylab=True,xlab=True,sinex=sinex)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="{}%".format(wspc), pad=0)
        cax.set_facecolor('none')
        for axis in ['top','bottom','left','right']:
            cax.spines[axis].set_linewidth(0)
        cax.set_xticks([])
        cax.set_yticks([])
        fig.tight_layout(pad=0.3)
        plt.savefig(f"B_{X}.pdf",format='pdf',bbox_inches='tight',dpi=300)
        plt.close()
    
    for X in metal:
        mkmeshbetaplot(X,sinex=False) ## set sinex=True to change x axis from theta to sin^2(theta)
    
    ## define line colors for every metal in the same plot
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe110':'darkgreen', 'Nb110':'firebrick', 'Znbasal':'purple', 'Sn':'black', 'Ag':'lightblue', 'Au':'goldenrod', 'Cdbasal':'lightgreen', 'Mgbasal':'lightsalmon', 'Mo110':'magenta', 'Ni':'silver',\
                   'Tibasal':'olive', 'Zrbasal':'cyan', 'Fe112':'yellowgreen', 'Fe123':'olivedrab', 'Mo112':'deeppink', 'Mo123':'hotpink', 'Nb112':'red', 'Nb123':'darkred', 'Cdprismatic':'khaki', 'Cdpyramidal':'darkkhaki',\
                   'Mgprismatic':'deepskyblue', 'Mgpyramidal':'royalblue', 'Tiprismatic':'orange', 'Tipyramidal':'yellow', 'Znprismatic':'gray', 'Znpyramidal':'slateblue', 'Zrprismatic':'darkcyan', 'Zrpyramidal':'darkturquoise'}
    
    popt_edge = {}
    pcov_edge = {}
    popt_screw = {}
    pcov_screw = {}
    popt_aver = {}
    pcov_aver = {}
    scale_plot = 1 ## need to increase plot and fitting range for higher temperatures
    for X in metal:
        popt_edge[X], pcov_edge[X], popt_screw[X], pcov_screw[X], popt_aver[X], pcov_aver[X] = mkfit_Bv(Y[X],Y[X].Broom,scale_plot=scale_plot)
        scale_plot=max(scale_plot,Y[X].scale_plot)
    
    with open("drag_semi_iso_fit.txt","w", encoding="utf8") as fitfile:
        fitfile.write("Fitting functions for B[$\\mu$Pas] at room temperature:\nEdge dislocations:\n")
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
        fitfile.write("\n $c_{\\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].ct:.0f}")
        fitfile.write(" \\\\\n $v_c^{\\mathrm{e}}/c_{\\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].vcrit_edge/Y[X].ct:.3f}")
        fitfile.write("\n\\\\\\hline\\hline")
        fitfile.write("\n $v_c^{\\mathrm{s}}/c_{\\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].vcrit_screw/Y[X].ct:.3f}")
        fitfile.write("\n\\\\\\hline\\hline")
        fitfile.write("\n $v_c^{\\mathrm{av}}/c_{\\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].vcrit_smallest/Y[X].ct:.3f}")
        
    def mkfitplot(metal_list,filename,figtitle,scale_plot=1):
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
                vcrit = Y[X].vcrit_edge/Y[X].ct
                popt = popt_edge[X]
                cutat = 1.001*vcrit ## vcrit is often rounded, so may want to plot one more point in some cases
                B = Y[X].Broom.iloc[beta<cutat,-1].to_numpy()
            elif filename=="screw":
                vcrit = Y[X].vcrit_screw/Y[X].ct
                popt = popt_screw[X]
                cutat = 1.0*vcrit
                B = Y[X].Broom.iloc[beta<cutat,Y[X].scrind].to_numpy()
            elif filename=="aver":
                vcrit = Y[X].vcrit_smallest/Y[X].ct
                popt = popt_aver[X]
                cutat = 1.007*vcrit
                B = Y[X].Baver[beta<cutat]
            else:
                raise ValueError(f"keyword {filename=} undefined.")
            if X in metalcolors.keys():
                ax.plot(beta[beta<cutat],B,color=metalcolors[X],label=X)
            else:
                ax.plot(beta[beta<cutat],B,label=X) ## fall back to automatic colors
            beta_highres = np.linspace(0,vcrit,1000)
            with np.errstate(divide='ignore'):
                ax.plot(beta_highres,fit_mix(beta_highres/vcrit,*popt),':',color='gray')
        ax.legend(loc='upper left', ncol=ncols, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig(f"B_{filename}+fits.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge","pure edge",scale_plot)
    mkfitplot(metal,"screw","pure screw",scale_plot)
    mkfitplot(metal,"aver","averaged over $\\vartheta$",scale_plot)
    
    ### finally, also plot B as a function of stress using the fits computed above
    def plotall_B_of_sigma(character,ploteach=False):
        '''plot dislocation drag over stress'''
        B_of_sig = {}
        sigma = {}
        B0 = {}
        vc = {}
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
            popt = popt_screw
        elif character=='edge':
            ax.set_title("edge",fontsize=fntsize)
            fname = "B_of_sigma_all_edge.pdf"
            popt = popt_edge
        else:
            ax.set_title("averaged over $\\vartheta$",fontsize=fntsize)
            fname = "B_of_sigma_all.pdf"
            popt = popt_aver
        for X in metal:
            Xc = X+character
            B0[Xc], vc[Xc], sigma[Xc], B_of_sig[Xc] = B_of_sigma(Y[X],popt[X],character,mkplot=ploteach,B0fit='weighted',indirect=False)
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
