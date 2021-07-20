# Compute the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - July 19, 2021
'''This script will calculate the drag coefficient from phonon wind in the isotropic limit and generate nice plots;
   it is not meant to be used as a module.
   The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
   metals that are predefined in metal_data.py, falling back to all available if no argument is passed.'''
#################################
import sys
import os
import numpy as np
from scipy.optimize import curve_fit, fmin, fsolve
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
fntsize=11
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
##################
## workaround for spyder's runfile() command when cwd is somewhere else:
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
import metal_data as data
from elasticconstants import elasticC2, elasticC3, Voigt, UnVoigt
from dislocations import fourieruij_iso, ompthreads
from linetension_calcs import readinputfile, Dislocation
from phononwind import elasticA3, dragcoeff_iso
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    ## Ncores=0 bypasses phononwind calculations entirely and only generates plots using data from a previous run
    Ncores = max(1,int(Ncpus/max(2,ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    # Ncores = -2
    if ompthreads == 0 and Ncores != 0: # check if subroutines were compiled with OpenMP support
        print("using joblib parallelization with ",Ncores," cores")
    elif Ncores != 0:
        print("Parallelization: joblib with ",Ncores," cores and OpenMP with ",ompthreads," threads, total = ",Ncores*ompthreads)
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = 1 ## must be 1 (or 0) without joblib
    if ompthreads > 0:
        print("using OpenMP parallelization with ",ompthreads," threads")

### choose various resolutions and other parameters:
Ntheta = 2 # number of angles between burgers vector and dislocation line (minimum 2, i.e. pure edge and pure screw)
Nbeta = 99 # number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)
minb = 0.01
maxb = 0.99
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
# modes = 'TT'
skip_plots=False ## set to True to skip generating plots from the results
use_exp = True # if using data from metal_data, choose beteen experimentally determined Lame and Murnaghan constants (defaul) or analytical averages of SOEC and TOEC (use_exp = False)
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
### and range & step sizes
beta = np.linspace(minb,maxb,Nbeta)
phi = np.linspace(0,2*np.pi,Nphi)

if use_exp:
    metal = sorted(list(data.fcc_metals.union(data.bcc_metals).intersection(data.ISO_l.keys())))
else:
    metal = sorted(list(data.c111.keys()))

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
                inputdata[i]=readinputfile(args[i],Ntheta=Ntheta,isotropify=True)
                X = inputdata[i].name
                metal_list.append(X)
                Y[X] = inputdata[i]
            use_metaldata=False
            metal = metal_list
            print("success reading input files ",args)
        except FileNotFoundError:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = sys.argv[1].split()
    
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        for X in metal:
            data.writeinputfile(X,X,iso=use_exp) # write temporary input files for requested X of metal_data
            metal_list.append(X)
        for X in metal_list:
            if use_exp:
                Y[X] = readinputfile(X,Ntheta=Ntheta)
            else:
                Y[X] = readinputfile(X,Ntheta=Ntheta,isotropify=True)
        os.chdir("..")
        metal = metal_list
    
    if Ncores == 0:
        print("skipping phonon wind calculations as requested")
    else:
        with open("beta.dat","w") as betafile:
            betafile.write('\n'.join(map("{:.5f}".format,beta)))
        for X in metal:
            with open(X+"_iso.log", "w") as logfile:
                logfile.write(Y[X].__repr__())
                logfile.write("\n\nbeta =v/ct:\n")
                logfile.write('\n'.join(map("{:.5f}".format,beta)))
                logfile.write("\n\ntheta:\n")
                logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
        
        print("Computing the drag coefficient from phonon wind ({} modes) for: ".format(modes),metal)
    
    A3 = {}
    highT = {}
    for X in metal:
        highT[X] = np.linspace(Y[X].T,Y[X].T+increaseTby,NT)
        if constantrho:
            Y[X].alpha_a = 0
        ## only write temperature to files if we're computing temperatures other than baseT=Y[X].T
        if len(highT[X])>1 and Ncores !=0:
            with open("temperatures_{}.dat".format(X),"w") as Tfile:
                Tfile.write('\n'.join(map("{:.2f}".format,highT[X])))
        A3[X] = elasticA3(UnVoigt(Y[X].C2/Y[X].mu), UnVoigt(Y[X].C3/Y[X].mu))
    for X in metal:
        def maincomputations(bt,X,modes=modes):
            '''wrap all main computations into a single function definition to be run in a parallelized loop'''
            Bmix = np.zeros((len(Y[X].theta),len(highT[X])))
                                    
            dij = fourieruij_iso(bt, Y[X].ct_over_cl, Y[X].theta, phi)
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3[X], qBZ=Y[X].qBZ, ct=Y[X].ct, cl=Y[X].cl, beta=bt, burgers=Y[X].burgers, T=Y[X].T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
            
            for Ti in range(len(highT[X])-1):
                T = highT[X][Ti+1]
                expansionratio = (1 + Y[X].alpha_a*(T - Y[X].T)) ## TODO: replace with values from eos!
                qBZT = Y[X].qBZ/expansionratio
                burgersT = Y[X].burgers*expansionratio
                rhoT = Y[X].rho/expansionratio**3
                c44T = Y[X].mu ## TODO: need to implement T dependence of shear modulus!
                c12T = Y[X].bulk - 2*c44T/3 ## TODO: need to implement T dependence of bulk modulus!
                ctT = np.sqrt(c44T/rhoT)
                ct_over_cl_T = np.sqrt(c44T/(c12T+2*c44T))
                clT = ctT/ct_over_cl_T
                ## beta, as it appears in the equations, is v/ctT, therefore:
                if beta_reference == 'current':
                    betaT = bt
                else:
                    betaT = bt*Y[X].ct/ctT
                
                dij = fourieruij_iso(betaT, ct_over_cl_T, Y[X].theta, phi)
                ### TODO: need models for T dependence of TOECs!
                c123T = Y[X].C3[0,1,2]
                c144T = Y[X].C3[0,3,3]
                c456T = Y[X].C3[3,4,5]
                ##
                A3T = elasticA3(elasticC2(c12=c12T, c44=c44T), elasticC3(c123=c123T,c144=c144T,c456=c456T))/c44T
                Bmix[:,Ti+1] = dragcoeff_iso(dij=dij, A3=A3T, qBZ=qBZT, ct=ctT, cl=clT, beta=betaT, burgers=burgersT, T=T, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
                
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
            with open("drag_{}.dat".format(X),"w") as Bfile:
                Bfile.write("### B(beta,theta) for {} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,Y[X].theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT[X])>1 and Ncores !=0:
            with open("drag_T_{}.dat".format(X),"w") as Bfile:
                Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,Y[X].theta[1:-1])) + '\tBedge[mPas]' + '\n')
                for bi in range(len(beta)):
                    for Ti in range(len(highT[X])):
                        Bfile.write("{:.1f}".format(highT[X][Ti]) +'\t' + "{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')
            
            with open("drag_T_screw_{}.dat".format(X),"w") as Bscrewfile:
                for bi in range(len(beta)):
                    Bscrewfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,0])) + '\n')
            
            with open("drag_T_edge_{}.dat".format(X),"w") as Bedgefile:
                for bi in range(len(beta)):
                    Bedgefile.write('\t'.join(map("{:.6f}".format,Bmix[bi,-1])) + '\n')
    
            for th in range(len(Y[X].theta[1:-1])):
                with open("drag_T_mix{0:.6f}_{1}.dat".format(Y[X].theta[th+1],X),"w") as Bmixfile:
                    for bi in range(len(beta)):
                        Bmixfile.write('\t'.join(map("{:.6f}".format,Bmix[bi,th+1])) + '\n')

    #############################################################################################################################

    if skip_plots:
        print("skipping plots as requested")
        sys.exit()
    
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
    lnstyles = {'Al':'-', 'Cu':'--', 'Fe':':', 'Nb':'-.', 'Cd':'-', 'Mg':'--', 'Zn':':', 'Sn':'-.', 'Ni':'-.', 'Mo':'--', 'Ag':':', 'Au':'-.', 'Ti':'-', 'Zr':'-.'}
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe':'green', 'Nb':'red', 'Zn':'purple', 'Sn':'black', 'Ag':'lightblue', 'Au':'goldenrod', 'Cd':'lightgreen', 'Mg':'lightsalmon', 'Mo':'magenta', 'Ni':'silver', 'Ti':'olive', 'Zr':'cyan'}
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
            if X in metalcolors.keys():
                ax.plot(Broom[X][:,0],Broom[X][:,th+1],lnstyles[X], color=metalcolors[X], label=X)
            else:
                ax.plot(Broom[X][:,0],Broom[X][:,th+1],label=X) # fall back to automatic colors / line styles
        legend = ax.legend(loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.2, shadow=True)
        plt.savefig("B_iso_{:.4f}.pdf".format(theta[th]),format='pdf',bbox_inches='tight')
        plt.close()
        
    ## define fitting fcts.:
    if modes=='TT': ## degree of divergence is reduced for purely transverse modes
        print("fitting for transverse modes only (assuming reduced degrees of divergence)")
        def fit_edge(x, c0, c1, c2, c3, c4):
            '''define a fitting function for edge dislocations'''
            return c0 - c1*x + c2*x**2 + c3*np.log(1-x**2) + c4*(1/np.sqrt(1-x**2) - 1)
            
        def fit_screw(x, c0, c1, c2, c3, c4):
            '''define a fitting function for screw dislocations'''
            return c0 - c1*x + c2*x**2 + c3*x**4 + c4*x**16
    else:
        def fit_edge(x, c0, c1, c2, c3, c4):
            '''define a fitting function for edge dislocations'''
            return c0 - c1*x + c2*np.log(1-x**2) + c3*(1/(1-x**2)**(1/2) - 1) + c4*(1/(1-x**2)**(3/2) - 1)
            
        def fit_screw(x, c0, c1, c2, c3, c4):
            '''define a fitting function for screw dislocations'''
            return c0 - c1*x + c2*x**2 + c3*np.log(1-x**2) + c4*(1/np.sqrt(1-x**2) - 1)
    
    popt_edge = {}
    pcov_edge = {}
    popt_screw = {}
    pcov_screw = {}
    Bmax_fit = 0.20 ## only fit up to Bmax_fit [mPas]
    for X in metal:
        popt_edge[X], pcov_edge[X] = curve_fit(fit_edge, beta, Broom[X][:,Ntheta], bounds=([0.9*Broom[X][0,Ntheta],0.,-0.,-0., -0.], [1.1*Broom[X][0,Ntheta], 2*Broom[X][0,Ntheta], 1., 1.,1.]))
        popt_screw[X], pcov_screw[X] = curve_fit(fit_screw, beta, Broom[X][:,1], bounds=([0.9*Broom[X][0,1],0.,-0.,-0.,-0.], [1.1*Broom[X][0,1], 2*Broom[X][0,1], 1., 1., 1.]))
    
    with open("drag_iso_fit.txt","w") as fitfile:
        if modes=='TT': ## degree of divergence is reduced for purely transverse modes
            fitfile.write("Fitting functions for B[$\mu$Pas] at room temperature (transverse modes only):\nEdge dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*log(1-x**2) + {4:.2f}*(1/(1-x**2)**(1/2) - 1)\n".format(*1e3*popt_edge[X]))
            fitfile.write("\nScrew dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*x**4 + {4:.2f}*x**16\n".format(*1e3*popt_screw[X]))
        else:
            fitfile.write("Fitting functions for B[$\mu$Pas] at room temperature:\nEdge dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*log(1-x**2) + {3:.2f}*(1/(1-x**2)**(1/2) - 1) + {4:.2f}*(1/(1-x**2)**(3/2) - 1)\n".format(*1e3*popt_edge[X]))
            fitfile.write("\nScrew dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*log(1-x**2) + {4:.2f}*(1/(1-x**2)**(1/2) - 1)\n".format(*1e3*popt_screw[X]))
        fitfile.write("\nwhere $x=v/c_{\mathrm{t}$\n\n")
        fitfile.write(" & "+" & ".join((metal))+r" \\\hline\hline")
        fitfile.write("\n $c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].ct:.0f}")

    def mkfitplot(metal_list,filename):
        '''Plot the dislocation drag over velocity and show the fitting function.'''
        fig, ax = plt.subplots(1, 1, figsize=(4.5,4.))
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.set_xticks(np.arange(11)/10)
        ax.set_yticks(np.arange(12)/100)
        ax.axis((0,maxb,0,0.11)) ## define plot range
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(r'$\beta_\mathrm{t}$',fontsize=fntsize)
        ax.set_ylabel(r'$B$[mPa$\,$s]',fontsize=fntsize)
        ax.set_title(filename,fontsize=fntsize)
        for X in metal_list:
            beta_highres = np.linspace(0,1,1000)
            if filename=="edge":
                if X in metalcolors.keys():
                    ax.plot(beta,Broom[X][:,Ntheta],color=metalcolors[X],label=X)
                else:
                    ax.plot(beta,Broom[X][:,Ntheta],label=X) # fall back to automatic colors
                with np.errstate(all='ignore'):
                    ax.plot(beta_highres,fit_edge(beta_highres,*popt_edge[X]),':',color='gray')
            elif filename=="screw":
                if X in metalcolors.keys():
                    ax.plot(beta,Broom[X][:,1],color=metalcolors[X],label=X)
                else:
                    ax.plot(beta,Broom[X][:,1],label=X) # fall back to automatic colors
                with np.errstate(all='ignore'):
                    ax.plot(beta_highres,fit_screw(beta_highres,*popt_screw[X]),':',color='gray')
            else:
                raise ValueError("keyword 'filename'={} undefined.".format(filename))
        ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig(f"B_iso_{filename}+fits.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge")
    mkfitplot(metal,"screw")

    ### finally, also plot B as a function of stress using the fits computed above
    def B_of_sigma(X,character,mkplot=True,B0fit='weighted',resolution=500):
        '''Computes arrays sigma and B_of_sigma of length 'resolution', and returns a tuple (B0,sigma,B_of_sigma) where B0 is either the minimum value, or B(v=0) if B0fit=='zero'
           or a weighted average of the two (B0fit='weighted',default).
           A plot of the results is saved to disk if mkplot=True (default).'''
        popt_e = popt_edge[X]
        popt_s = popt_screw[X]
        vcrit = Y[X].ct
        if character=='screw':
            if modes == 'TT':
                print("Warning: B for screw dislocations from purely transverse phonons does not diverge. Analytic expression for B(sigma) will not be a good approximation!")
            fname = "Biso_of_sigma_screw_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "screw"
        elif character=='edge':
            fname = "Biso_of_sigma_edge_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "edge"
        else: ## 'aver' = default
            fname = "Biso_of_sigma_{}.pdf".format(X)
            ftitle = "{}, ".format(X) + "averaged over $\\vartheta$"
        burg = Y[X].burgers
        
        @np.vectorize
        def B(v):
            bt = abs(v/vcrit)
            if bt<1:
                if character == 'edge':
                    out = 1e-3*fit_edge(bt, *popt_e)
                elif character == 'screw':
                    out = 1e-3*fit_screw(bt, *popt_s)
                else:
                    out = 1e-3*(fit_edge(bt, *popt_e)+fit_screw(bt, *popt_s))/2
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
            
        ## determine stress that will lead to velocity of 95% transverse sound speed and stop plotting there, or at 1.5GPa (whichever is smaller)
        sigma_max = sigma_eff(0.95*vcrit)
        # print("{}, {}: sigma(0.95ct) = {:.1f} MPa".format(X,character,sigma_max/1e6))
        if sigma_max<6e8 and B(0.95*vcrit)<1e-4: ## if B, sigma still small, increase to 99% vcrit
            sigma_max = sigma_eff(0.99*vcrit)
            # print("{}, {}: sigma(0.99ct) = {:.1f} MPa".format(X,character,sigma_max/1e6))
        if sigma_max<6e8 and B(0.99*vcrit)<1e-4: ## if B, sigma still small, increase to 99.9% vcrit
            sigma_max = sigma_eff(0.999*vcrit)
            # print("{}, {}: sigma(0.999ct) = {:.1f} MPa".format(X,character,sigma_max/1e6))
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
        B_of_sig = B(vr(sigma))
        if mkplot:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(3.,2.5))
            ax.set_xlabel(r'$\sigma$[MPa]',fontsize=fntsize)
            ax.set_ylabel(r'$B$[mPas]',fontsize=fntsize)
            ax.set_title(ftitle,fontsize=fntsize)
            ax.axis((0,sigma[-1]/1e6,0,B_of_sig[-1]*1e3))
            ax.plot(sigma/1e6,Bsimple(sigma,B0)*1e3,':',color='gray',label=r"$\sqrt{B_0^2\!+\!\left(\frac{\sigma b}{c_\mathrm{t}}\right)^2}$, $B_0=$"+f"{1e6*B0:.1f}"+r"$\mu$Pas")
            ax.plot(sigma/1e6,Bstraight(sigma,Boffset)*1e3,':',color='green',label=r"$B_0+\frac{\sigma b}{c_\mathrm{t}}$, $B_0=$"+f"{1e6*Boffset:.1f}"+r"$\mu$Pas")
            ax.plot(sigma/1e6,B_of_sig*1e3,label=r"$B_\mathrm{fit}(v(\sigma))$")
            plt.xticks(fontsize=fntsize)
            plt.yticks(fontsize=fntsize)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.legend(loc='upper left',handlelength=1.1, frameon=False, shadow=False,fontsize=8)
            plt.savefig(fname,format='pdf',bbox_inches='tight')
            plt.close()
        return (B0,sigma,B_of_sig)
        
    B_of_sig = {}
    sigma = {}
    B0 = {}
    for character in ['aver', 'screw', 'edge']:
        for X in metal:
            Xc = X+character
            B0[Xc], sigma[Xc], B_of_sig[Xc] = B_of_sigma(X,character,mkplot=True,B0fit='weighted')
        fig, ax = plt.subplots(1, 1, sharey=False, figsize=(3.5,3.5))
        ax.set_xlabel(r'$\sigma b/(c_\mathrm{t}B_0)$',fontsize=fntsize)
        ax.set_ylabel(r'$B/B_0$',fontsize=fntsize)
        if character=='screw':
            ax.set_title("screw",fontsize=fntsize)
            fname = "Biso_of_sigma_all_screw.pdf"
        elif character=='edge':
            ax.set_title("edge",fontsize=fntsize)
            fname = "Biso_of_sigma_all_edge.pdf"
        else:
            ax.set_title("averaged over $\\vartheta$",fontsize=fntsize)
            fname = "Biso_of_sigma_all.pdf"
        sig_norm = np.linspace(0,3.6,500)
        ax.axis((0,sig_norm[-1],0.5,4))
        for X in metal:
            Xc = X+character
            sig0 = Y[X].ct*B0[Xc]/Y[X].burgers
            ax.plot(sigma[Xc]/sig0,B_of_sig[Xc]/B0[Xc],label=fr"{X}, $B_0={1e6*B0[Xc]:.1f}\mu$Pas")
        ax.plot(sig_norm,np.sqrt(1+sig_norm**2),':',color='black',label=r"$\sqrt{1+\left(\frac{\sigma b}{c_\mathrm{t}B_0}\right)^2}$")
        # ax.plot(sig_norm,0.25 + sig_norm,':',color='green',label="$0.25+\\frac{\sigma b}{c_\mathrm{t}B_0}$")
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.legend(loc='best',handlelength=1.1, frameon=False, shadow=False,fontsize=fntsize-1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.savefig(fname,format='pdf',bbox_inches='tight')
        plt.close()
    
