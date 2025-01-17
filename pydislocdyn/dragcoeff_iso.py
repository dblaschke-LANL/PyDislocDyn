#!/usr/bin/env python3
# Compute the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Jan. 17, 2025
'''This script will calculate the drag coefficient from phonon wind in the isotropic limit and generate nice plots;
   it is not meant to be used as a module.
   The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
   metals that are predefined in metal_data.py, falling back to all available if no argument is passed.'''
#################################
import sys
import os
import copy
import numpy as np
from scipy.optimize import curve_fit
dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)
##
import pydislocdyn.metal_data as data
from pydislocdyn.utilities import ompthreads, printthreadinfo, parse_options, showoptions, Ncores, read_2dresults, \
    plt, fntsettings, AutoMinorLocator, pd ## matplotlib stuff
from pydislocdyn.dislocations import readinputfile
from pydislocdyn.phononwind import phonondrag, B_of_sigma, OPTIONS

### choose various resolutions and other parameters:
Ntheta = 2 # number of angles between burgers vector and dislocation line (minimum 2, i.e. pure edge and pure screw)
Nbeta = 99 # number of velocities to consider ranging from minb to maxb (as fractions of transverse sound speed)
minb = 0.01
maxb = 0.99
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
# modes = 'TT'
skip_plots=False ## set to True to skip generating plots from the results
use_exp_Lame= True # if using data from metal_data, choose between experimentally determined Lame and Murnaghan constants (default) or analytical averages of SOEC and TOEC (use_exp_Lame = False)
NT = 1 # number of temperatures between baseT and maxT (WARNING: implementation of temperature dependence is incomplete!)
constantrho = False ## set to True to override thermal expansion coefficient and use alpha_a = 0 for T > baseT
increaseTby = 300 # so that maxT=baseT+increaseTby (default baseT=300 Kelvin, but may be overwritten by an input file below)
beta_reference = 'base'  ## define beta=v/ct, choosing ct at baseT ('base') or current T ('current') as we increase temperature
#####
# in Fourier space:
Nphi = 50 # keep this (and other Nphi below) an even number for higher accuracy (because we integrate over pi-periodic expressions in some places and phi ranges from 0 to 2pi)
## the following options can be set on the commandline with syntax --keyword=value:
phononwind_opts = {} ## pass additional options to dragcoeff_iso() of phononwind.py

#########
if __name__ == '__main__':
    Y={}
    use_metaldata=True
    if len(sys.argv) > 1:
        args, kwargs = parse_options(sys.argv[1:],OPTIONS,globals())
        phononwind_opts.update(kwargs)
    phononwind_opts['modes']=modes
    printthreadinfo(Ncores,ompthreads)
    ### set range & step sizes after parsing the command line for options
    beta = np.linspace(minb,maxb,Nbeta)
    phi = np.linspace(0,2*np.pi,Nphi)
    if use_exp_Lame:
        metal = sorted(list(data.ISO_l.keys()))
    else:
        metal = sorted(list(data.c111.keys()))
    metal_kws = metal.copy()
    if len(sys.argv) > 1 and len(args)>0:
        try:
            inputdata = [readinputfile(i, Ntheta=Ntheta, isotropify=True) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError as fnameerror:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = args[0].split()
            for X in metal:
                if X not in metal_kws:
                    raise ValueError(f"One or more input files not found and {X} is not a valid keyword") from fnameerror
    
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        for X in metal:
            data.writeinputfile(X,X,iso=use_exp_Lame) # write temporary input files for requested X of metal_data
            Y[X] = readinputfile(X,Ntheta=Ntheta,isotropify=True)
        os.chdir("..")
    
    if Ncores == 0:
        print("skipping phonon wind calculations as requested")
    else:
        with open("dragcoeff_iso_options.log","w", encoding="utf8") as logfile:
            optiondict = showoptions(OPTIONS,globals())
            for key, item in optiondict.items():
                if key not in ['Ncores', 'skip_plots']:
                    logfile.write(f"{key} = {item}\n")
        for X in metal:
            with open(X+"_iso.log", "w", encoding="utf8") as logfile:
                logfile.write(repr(Y[X]))
                logfile.write("\n\nbeta =v/ct:\n")
                logfile.write('\n'.join(map("{:.5f}".format,beta)))
                logfile.write("\n\ntheta:\n")
                logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
        
        print(f"Computing the drag coefficient from phonon wind ({modes} modes in the isotropic limit) for: {metal}")
    
    highT = {}
    for X in metal:
        highT[X] = np.linspace(Y[X].T,Y[X].T+increaseTby,NT)
        if constantrho:
            Y[X].alpha_a = 0
        ## only write temperature to files if we're computing temperatures other than baseT=Y[X].T
        if len(highT[X])>1 and Ncores !=0:
            with open(X+"_iso.log","a", encoding="utf8") as logfile:
                logfile.write("\n\nT:\n")
                logfile.write('\n'.join(map("{:.2f}".format,highT[X])))
    for X in metal:
        if Ncores != 0:
            Bmix = np.zeros((len(beta),len(Y[X].theta),len(highT[X])))
            Bmix[:,:,0] = phonondrag(Y[X], beta, Ncores=Ncores, pandas_out=False, **phononwind_opts)
            for Ti in range(len(highT[X])-1):
                Z = copy.copy(Y[X]) ## local copy we can modify for higher T
                Z.T = highT[X][Ti+1]
                expansionratio = 1 + Y[X].alpha_a*(Z.T - Y[X].T) ## TODO: replace with values from eos!
                Z.qBZ = Y[X].qBZ/expansionratio
                Z.burgers = Y[X].burgers*expansionratio
                Z.rho = Y[X].rho/expansionratio**3
                Z.c44 = Y[X].mu ## TODO: need to implement T dependence of shear modulus!
                Z.c12 = Y[X].bulk - 2*Z.c44/3 ## TODO: need to implement T dependence of bulk modulus!
                Z.ct = np.sqrt(Z.c44/Z.rho)
                Z.ct_over_cl = np.sqrt(Z.c44/(Z.c12+2*Z.c44))
                Z.cl = Z.ct/Z.ct_over_cl
                Z.init_C2()
                ## beta, as it appears in the equations, is v/ctT, therefore:
                if beta_reference == 'current':
                    betaT = beta
                else:
                    betaT = beta*Y[X].ct/Z.ct
                ### TODO: need models for T dependence of TOECs!
                Z.c123 = Y[X].C3[0,1,2]
                Z.c144 = Y[X].C3[0,3,3]
                Z.c456 = Y[X].C3[3,4,5]
                Z.init_C3()
                Z.vcrit_all = None ## needs to be recomputed
                ##
                Bmix[:,:,Ti+1] = phonondrag(Z, betaT, Ncores=Ncores, pandas_out=False, **phononwind_opts)
            
            # and write the results to disk (in various formats)
            with open(f"drag_{X}.dat","w", encoding="utf8") as Bfile:
                Bfile.write(f"### B(beta,theta) for {X} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.\n")
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,Y[X].theta/np.pi)) + '\n')
                for bi, bt in enumerate(beta):
                    Bfile.write(f"{bt:.4f}\t" + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT[X])>1 and Ncores !=0:
            with open(f"drag_T_{X}.dat","w", encoding="utf8") as Bfile:
                if len(Y[X].theta)>2:
                    Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\t' + '\t'.join(map("{:.5f}".format,Y[X].theta[1:-1])) + '\tBedge[mPas]' + '\n')
                else:
                    Bfile.write('temperature[K]\tbeta\tBscrew[mPas]\tBedge[mPas]' + '\n')
                for bi, bt in enumerate(beta):
                    for Ti, hTi in enumerate(highT[X]):
                        Bfile.write(f"{hTi:.1f}\t{bt:.4f}\t" + '\t'.join(map("{:.6f}".format,Bmix[bi,:,Ti])) + '\n')

    #############################################################################################################################

    if skip_plots:
        print("skipping plots as requested")
        sys.exit()
    
    ###### plot room temperature results:
    ## load data from isotropic calculation
    Broom = {}
    for X in metal:
        ## for every X, Broom has shape (len(theta+1),Nbeta), first column is beta all others all B for various dislocation types theta in the range 0 to  pi/2
        Broom[X] = read_2dresults(f"drag_{X}.dat")
        beta = Broom[X].index.to_numpy()
        theta = Broom[X].columns.to_numpy()
            
    ## define line styles for every metal in the same plot
    lnstyles = {'Al':'-', 'Cu':'--', 'Fe':':', 'Nb':'-.', 'Cd':'-', 'Mg':'--', 'Zn':':', 'Sn':'-.', 'Ni':'-.', 'Mo':'--', 'Ag':':', 'Au':'-.', 'Ti':'-', 'Zr':'-.'}
    metalcolors = {'Al':'blue', 'Cu':'orange', 'Fe':'green', 'Nb':'red', 'Zn':'purple', 'Sn':'black', 'Ag':'lightblue', 'Au':'goldenrod', 'Cd':'lightgreen', 'Mg':'lightsalmon', 'Mo':'magenta', 'Ni':'silver', 'Ti':'olive', 'Zr':'cyan'}
        
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
    def fit_aver(x,cs0,cs1,cs2,cs3,cs4,ce0,ce1,ce2,ce3,ce4):
        '''returns the average of fit_screw and fit_edge'''
        return (fit_screw(x,cs0,cs1,cs2,cs3,cs4) + fit_edge(x,ce0,ce1,ce2,ce3,ce4))/2
    
    popt_edge = {}
    pcov_edge = {}
    popt_screw = {}
    pcov_screw = {}
    Bmax_fit = 0.20 ## only fit up to Bmax_fit [mPas]
    for X in metal:
        popt_edge[X], pcov_edge[X] = curve_fit(fit_edge, beta, Broom[X].iloc[:,-1], bounds=([0.9*Broom[X].iloc[0,-1],0.,-0.,-0., -0.], [1.1*Broom[X].iloc[0,-1], 2*Broom[X].iloc[0,-1], 1., 1.,1.]))
        popt_screw[X], pcov_screw[X] = curve_fit(fit_screw, beta, Broom[X].iloc[:,0], bounds=([0.9*Broom[X].iloc[0,0],0.,-0.,-0.,-0.], [1.1*Broom[X].iloc[0,0], 2*Broom[X].iloc[0,0], 1., 1., 1.]))
    
    with open("drag_iso_fit.txt","w", encoding="utf8") as fitfile:
        if modes=='TT': ## degree of divergence is reduced for purely transverse modes
            fitfile.write("Fitting functions for B[$\\mu$Pas] at room temperature (transverse modes only):\nEdge dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*log(1-x**2) + {4:.2f}*(1/(1-x**2)**(1/2) - 1)\n".format(*1e3*popt_edge[X]))
            fitfile.write("\nScrew dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*x**4 + {4:.2f}*x**16\n".format(*1e3*popt_screw[X]))
        else:
            fitfile.write("Fitting functions for B[$\\mu$Pas] at room temperature:\nEdge dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*log(1-x**2) + {3:.2f}*(1/(1-x**2)**(1/2) - 1) + {4:.2f}*(1/(1-x**2)**(3/2) - 1)\n".format(*1e3*popt_edge[X]))
            fitfile.write("\nScrew dislocations:\n")
            for X in metal:
                fitfile.write("f"+X+"(x) = {0:.2f} - {1:.2f}*x + {2:.2f}*x**2 + {3:.2f}*log(1-x**2) + {4:.2f}*(1/(1-x**2)**(1/2) - 1)\n".format(*1e3*popt_screw[X]))
        fitfile.write("\nwhere $x=v/c_{\\mathrm{t}$\n\n")
        fitfile.write(" & "+" & ".join((metal))+r" \\\hline\hline")
        fitfile.write("\n $c_{\\mathrm{t}}$")
        for X in metal:
            fitfile.write(f" & {Y[X].ct:.0f}")

    def mkfitplot(metal_list,filename):
        '''Plot the dislocation drag over velocity and show the fitting function.'''
        fig, ax = plt.subplots(1, 1, figsize=(4.5,4.))
        plt.xticks(**fntsettings)
        plt.yticks(**fntsettings)
        ax.set_xticks(np.arange(11)/10)
        ax.set_yticks(np.arange(12)/100)
        ax.axis((0,maxb,0,0.11)) ## define plot range
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlabel(r'$\beta_\mathrm{t}$',**fntsettings)
        ax.set_ylabel(r'$B$[mPa$\,$s]',**fntsettings)
        ax.set_title(filename,**fntsettings)
        for X in metal_list:
            beta_highres = np.linspace(0,1,1000)
            if filename=="edge":
                if X in metalcolors:
                    ax.plot(beta,Broom[X].iloc[:,-1],color=metalcolors[X],label=X)
                else:
                    ax.plot(beta,Broom[X].iloc[:,-1],label=X) # fall back to automatic colors
                with np.errstate(all='ignore'):
                    ax.plot(beta_highres,fit_edge(beta_highres,*popt_edge[X]),':',color='gray')
            elif filename=="screw":
                if X in metalcolors:
                    ax.plot(beta,Broom[X].iloc[:,0],color=metalcolors[X],label=X)
                else:
                    ax.plot(beta,Broom[X].iloc[:,0],label=X) # fall back to automatic colors
                with np.errstate(all='ignore'):
                    ax.plot(beta_highres,fit_screw(beta_highres,*popt_screw[X]),':',color='gray')
            else:
                raise ValueError(f"keyword {filename=} undefined.")
        ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsettings['fontsize'])
        plt.savefig(f"B_iso_{filename}+fits.pdf",format='pdf',bbox_inches='tight')
        plt.close()
        
    mkfitplot(metal,"edge")
    mkfitplot(metal,"screw")
    
    maxerror = {}
    for X in metal:
        maxerror[X] = {'edge':max(abs(1-fit_edge(beta,*popt_edge[X])/Broom[X].iloc[:,-1])), 'screw':max(abs(1-fit_screw(beta,*popt_screw[X])/Broom[X].iloc[:,0]))}
        Y[X].vcrit_smallest=Y[X].vcrit_screw=Y[X].vcrit_edge=Y[X].ct
        Y[X].name += f"_{Y[X].sym}"
        
    B_of_sig = {}
    sigma = {}
    B0 = {}
    vcrit = {} ## dummy variable, always ct
    for character in ['aver', 'screw', 'edge']:
        for X in metal:
            if character=='screw':
                popt = popt_screw[X]
                fit=fit_screw
            elif character=='edge':
                popt = popt_edge[X]
                fit=fit_edge
            elif character=='aver':
                popt = tuple(popt_screw[X])+tuple(popt_edge[X])
                fit=fit_aver
            Xc = X+character
            if character=='screw' and modes == 'TT':
                print("Warning: B for screw dislocations from purely transverse phonons does not diverge. Analytic expression for B(sigma) will not be a good approximation!")
            B0[Xc], vcrit[Xc], sigma[Xc], B_of_sig[Xc] = B_of_sigma(Y[X],popt,character,mkplot=False,B0fit='weighted',fit=fit)
        if len(metal)<5:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.,4.))
            legendops={'loc':'best','ncol':1}
        else:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(5.,5.))
            legendops={'loc':'upper left','bbox_to_anchor':(1.01,1),'ncol':1}
        ax.set_xlabel(r'$\sigma b/(c_\mathrm{t}B_0)$',**fntsettings)
        ax.set_ylabel(r'$B/B_0$',**fntsettings)
        if character=='screw':
            ax.set_title("screw",**fntsettings)
            fname = "Biso_of_sigma_all_screw.pdf"
        elif character=='edge':
            ax.set_title("edge",**fntsettings)
            fname = "Biso_of_sigma_all_edge.pdf"
        else:
            ax.set_title("averaged over $\\vartheta$",**fntsettings)
            fname = "Biso_of_sigma_all.pdf"
        sig_norm = np.linspace(0,3.6,500)
        ax.axis((0,sig_norm[-1],0.5,4))
        if not os.path.exists("BofSig_iso"):
            os.mkdir("BofSig_iso")
        for X in metal:
            Xc = X+character
            sig0 = Y[X].ct*B0[Xc]/Y[X].burgers
            ax.plot(sigma[Xc]/sig0,B_of_sig[Xc]/B0[Xc],label=fr"{X}, $B_0={1e6*B0[Xc]:.1f}\mu$Pas")
            Bsig = pd.Series(1e3*B_of_sig[Xc],index=1e-6*sigma[Xc],name=f"{Xc}, B [mPa s]")
            Bsig.index.name="resolved shear stress [MPa]"
            Bsig.to_csv(os.path.join("BofSig_iso",f"B_of_sigma_{Xc}.csv.xz"),compression='xz',header=True)
        ax.plot(sig_norm,np.sqrt(1+sig_norm**2),':',color='black',label=r"$\sqrt{1+\left(\frac{\sigma b}{c_\mathrm{t}B_0}\right)^2}$")
        # ax.plot(sig_norm,0.25 + sig_norm,':',color='green',label="$0.25+\\frac{\sigma b}{c_\mathrm{t}B_0}$")
        plt.xticks(**fntsettings)
        plt.yticks(**fntsettings)
        ax.legend(**legendops, columnspacing=0.8, handlelength=1.1, frameon=False, shadow=False,fontsize=fntsettings['fontsize']-1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.savefig(fname,format='pdf',bbox_inches='tight')
        plt.close()
    
