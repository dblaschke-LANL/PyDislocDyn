# Compute the drag coefficient of a moving dislocation from phonon wind in an isotropic crystal
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Jan. 23, 2021
#################################
import sys
import numpy as np
from scipy.optimize import curve_fit, fmin, fsolve
##################
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
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
## workaround for spyder's runfile() command when cwd is somewhere else:
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
import metal_data as data
from elasticconstants import elasticC2, elasticC3
from dislocations import fourieruij_iso, ompthreads
from phononwind import elasticA3, dragcoeff_iso
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    ## Ncores=0 bypasses phononwind calculations entirely and only generates plots using data from a previous run
    Ncores = max(1,int(Ncpus/max(2,ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    # Ncores = -2
    if ompthreads == 0: # check if subroutines were compiled with OpenMP support
        print("using joblib parallelization with ",Ncores," cores")
    else:
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
NT = 1 # number of temperatures between roomT and maxT (WARNING: implementation of temperature dependence is incomplete!)
constantrho = False ## set to True to override thermal expansion coefficient and use alpha_a = 0 for T > roomT
beta_reference = 'base'  ## define beta=v/ct, choosing ct at roomT ('base') or current T ('current') as we increase temperature
roomT = 300 # in Kelvin
maxT = 600
## phonons to include ('TT'=pure transverse, 'LL'=pure longitudinal, 'TL'=L scattering into T, 'LT'=T scattering into L, 'mix'=TL+LT, 'all'=sum of all four):
modes = 'all'
# modes = 'TT'
skip_plots=False ## set to True to skip generating plots from the results
# in Fourier space:
Nphi = 50
Nphi1 = 50
Nq1 = 400
Nt = 321 # base value, grid is adaptive in Nt
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
# import polycrystal_averaging as pca
# c12 = {}
# c44 = {}
# cMl = {}
# cMm = {}
# cMn = {}
# ### generate a list of those fcc and bcc metals for which we have sufficient data (i.e. TOEC)
# metal = sorted(list(data.c111.keys()))
# metal_cubic = data.fcc_metals.union(data.bcc_metals).intersection(metal)
# ### compute average elastic constants for these metals:
# print("computing averaged elastic constants ...")
# C2 = {}
# aver = pca.IsoAverages(pca.lam,pca.mu,pca.Murl,pca.Murm,pca.Murn)
# for X in metal:
#     C2[X] = elasticC2(c11=pca.c11[X], c12=pca.c12[X], c44=pca.c44[X], c13=pca.c13[X], c33=pca.c33[X], c66=pca.c66[X])   
#     S2 = pca.elasticS2(C2[X])
#     C3 = elasticC3(c111=pca.c111[X], c112=pca.c112[X], c113=pca.c113[X], c123=pca.c123[X], c133=pca.c133[X], c144=pca.c144[X], c155=pca.c155[X], c166=pca.c166[X], c222=pca.c222[X], c333=pca.c333[X], c344=pca.c344[X], c366=pca.c366[X], c456=pca.c456[X])
#     S3 = pca.elasticS3(S2,C3)
#     aver.voigt_average(C2[X],C3)
#     aver.reuss_average(S2,S3)
#     HillAverage = aver.hill_average()
#     ### use Hill average for Lame constants for non-cubic metals, as we do not have a better scheme at the moment
#     c12[X] = float(HillAverage[pca.lam])
#     c44[X] = float(HillAverage[pca.mu])
#     ### use Hill average for Murnaghan constants, as we do not have a better scheme at the moment
#     cMl[X] = float(HillAverage[pca.Murl])
#     cMm[X] = float(HillAverage[pca.Murm])
#     cMn[X] = float(HillAverage[pca.Murn])
#     
# # replace Hill with improved averages for effective Lame constants of cubic metals:
# for X in metal_cubic:
#     ### don't waste time computing the "improved average" for the Murnaghan constants when we are going to use the Hill average
#     ImprovedAv = aver.improved_average(C2[X],None)
#     c12[X] = float(ImprovedAv[pca.lam])
#     c44[X] = float(ImprovedAv[pca.mu])
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
            Tfile.write('\n'.join(map("{:.2f}".format,highT)))
            
        print("Computing the drag coefficient from phonon wind ({} modes) for: ".format(modes),metal)
    
    A3 = {}
    for X in metal:
        A3[X] = elasticA3(elasticC2(c12=c12[X], c44=c44[X]), elasticC3(l=cMl[X], m=cMm[X], n=cMn[X]))/c44[X]
    for X in metal:
        # wrap all main computations into a single function definition to be run in a parallelized loop below
        def maincomputations(bt,X,modes=modes):
            Bmix = np.zeros((len(theta),len(highT)))
                                    
            dij = fourieruij_iso(bt, ct_over_cl[X], theta, phi)
            Bmix[:,0] = dragcoeff_iso(dij=dij, A3=A3[X], qBZ=qBZ[X], ct=ct[X], cl=cl[X], beta=bt, burgers=burgers[X], T=roomT, modes=modes, Nt=Nt, Nq1=Nq1, Nphi1=Nphi1)
            
            for Ti in range(len(highT)-1):
                T = highT[Ti+1]
                expansionratio = (1 + alpha_a[X]*(T - roomT)) ## TODO: replace with values from eos!
                if constantrho == True:
                    expansionratio = 1 ## turn off expansion
                qBZT = qBZ[X]/expansionratio
                burgersT = burgers[X]*expansionratio
                rhoT = rho[X]/expansionratio**3
                c44T = c44[X] ## TODO: need to implement T dependence of shear modulus!
                c12T = bulk[X] - 2*c44T/3 ## TODO: need to implement T dependence of bulk modulus!
                ctT = np.sqrt(c44T/rhoT)
                ct_over_cl_T = np.sqrt(c44T/(c12T+2*c44T))
                clT = ctT/ct_over_cl_T
                ## beta, as it appears in the equations, is v/ctT, therefore:
                if beta_reference == 'current':
                    betaT = bt
                else:
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
            Bmix = np.array([maincomputations(bt,X,modes) for bt in beta])
        elif Ncores == 0:
            pass
        else:
            Bmix = np.array(Parallel(n_jobs=Ncores)(delayed(maincomputations)(bt,X,modes) for bt in beta))

        # and write the results to disk (in various formats)
        if Ncores != 0:
            with open("drag_{}.dat".format(X),"w") as Bfile:
                Bfile.write("### B(beta,theta) for {} in units of mPas, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
                Bfile.write('beta/theta[pi]\t' + '\t'.join(map("{:.4f}".format,theta/np.pi)) + '\n')
                for bi in range(len(beta)):
                    Bfile.write("{:.4f}".format(beta[bi]) + '\t' + '\t'.join(map("{:.6f}".format,Bmix[bi,:,0])) + '\n')
            
        # only print temperature dependence if temperatures other than room temperature are actually computed above
        if len(highT)>1 and Ncores !=0:
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
            ax.plot(Broom[X][:,0],Broom[X][:,th+1],lnstyles[X], color=metalcolors[X], label=X)
        legend = ax.legend(loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.2, shadow=True)
        plt.savefig("B_iso_{:.4f}.pdf".format(theta[th]),format='pdf',bbox_inches='tight')
        plt.close()
        
    ## define fitting fcts.:
    if modes=='TT': ## degree of divergence is reduced for purely transverse modes
        print("fitting for transverse modes only (assuming reduced degrees of divergence)")
        def fit_edge(x, c0, c1, c2, c3, c4):
            return c0 - c1*x + c2*x**2 + c3*np.log(1-x**2) + c4*(1/np.sqrt(1-x**2) - 1)
            
        def fit_screw(x, c0, c1, c2, c3, c4):
            return c0 - c1*x + c2*x**2 + c3*x**4 + c4*x**16
    else:
        def fit_edge(x, c0, c1, c2, c3, c4):
            return c0 - c1*x + c2*np.log(1-x**2) + c3*(1/(1-x**2)**(1/2) - 1) + c4*(1/(1-x**2)**(3/2) - 1)
            
        def fit_screw(x, c0, c1, c2, c3, c4):
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
        fitfile.write(" & "+" & ".join((metal))+" \\\\\hline\hline")
        fitfile.write("\n $c_{\mathrm{t}}$")
        for X in metal:
            fitfile.write(" & "+"{:.0f}".format(ct[X]))

    def mkfitplot(metal_list,filename):
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
                ax.plot(beta,Broom[X][:,Ntheta],color=metalcolors[X],label=X)
                ax.plot(beta_highres,fit_edge(beta_highres,*popt_edge[X]),':',color='gray')
            elif filename=="screw":
                ax.plot(beta,Broom[X][:,1],color=metalcolors[X],label=X)
                ax.plot(beta_highres,fit_screw(beta_highres,*popt_screw[X]),':',color='gray')
            else:
                raise ValueError("keyword 'filename'={} undefined.".format(filename))
        ax.legend(loc='upper left', ncol=3, columnspacing=0.8, handlelength=1.2, frameon=True, shadow=False, numpoints=1,fontsize=fntsize)
        plt.savefig("B_iso_{0}K_{1}+fits.pdf".format(roomT,filename),format='pdf',bbox_inches='tight')
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
        vcrit = ct[X]
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
        burg = burgers[X]
        
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
            ax.plot(sigma/1e6,Bsimple(sigma,B0)*1e3,':',color='gray',label="$\sqrt{B_0^2\!+\!\\left(\\frac{\sigma b}{c_\mathrm{t}}\\right)^2}$, $B_0=$"+"{:.1f}".format(1e6*B0)+"$\mu$Pas")
            ax.plot(sigma/1e6,Bstraight(sigma,Boffset)*1e3,':',color='green',label="$B_0+\\frac{\sigma b}{c_\mathrm{t}}$, $B_0=$"+"{:.1f}".format(1e6*Boffset)+"$\mu$Pas")
            ax.plot(sigma/1e6,B_of_sig*1e3,label="$B_\mathrm{fit}(v(\sigma))$")
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
            sig0 = ct[X]*B0[Xc]/burgers[X]
            ax.plot(sigma[Xc]/sig0,B_of_sig[Xc]/B0[Xc],label="{}, $B_0={:.1f}\mu$Pas".format(X,1e6*B0[Xc]))
        ax.plot(sig_norm,np.sqrt(1+sig_norm**2),':',color='black',label="$\sqrt{1+\\left(\\frac{\sigma b}{c_\mathrm{t}B_0}\\right)^2}$")
        # ax.plot(sig_norm,0.25 + sig_norm,':',color='green',label="$0.25+\\frac{\sigma b}{c_\mathrm{t}B_0}$")
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.legend(loc='best',handlelength=1.1, frameon=False, shadow=False,fontsize=fntsize-1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.savefig(fname,format='pdf',bbox_inches='tight')
        plt.close()
    
