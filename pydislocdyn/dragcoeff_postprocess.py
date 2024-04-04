#!/usr/bin/env python3
# Compute the drag coefficient of a moving dislocation from phonon wind in a semi-isotropic approximation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Apr. 3, 2024
'''This submodule can be used to derive drag coefficient B(stress) from B(velocity) by using a fitting funcion 
   for the latter (which is a required input, see submodule phononwind)'''
#################################
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
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import pandas as pd
##
from pydislocdyn.dislocations import Dislocation

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
