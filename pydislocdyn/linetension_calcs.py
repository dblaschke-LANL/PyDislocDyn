#!/usr/bin/env python3
# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Apr. 2, 2024
'''This module defines the Dislocation class which inherits from metal_props of polycrystal_averaging.py
   and StrohGeometry of dislocations.py. As such, it is the most complete class to compute properties
   dislocations, both steady state and accelerating. Additionally, the Dislocation class can calculate
   additional properties like limiting velocities of dislocations. We also define a function, readinputfile,
   which reads a PyDislocDyn input file and returns an instance of the Dislocation class.
   If run as a script, this file will compute the dislocation line tension and generate various plots.
   The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
   metals that are predefined in metal_data.py, falling back to all available if no argument is passed.'''
#################################
import sys
import os
import time
import shutil, lzma
import numpy as np
import sympy as sp
from scipy import optimize, integrate
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
## workaround for spyder's runfile() command when cwd is somewhere else:
dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)
##
from pydislocdyn import metal_data as data
from pydislocdyn.elasticconstants import roundcoeff, Voigt, UnVoigt, CheckReflectionSymmetry
from pydislocdyn.polycrystal_averaging import metal_props, loadinputfile
from pydislocdyn.dislocations import StrohGeometry, ompthreads, printthreadinfo, Ncpus, elbrak1d, accedge_theroot, rotaround
try:
    from joblib import Parallel, delayed
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/max(2,ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used) ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = Ncpus = 1 ## must be 1 without joblib

## choose which shear modulus to use for rescaling to dimensionless quantities
## allowed values are: 'crude', 'aver', and 'exp'
## choose 'crude' for mu = (c11-c12+2c44)/4, 'aver' for mu=Hill average  (resp. improved average for cubic), and 'exp' for experimental mu supplemented by 'aver' where data are missing
## (note: when using input files, 'aver' and 'exp' are equivalent in that mu provided in that file will be used and an average is only computed if mu is missing)
scale_by_mu = 'exp'
skip_plots=False ## set to True to skip generating line tension plots from the results
### choose resolution of discretized parameters: theta is the angle between disloc. line and Burgers vector, beta is the dislocation velocity,
### and phi is an integration angle used in the integral method for computing dislocations
Ntheta = 600
Ntheta2 = 21 ## number of character angles for which to calculate critical velocities (set to None or 0 to bypass entirely)
Nbeta = 500 ## set to 0 to bypass line tension calculations
Nphi = 1000
## choose among predefined slip systems when using metal_data.py (see that file for details)
bccslip = 'all' ## allowed values: '110', '112', '123', 'all' (for all three)
hcpslip = 'all' ## allowed values: 'basal', 'prismatic', 'pyramidal', 'all' (for all three)
##### the following options can be set on the commandline with syntax --keyword=value:
def str2bool(arg):
    '''converts a string to bool'''
    if arg in ['True', 'true', '1', 't', 'yes', 'y']:
        out=True
    elif arg in ['False', 'false', '0', 'f', 'no', 'n']:
        out=False
    else:
        raise ValueError(f"cannot convert {arg} to bool")
    return out
def guesstype(arg):
    '''takes a string and tries to convert to int, float, bool, falling back to a string'''
    try:
        out = int(arg)
    except ValueError:
        try:
            out = float(arg)
        except ValueError:
            try:
                out = bool(arg)
            except ValueError:
                out = arg ## fall back to string
    return out
OPTIONS = {"Ntheta":int, "Ntheta2":int, "Nbeta":int, "Nphi":int, "scale_by_mu":str, "skip_plots":str2bool, "bccslip":str, "hcpslip":str, "Ncores":int}
metal = sorted(list(data.all_metals | {'ISO'})) ### input data; also test isotropic limit

class Dislocation(StrohGeometry,metal_props):
    '''This class has all properties and methods of classes StrohGeometry and metal_props, as well as some additional methods: computevcrit, findvcrit_smallest, findRayleigh.
       If optional keyword Miller is set to True, b and n0 are interpreted as Miller indices (and Cartesian otherwise); note since n0 defines a plane its Miller indices are in reziprocal space.'''
    def __init__(self,b, n0, theta=[0,np.pi/2], Nphi=500,sym='iso', name='some_crystal',Miller=False,lat_a=None,lat_b=None,lat_c=None,lat_alpha=None,lat_beta=None,lat_gamma=None):
        metal_props.__init__(self, sym, name)
        if lat_a is not None: self.ac=lat_a
        if lat_b is not None: self.bc=lat_b
        if lat_c is not None: self.cc=lat_c
        if lat_alpha is not None: self.alphac=lat_alpha
        if lat_beta is not None: self.betac=lat_beta
        if lat_gamma is not None: self.gammac=lat_gamma
        if Miller:
            self.Millerb = b
            b = self.Miller_to_Cart(self.Millerb)
            self.Millern0 = n0
            n0 = self.Miller_to_Cart(self.Millern0,reziprocal=True)
        StrohGeometry.__init__(self, b, n0, theta, Nphi)
        self.vcrit_smallest=self.vcrit_screw=self.vcrit_edge=None
        self.C2_aligned_screw = self.C2_aligned_edge = None
        self.sym = sym
        self.vcrit_barnett = None
        self.vcrit_all = None
        self.Rayleigh = None
        self.vRF = None
    
    def alignC2(self):
        '''Calls self.computerot() and then computes the rotated SOEC tensor C2_aligned in coordinates aligned with the slip plane for each character angle.'''
        self.computerot()
        if self.C2.dtype == np.dtype('O'):
            self.C2_aligned = np.zeros((self.Ntheta,6,6),dtype=object)
        else:
            self.C2_aligned = np.zeros((self.Ntheta,6,6)) ## compute C2 rotated into dislocation coordinates
        for th in range(self.Ntheta):
            if self.sym=='iso':
                self.C2_aligned[th] = self.C2 ## avoids rounding errors in the isotropic case where C2 shouldn't change
            else:
                self.C2_aligned[th] = Voigt(np.dot(self.rot[th],np.dot(self.rot[th],np.dot(self.rot[th],np.dot(UnVoigt(self.C2),self.rot[th].T)))))
    
    def computevcrit_barnett(self, theta_list=None, setvcrit=True, verbose=False):
        '''Computes the limiting velocities following Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5.
           All parameters are optional: unless a list of character angles is passed explicitly via theta_list,
           we calculate limiting velocities for all character angles in self.theta.
           Option setvcrit determines whether or not to overwrite attribute self.vcrit_barnett.
           Note that this method does not check for subtle cancellations that may occur in the dislocation displacement gradient at those velocities;
           use the frontend method .computevcrit(theta) for fully automated determination of the lowest critical velocity at each character angle.'''
        norm=(self.C2[3,3]/self.rho)
        C2 = UnVoigt(self.C2/self.C2[3,3])
        if theta_list is None:
            Ntheta = self.Ntheta
            m0 = self.m0
            theta = self.theta
        else:
            Ntheta = len(theta_list)
            theta = np.asarray(theta_list)
            t = np.outer(np.cos(theta),self.b) + np.outer(np.sin(theta),np.cross(self.b,self.n0))
            m0 = np.cross(self.n0,t)
        out = np.zeros((2,Ntheta,3))
        for th in range(Ntheta):
            def findvlim(phi,i):
                M = m0[th]*np.cos(phi) + self.n0*np.sin(phi)
                MM = np.dot(M,np.dot(C2,M))
                P = -np.trace(MM)
                Q = 0.5*(P**2-np.trace((MM @ MM)))
                # R = -np.linalg.det(MM)
                R = -( MM[0,0]*MM[1,1]*MM[2,2] + MM[0,2]*MM[1,0]*MM[2,1] + MM[0,1]*MM[1,2]*MM[2,0] \
                      - MM[0,2]*MM[1,1]*MM[2,0] - MM[0,0]*MM[1,2]*MM[2,1] - MM[0,1]*MM[1,0]*MM[2,2] )
                a = Q - P**2/3
                d = (2*P**3-9*Q*P+27*R)/27
                gamma = np.arccos(-0.5*d/np.sqrt(-a**3/27))
                tmpout = -P/3 + 2*np.sqrt(-a/3)*np.cos((gamma+2*i*np.pi)/3)
                return np.abs(np.sqrt(tmpout*norm)/np.cos(phi))
            for i in range(3):
                ## default minimizer sometimes yields nan, but bounded method doesn't always find the smallest value, so run both:
                with np.errstate(invalid='ignore'): ## don't need to know about arccos producing nan while optimizing
                    minresult1 = optimize.minimize_scalar(findvlim,bounds=(0,2.04*np.pi),args=i) # slightly enlarge interval for better results despite rounding errors in some cases
                    minresult2 = optimize.minimize_scalar(findvlim,method='bounded',bounds=(0,2.04*np.pi),args=i)
                if verbose and not (minresult1.success and minresult2.success):
                    print(f"Warning ({self.name}, theta={theta[th]}):\n{minresult1}\n{minresult2}\n\n")
                ## always take the smaller result, ignore nan:
                choose = np.nanargmin(np.array([minresult1.fun,minresult2.fun]))
                if choose == 0: minresult = minresult1
                else: minresult = minresult2
                out[0,th,i] = minresult.fun
                out[1,th,i] = minresult.x
        if setvcrit: self.vcrit_barnett = out
        return out[0]
        
    def computevcrit_screw(self):
        '''Compute the limiting velocity of a pure screw dislocation analytically, provided the slip plane is a reflection plane, use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        self.C2_aligned_screw = self.C2_aligned[self.findedgescrewindices()[0]]
        A = self.C2_aligned_screw[4,4]
        B = 2*self.C2_aligned_screw[3,4]
        C = self.C2_aligned_screw[3,3]
        if CheckReflectionSymmetry(self.C2_aligned_screw):
            self.vcrit_screw = np.sqrt((A-B**2/(4*C))/self.rho)
        return self.vcrit_screw
    
    def computevcrit_edge(self):
        '''Compute the limiting velocity of a pure edge dislocation analytically, provided the slip plane is a reflection plane (cf. L. J. Teutonico 1961, Phys. Rev. 124:1039), use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        self.C2_aligned_edge = self.C2_aligned[self.findedgescrewindices()[1]]
        if CheckReflectionSymmetry(self.C2_aligned_edge):
            c11=self.C2_aligned_edge[0,0]
            c22=self.C2_aligned_edge[1,1]
            c66=self.C2_aligned_edge[5,5]
            c12=self.C2_aligned_edge[0,1]
            test = np.abs(self.C2_aligned_edge/self.C2[3,3]) ## check for additional symmetry requirements
            if test[0,5] + test[1,5] < 1e-12:
                self.vcrit_edge = np.sqrt(min(c66,c11)/self.rho)
                ## cover case of q<0 (cf. Teutonico 1961 paper, eq (39); line above was only for q>0):
                if ((c11*c22-c12**2-2*c12*c66) - (c22+c66)*min(c66,c11))/(c22*c66)<0:
                    ## analytic solution to Re(lambda=0) in eq. (39) (with sp.solve); sqrt below is real because of if statement above:
                    minval = (2*np.sqrt(c22*c66*(-c11*c22 + c11*c66 + c12**2 + 2*c12*c66 + c22*c66))*(c12 + c66) - (-c11*c22**2 + c11*c22*c66 + c12**2*c22 + c12**2*c66 + 2*c12*c22*c66 + 2*c12*c66**2 + 2*c22*c66**2))/((c22 - c66)**2)
                    self.vcrit_edge = min(self.vcrit_edge,np.sqrt(minval/self.rho))
            else:
                c16 = self.C2_aligned_edge[0,5]
                c26 = self.C2_aligned_edge[1,5]
                def theroot(y,rv2):
                    return accedge_theroot(y,rv2,c11,c12,c16,c22,c26,c66)
                y,rv2 = sp.symbols('y,rv2')
                ysol = sp.solve(theroot(y,rv2),y) ## 4 complex roots as fcts of rv2=rho*v**2
                yfct=sp.lambdify(rv2,ysol,modules=[np.emath,'scipy'])
                def f(x):
                    return np.abs(np.asarray(yfct(x)).imag.prod()) ## lambda=i*y, and any Re(lambda)=0 implies a divergence/limiting velocity
                with np.errstate(invalid='ignore'):
                    rv2limit = optimize.fsolve(f,1e5)
                    if f(rv2limit) < 1e-11: ## check if fsolve was successful
                        self.vcrit_edge = np.sqrt(rv2limit[0]/self.rho)
                    else:
                        print(f'Warning: {self.name}.computevcrit_edge() (resp. fsolve) failed, debug info: {rv2limit=}, {np.sqrt(rv2limit[0]/self.rho)=}, {f(rv2limit)=}')
        return self.vcrit_edge

    def computevcrit(self,theta=None,set_screwedge=True,setvcrit=True):
        '''Compute the lowest critical (or limiting) velocities for all dislocation character angles within list 'theta'. If theta is omitted, we fall back to attribute .theta (default).
        The list of results will be stored in method .vcrit_all, i.e. .vcrit_all[0]=theta and .vcrit_all[1] contains the corresponding lowest limiting velocities.
        Additionally, .vcrit_all[3] contains the highest critical velocities and .vcrit_all[2] contains the intermediate critical velocities.
        Option set_screwedge=True guarantees that attributes .vcrit_screw and .vcrit_edge will be set, and 'setvrit=True' will overwrite self.vcrit_barnett.'''
        if theta is None:
            theta=self.theta
        indices = self.findedgescrewindices(theta)
        self.vcrit_all = np.empty((4,len(theta)))
        self.vcrit_all[0] = theta
        if self.sym=='iso':
            self.computevcrit_screw()
            self.vcrit_all[1:] = self.vcrit_screw
            if self.cl==0:
                self.init_all()
            self.vcrit_all[3] = self.cl
        else:
            self.vcrit_all[1:] = np.sort(self.computevcrit_barnett(theta_list=np.asarray(theta),setvcrit=setvcrit),axis=1).T
        if indices[0] is not None:
            self.computevcrit_screw()
            if CheckReflectionSymmetry(self.C2_aligned_screw):
                self.vcrit_all[1:,indices[0]] = self.vcrit_screw
            elif set_screwedge:
                self.vcrit_screw = self.vcrit_all[1,indices[0]]
        if indices[1] is not None:
            self.computevcrit_edge()
            if CheckReflectionSymmetry(self.C2_aligned_edge) and self.vcrit_edge is not None:
                self.vcrit_all[2,indices[1]] = self.vcrit_all[1,indices[1]] = self.vcrit_edge
                if len(indices) == 3:
                    self.vcrit_all[2,indices[2]] = self.vcrit_all[1,indices[2]] = self.vcrit_edge
            elif set_screwedge:
                self.vcrit_edge = self.vcrit_all[1,indices[1]]
                if len(indices) == 3:
                    self.vcrit_edge = min(self.vcrit_edge,self.vcrit_all[1,indices[2]])
        return self.vcrit_all[1]
    
    def findvcrit_smallest(self,xatol=1e-2):
        '''Computes the smallest critical velocity, which subsequently is stored as attribute .vcrit_smallest and the full result of scipy.minimize_scalar is returned
           (as type 'OptimizeResult' with its 'fun' being vcrit_smallest and 'x' the associated character angle theta).
           The absolute tolerance for theta can be passed via xatol; in order to improve accuracy and speed of this routine, we make use of computevcrit with Ntheta>=11 resolution
           in order to be able to pass tighter bounds to the subsequent call to minimize_scalar(). If .vcrit_all already exists in sufficient resolution from an earlier call,
           this step is skipped.'''
        backupvcrit = self.vcrit_all
        if self.vcrit_all is None or self.vcrit_all.shape[1]<11:
            self.computevcrit(theta=np.linspace(self.theta[0],self.theta[-1],11),set_screwedge=False,setvcrit=False)
        vcrit_smallest = np.nanmin(self.vcrit_all[1])
        thind = np.where(self.vcrit_all[1]==vcrit_smallest)[0][0] ## find index of theta so that we may pass tighter bounds to minimize_scalar below for more accurate (and faster) results
        bounds=(max(-np.pi/2,self.vcrit_all[0][max(0,thind-1)]),min(np.pi/2,self.vcrit_all[0][min(thind+1,len(self.vcrit_all[0])-1)]))
        def f(x):
            return np.min(self.computevcrit_barnett(theta_list=[x],setvcrit=False))
        if self.sym=='iso': result = vcrit_smallest
        else: result = optimize.minimize_scalar(f,method='bounded',bounds=bounds,options={'xatol':xatol})
        if self.sym=='iso': self.vcrit_smallest = vcrit_smallest
        elif result.success: self.vcrit_smallest = min(result.fun,vcrit_smallest)
        if backupvcrit is not None: self.vcrit_all = backupvcrit ## don't change vcrit_all, restore from our backup
        return result
    
    def findRayleigh(self):
        '''Computes the Rayleigh wave speed for every dislocation character self.theta.'''
        Rayleigh=np.zeros((self.Ntheta))
        norm = self.C2[3,3] # use c44
        C2norm = UnVoigt(self.C2/norm)
        if self.vcrit_all is None or len(self.vcrit_all[0])!=self.Ntheta or np.any(self.vcrit_all[0]!=self.theta):
            self.computevcrit(set_screwedge=False) ## need it as an upper bound on the Rayleigh speed
        vcrit = self.vcrit_all[1]
        for th in range(self.Ntheta):
            def Rayleighcond(B):
                return abs((B[0,0]+B[1,1])/2-np.sqrt((B[0,0]-B[1,1])**2/4 + B[0,1]**2))
            def findrayleigh(x):
                tmpC = C2norm - self.Cv[:,:,:,:,th]*x**2
                M=self.M[:,th].T
                N=self.N[:,th].T
                MM = elbrak1d(M,M,tmpC)
                MN = elbrak1d(M,N,tmpC)
                NM = elbrak1d(N,M,tmpC)
                NN = elbrak1d(N,N,tmpC)
                NNinv = np.linalg.inv(NN)
                S = - NNinv @ NM
                B = MM + MN @ S
                return Rayleighcond(integrate.trapezoid(B,x=self.phi,axis=0)/(4*np.pi**2))
            bounds=(0.0,vcrit[th]*np.sqrt(self.rho/norm))
            result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=bounds,options={'xatol':1e-12})
            # if result.fun>=1e-3: print(f"{bounds}\n{result}")  ## if this failed, try enlarging the search interval slightly above vcrit (there was some numerical uncertainty there too):
            if result.success and result.fun>=1e-3: result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=(0.5*bounds[1],1.25*bounds[1]),options={'xatol':1e-12})
            if result.fun>=1e-3 or not result.success: print(f"Failed: Rayleigh not found in [{bounds[0]},{1.25*bounds[1]}]\n",result)
            if result.success and result.fun<1e-3: Rayleigh[th] = result.x * np.sqrt(norm/self.rho)
        self.Rayleigh = Rayleigh
        return Rayleigh
    
    def find_vRF(self,fast=True,verbose=False,resolution=50,thetaind=None,partial_burgers=None):
        '''Compute the radiation-free velocity for dislocations (default: edge dislocations) in the transonic regime.
           For details on the method, see Gao, Huang, Gumbsch, Rosakis, JMPS 47 (1999) 1941.
           If the slip system has orthotropic symmetry, the analytic solution derived in that paper is used.
           Otherwise, we use a numerical method.
           In the first transonic regime, there may be a range of radiation free velocities; option resolution
           determines how many values to probe for in this regime. In this regime, we also support searching for
           radiation-free velocities for mixed dislocations: option 'thetaind' may be any index pointing to an
           element of self.theta. Furthermore, partial_burgers may be passed using Miller indices to check for vRF 
           of a partial dislocation (an instance of the Dislocation class always represents a perfect dislocation
           with normalized Burgers vector self.b).
           Option 'fast=False' is only used for orthotropic slip systems in which case it will bypass the
           analytic solution in favor of the numerical one in order to facilitate testing the latter;
           this option may be removed in future versions.'''
        if self.C2_aligned is None:
            self.alignC2()
        if self.vcrit_all is None or len(self.vcrit_all[0])!=self.Ntheta or np.any(self.vcrit_all[0]!=self.theta):
            self.computevcrit()
        if thetaind is None:
            edgind = self.findedgescrewindices()[1]
        else:
            edgind=thetaind
            fast=False ## always use numeric code for mixed or partial disloc.
        if partial_burgers is not None: fast=False
        c = self.C2_aligned[edgind]
        test = np.abs(c/self.C2[3,3]) ## check for symmetry requirements
        if fast and CheckReflectionSymmetry(c,strict=True) and test[0,5]+test[1,5]+test[2,5]+test[3,4] < 1e-12:
            self.vRF = out = np.sqrt((c[0,0]*c[1,1]-c[0,1]**2)/(c[0,1]+c[1,1])/self.rho)
            if verbose: print("orthotropic symmetry detected, using analytic solution")
        else:
            rv2 = sp.symbols('rv2') # abbreviation for rho*v^2;
            delta = np.identity(3,dtype=int)
            p = sp.symbols('p')
            m = np.array([1,0,0]) ## we already rotated C2 so that now x1 is in the direction of v and x2 is the slip plane normal
            n = np.array([0,1,0])
            l = m + sp.I*p*n
            norm = self.C2[3,3] ## use c44 to normalize some numbers below
            C2eC = UnVoigt(c/norm)
            if CheckReflectionSymmetry(c):
                delta = delta[:2,:2] ## only need 2x2 submatrix for edge in this case
                n = n[:2]
                l = l[:2]
                C2eC = C2eC[:2,:2,:2,:2]
            C2M = sp.simplify(sp.Matrix(np.dot(l,np.dot(C2eC,l)) - rv2*delta))
            thedet = sp.simplify(sp.det(C2M))
            fct = sp.lambdify((p,rv2),thedet,modules=[np.emath])
            vlim = self.vcrit_all[1:,edgind]
            burg = None
            if thetaind is not None:
                burg = self.rot[edgind] @ self.b
            if partial_burgers is not None:
                burg = self.rot[edgind] @ self.Miller_to_Cart(partial_burgers)
            if burg is not None:
                x = np.array([1,0,0])
                v = np.cross(x,burg)
                sv = np.sqrt(np.vdot(v,v))
                rot = rotaround(v/sv,sv,x@burg)
            bounds = (self.rho*(vlim[1])**2/norm,self.rho*(vlim[2])**2/norm)
            def L2_of_beta2(beta2,comp=1):
                '''Finds eigenvector L in the 2nd transonic regime; this function needs as input beta2 = (rho/c44)*v^2'''
                def f(x):
                    return float(np.abs(fct(x,beta2)))
                p1 = None
                for x0 in [0.5,1,1.5]:
                    psol = optimize.root(f,x0)
                    if psol.success:
                        p1 = abs(float(psol.x))
                        break
                if p1 is not None:
                    C2Mp = C2M.subs({rv2:beta2,p:p1})
                    Ak = C2Mp.eigenvects()
                    p0 = abs(Ak[0][0])
                    Ap = Ak[0][2][0]
                    if abs(Ak[1][0])<p0:
                        p0 = abs(Ak[1][0])
                        Ap = Ak[1][2][0]
                    if len(Ak)>2 and abs(Ak[2][0])<p0:
                        p0 = abs(Ak[2][0])
                        Ap = Ak[2][2][0]
                    if p0 > 1e-5:
                        Ap = np.repeat(np.inf,len(Ak))
                    L = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap)))).subs(p,p1).evalf()
                else:
                    L = [1e12,1e12] ## random high number so that minimize_scalar() below will not pick this velocity
                if comp=='all':
                    out = L.evalf()
                else:
                    out = float(np.abs(L[comp]))
                return out
            def L1_of_beta2(beta2,comp=1,burg = burg):
                '''Finds L1+L2 in the 1st transonic regime; this function needs as input beta2 = (rho/c44)*v^2'''
                @np.vectorize
                def f(x):
                    return float(np.abs(fct(x,beta2)))
                p1 = None
                p2 = None
                psol = optimize.root(f,np.array([0.9,1.5]))
                if psol.success:
                    if len(set(np.abs(np.round(psol.x,12))))==2:
                        p1 = abs(float(psol.x[0]))
                        p2 = abs(float(psol.x[1]))
                if p1 is not None and p2 is not None:
                    C2Mp1 = C2M.subs({rv2:beta2,p:p1})
                    C2Mp2 = C2M.subs({rv2:beta2,p:p2})
                    Ak1 = C2Mp1.eigenvects()
                    Ak2 = C2Mp2.eigenvects()
                    p01 = abs(Ak1[0][0])
                    Ap1 = Ak1[0][2][0]
                    p02 = abs(Ak2[0][0])
                    Ap2 = Ak2[0][2][0]
                    if abs(Ak1[1][0])<p01:
                        p01 = abs(Ak1[1][0])
                        Ap1 = Ak1[1][2][0]
                    if abs(Ak1[2][0])<p01:
                        p01 = abs(Ak1[2][0])
                        Ap1 = Ak1[2][2][0]
                    if p01 > 1e-5:
                        Ap1 = np.array([np.inf,np.inf,np.inf])
                    if abs(Ak2[1][0])<p02:
                        p02 = abs(Ak2[1][0])
                        Ap2 = Ak2[1][2][0]
                    if abs(Ak2[2][0])<p02:
                        p02 = abs(Ak2[2][0])
                        Ap2 = Ak2[2][2][0]
                    if p02 > 1e-5:
                        Ap2 = np.array([np.inf,np.inf,np.inf])
                    L1 = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap1)))).subs(p,p1).evalf()
                    L2 = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap2)))).subs(p,p2).evalf()
                    factor = complex(-L1[1]/L2[1])
                else:
                    L1 = L2 = sp.Matrix([1e12,1e12,1e12]) ## random high number so that minimize_scalar() below will not pick this velocity
                    factor = 1
                if comp=='all':
                    out = (L1+factor*L2).evalf()
                else:
                    L1plusL2 = np.array((L1+factor*L2).evalf(),dtype=complex)
                    if burg is not None:
                        L1plusL2 = rot @ L1plusL2
                    out = float(np.abs((L1plusL2)[1]) + abs((L1plusL2).imag[0]*(L1plusL2).real[2] - (L1plusL2).imag[2]*(L1plusL2).real[0]))
                return out
            def checkprops(vRF):
                '''checks properties (such as boundary conditions) for the solution-candidate vRF'''
                out = None
                tol = 1e-5
                if vRF is None: return out
                if vRF.success and vRF.fun < tol:
                    out_norm = np.sqrt(vRF.x*norm/self.rho)
                    checkL = np.array(roundcoeff(L2_of_beta2(vRF.x,comp='all'),int(-np.log10(tol))),dtype=complex)
                    if CheckReflectionSymmetry(c):
                        checkL[-1].imag = checkL[0].imag
                        checkL[-1].real=0 ## condition below reduces to Re(L[0])=0 or Im(L[0])=0 in this case
                    if not (abs(checkL[0].real*checkL[-1].imag - checkL[-1].real*checkL[0].imag)<tol) or sp.Matrix(checkL).norm()<tol:
                        if verbose: print(f"Error: eigenvector does not meet required properties: \n{checkL=}")
                    else:
                        # double check number of real eigenvalues (method above only works for one pair out of three)
                        checkdet = roundcoeff(thedet.subs(rv2,vRF.x))
                        eigenvals = np.array(roundcoeff(sp.Matrix(sp.solve(checkdet.subs({p**6:rv2**3,p**4:rv2**2,p**2:rv2}),rv2))),dtype=complex)
                        if (int(sum(eigenvals.real>0)) != 1) and (float(sum(eigenvals.imag**2)) > 1e-15):
                            print(f"Error: unexpected number of real eigenvalues detected: \n{eigenvals=}")
                            if verbose: print(f"{out_norm}")
                        else:
                            if verbose: print("success")
                            out = out_norm
                elif verbose: print(f"condition for disloc. glide not met (L[1]!=0);\n{vRF=} ")
                return out
            out = self.vRF = None
            if burg is None: #bypass for mixed or partial disloc. (not supported yet) TODO: check if we really never have a solution here for mixed dislocs.
                if verbose: print("searching in the 2nd transonic regime ...")
                vRF = optimize.minimize_scalar(L2_of_beta2,method='bounded',bounds=bounds)
                if (out:=checkprops(vRF)) is not None:
                    self.vRF = out
            bounds_fst = (self.rho*self.vcrit_edge**2/norm,self.rho*(vlim[1])**2/norm)
            if np.isclose(bounds_fst[0],bounds_fst[1]):
                if verbose and thetaind is None: print("found only one transonic regime for this gliding edge dislocation")
            else:
                if verbose: print("searching in the 1st transonic regime ...")
                vels = np.linspace(bounds_fst[0],bounds_fst[1],resolution)
                vRF_fst = []
                for v in vels:
                    if L1_of_beta2(v)<1e-9:
                        if (L1_of_beta2(v,'all').norm())>1e-9: # make sure we found a non-trivial eigenvector
                            vRF_fst.append(np.sqrt(v*norm/self.rho))
                if len(vRF_fst)==0:
                    vRF_fst = optimize.minimize_scalar(L1_of_beta2,method='bounded',bounds=bounds_fst)
                    if (vRF_fst.success and vRF_fst.fun < 1e-9):
                        vRF_fst = np.sqrt(vRF_fst.x*norm/self.rho)
                    else:
                        vRF_fst = None
                if vRF_fst is not None:
                    if verbose: print("success")
                    if self.vRF is None:
                        self.vRF = vRF_fst
                    else:
                        self.vRF = [vRF_fst,self.vRF]
                elif verbose: print("nothing found in 1st transonic regime")
            if self.vRF is None:
                print(f"Failed: could not find a solution for vRF of {self.name}")
        return self.vRF
    
    def plotdisloc(self,beta=None,character='screw',component=[2,0],a=None,eta_kw=None,etapr_kw=None,t=None,shift=None,fastapprox=False,Nr=250,nogradient=False,skipcalc=False,showplt=False,savefig=True,**kwargs):
        '''Calculates and generates a plot of the requested component of the dislocation displacement gradient; plotting is done by the function plotuij().
           Optional arguments are: the normalized velocity 'beta'=v/self.ct (defaults to self.beta, assuming one of the .computeuij() methods were called earlier).
           'character' is either 'edge', 'screw' (default), or an index of self.theta, and 'component' is
           a list of two indices indicating which component of displacement gradient u[ij] to plot.
           The steady-state solution is plotted unless an acceleration 'a' (or a more general function eta_kw) is passed. In the latter case,
           'slipsystem' is required except for those metals where its keyword coincides with self.sym (see documentation of self.computeuij_acc_screw()
           and self.computeuij_acc_edge() for details on capabilities and limitations of the current implementation of the accelerating solution).
           Option nogradient=True will plot the displacement field instead of its gradient; this option must be combined with an integer value for 'component'
           and is currently only implemented for steady-state solutions (a=None).
           Option skipcalc=True (implied when beta is not set) may be passed to plot results of an earlier calculation with the same input parameters (useful
           for plotting multiple components of the dislocation field).
           If option 'showplt' is set to 'True', the figure is shown in an interactive session in addition to being saved to a file. Warning: this will only work
           if the user sets matplotlib's backend to an interactive one after PyDislocDyn was loaded (e.g. by calling %matplotlib inline). Saving the figure to
           a file can be suppressed with option 'savefig=False'.
           See the documentation of plotting function plotuij() for additional options that may be passed to it via kwargs.'''
        if beta is None:
            beta = self.beta
            skipcalc = True
        ## make sure everything we need has been initialized:
        if self.ct==0:
            self.ct = np.sqrt(self.mu/self.rho)
        if np.count_nonzero(self.C2norm) == 0:
            self.C2norm = UnVoigt(self.C2/self.mu)
        if self.C2_aligned is None:
            self.alignC2()
        if skipcalc and self.r is not None:
            r = self.r
            Nr = len(r)
        else:
            r = np.linspace(0.0001,1,Nr)
        xylabel = {0:'x',1:'y',2:'z'}
        if a is None and eta_kw is None:
            if not skipcalc:
                if not nogradient:
                    self.computeuij(beta=beta)
                    self.alignuij() ## self.rot was computed as a byproduct of .alignC2() above
                else:
                    self.computeuk(beta=beta, r=r)
                    self.alignuk()
            if character == 'screw':
                index = self.findedgescrewindices()[0]
            elif character == 'edge':
                index = self.findedgescrewindices()[1]
            else:
                index=character
            if not nogradient:
                namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}{character}_{self.name}_v{beta*self.ct:.0f}"
                uijtoplot = self.uij_aligned[component[0],component[1],index]
                uijtoplot = np.outer(1/r,uijtoplot)
            else:
                namestring = f"u{xylabel[component]}{character}_{self.name}_v{beta*self.ct:.0f}"
                uijtoplot = self.uk_aligned[component,index]
        elif character=='screw' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_screw(a,beta,burgers=self.burgers,fastapprox=fastapprox,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None: acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}screw_{self.name}_v{beta*self.ct:.0f}_a{acc:}"
            uijtoplot = self.uij_acc_screw_aligned[component[0],component[1]]
        elif character=='edge' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_edge(a,beta,burgers=self.burgers,fastapprox=fastapprox,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None: acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}edge_{self.name}_v{beta*self.ct:.0f}_a{acc:}"
            uijtoplot = self.uij_acc_edge_aligned[component[0],component[1]]
        else:
            raise ValueError("not implemented")
        if savefig: savefig=namestring+".pdf"
        plotuij(uijtoplot,r,self.phi,**kwargs,showplt=showplt,title=namestring,savefig=savefig)
        
    def __repr__(self):
        return  "DISLOCATION\n" + metal_props.__repr__(self) + f"\n burgers:\t {self.burgers}\n" + StrohGeometry.__repr__(self)

def readinputfile(fname,init=True,theta=None,Nphi=500,Ntheta=2,symmetric=True,isotropify=False):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the Dislocation class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.
       Array theta contains all dislocation characters to be considered, and integer Nphi denotes the resolution to be used for polar angle phi.
       Alternatively, instead of passing theta explicitly, the number of characters angles between 0 and pi/2 can be passed via keyword Ntheta.
       In this case, keyword 'symmetric' will determine whether the generated theta array ranges from 0 to pi/2 (True) or from -pi/2 to pi/2 (False).
       The latter keyword can also be read from file 'fname'.
       If option isotropify is set to True, we calculate isotropic averages of the elastic constants and return an instance of the Dislocation class
       with sym=iso and using those averages.'''
    inputparams = loadinputfile(fname)
    sym = inputparams['sym']
    name = inputparams.get('name',str(fname))
    if 'Millerb' in inputparams or 'Millern0' in inputparams:
        temp = metal_props(sym,name) ## need a metal_props method to convert to Cartesian b, n0
        temp.populate_from_dict(inputparams)
        b = temp.b
        n0 = temp.n0
    else:
        b = np.asarray(inputparams['b'].split(','),dtype=float)
        n0 = np.asarray(inputparams['n0'].split(','),dtype=float)
    if theta is None:
        symmetric = inputparams.get('symmetric',symmetric)
        if symmetric is True or symmetric == 'True' or Ntheta<=2: theta = np.linspace(0,np.pi/2,Ntheta)
        else: theta = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
    out = Dislocation(sym=sym, name=name, b=b, n0=n0, theta=theta, Nphi=Nphi)
    out.populate_from_dict(inputparams)
    out.filename = fname ## remember which file we read
    if isotropify and sym != 'iso': # bypass if we're already isotropic
        inputparams['sym'] = 'iso'
        inputparams['name'] = name+'_ISO'
        if 'lam' in inputparams: inputparams.pop('lam') ## ignore if read from file, use averages instead
        if 'mu' in inputparams: inputparams.pop('mu')
        out.lam = out.mu = None
        out.init_all()
        inputparams['c12'] = out.lam
        inputparams['c44'] = out.mu
        inputparams['a'] = np.cbrt(out.Vc)
        if 'c123' in inputparams:
            print("Warning: there is no good averaging scheme for TOECs, calculating (unreliable) Hill averages for the Murnaghan constants.")
            out.compute_Lame(include_TOEC=True)
            inputparams['c123'] = 2*out.Murl-2*out.Murm+out.Murn
            inputparams['c144'] = out.Murm-out.Murn/2
            inputparams['c456'] = out.Murn/4
        out = Dislocation(sym='iso', name=name+'_ISO', b=b, n0=n0, theta=theta, Nphi=Nphi)
        out.populate_from_dict(inputparams)
    if init:
        out.init_all()
        out.C2norm = UnVoigt(out.C2/out.mu)
    return out

def plotuij(uij,r,phi,lim=(-1,1),showplt=True,title=None,savefig=False,fntsize=11,axis=(-0.5,0.5,-0.5,0.5),figsize=(3.5,4.0),cmap=plt.cm.rainbow,showcontour=False,**kwargs):
    '''Generates a heat map plot of a 2-dim. dislocation field, where the x and y axes are in units of Burgers vectors and
    the color-encoded values are dimensionless displacement gradients.
    Required parameters are the 2-dim. array for the displacement gradient field, uij, as well as arrays r and phi for 
    radius (in units of Burgers vector) and polar angle; note that the plot will be converted to Cartesian coordinates.
    Options include, the colorbar limits "lim", whether or not to call plt.show(), an optional title for the plot,
    which filename (if any) to save it as, the fontsize to be used, the plot range to be passed to pyplot.axis(), the size of
    the figure, which colormap to use, and whether or not show contours (showcontour may also include a list of levels).
    Additional options may be passed on to pyplot.contour via **kwargs (ignored if showcontour=False).'''
    phi_msh , r_msh = np.meshgrid(phi,r)
    x_msh = r_msh*np.cos(phi_msh)
    y_msh = r_msh*np.sin(phi_msh)
    plt.figure(figsize=figsize)
    plt.axis(axis)
    plt.xticks(np.linspace(*axis[:2],5),fontsize=fntsize)
    plt.yticks(np.linspace(*axis[2:],5),fontsize=fntsize)
    plt.xlabel(r'$x[b]$',fontsize=fntsize)
    plt.ylabel(r'$y[b]$',fontsize=fntsize)
    if title is not None: plt.title(title,fontsize=fntsize,loc='left')
    if np.all(uij==0): raise ValueError('Dislocation field contains only zeros, forgot to calculate?')
    if uij.shape != (len(r),len(phi)):
        uij = np.outer(1/r,uij)
    colmsh = plt.pcolormesh(x_msh, y_msh, uij, vmin=lim[0], vmax=lim[-1], cmap=cmap, shading='gouraud')
    colmsh.set_rasterized(True)
    cbar = plt.colorbar()
    if not isinstance(showcontour,bool):
        kwargs['levels'] = showcontour
        showcontour = True
    if showcontour:
        if 'levels' not in kwargs: kwargs['levels'] = np.linspace(-1,1,6)
        if 'colors' not in kwargs: kwargs['colors'] = 'white'
        if 'linewidths' not in kwargs: kwargs['linewidths'] = 0.7
        plt.contour(x_msh,y_msh,uij,**kwargs)
    cbar.ax.tick_params(labelsize = fntsize)
    if showplt: plt.show()
    if savefig is not False: plt.savefig(savefig,format='pdf',bbox_inches='tight',dpi=150)
    plt.close()

def read_2dresults(fname):
    '''Read results (such as line tension or drag coefficient) from file fname and return a Pandas DataFrame where index=beta (or [temperature,beta]) and columns=theta.'''
    if os.access((newfn:=fname+'.xz'), os.R_OK): fname = newfn # new default
    elif os.access((newfn:=fname), os.R_OK): pass # old default
    elif os.access((newfn:=fname+'.gz'), os.R_OK): fname = newfn
    else: raise FileNotFoundError(f'tried {fname}.xz, {fname}, and {fname}.gz')
    out = pd.read_csv(fname,skiprows=1,index_col=0,sep='\t')
    try:
        out.columns = pd.to_numeric(out.columns)*np.pi
    except ValueError:
        out = pd.read_csv(fname,skiprows=1,index_col=[0,1],sep='\t')
        out.columns = pd.to_numeric(out.columns)*np.pi
    if len(out.index.names)==1:
        out.index.name='beta'
    out.columns.name='theta'
    return out

def parse_options(arglist,optionlist=OPTIONS,globaldict=globals()):
    '''Search commandline arguments passed to this script for known options to set by comparing to a list of keyword strings "optionlist".
    These will then override default variables set above in this script. This function also returns a copy of 'arglist' stripped of all 
    option calls for further processing (e.g. opening input files that were passed etc.).'''
    out = arglist
    if '--help' in out:
        print(f"\nUsage: {sys.argv[0]} <options> <inputfile(s)>\n")
        print("available options (see code manual for details):")
        for key in optionlist:
            print(f'--{key}={optionlist[key]}')
        sys.exit()
    setoptions = [i for i in out if "--" in i and i[:2]=="--"]
    kwargs = {}
    for i in setoptions:
        out.remove(i)
        if "=" not in i: continue ## ignore options without assigned values
        key,val = i[2:].split("=")
        if key in optionlist:
            globaldict[key] = optionlist[key](val)
            print(f"setting {key}={globaldict[key]}")
        else:
            kwargs[key] = guesstype(val)
    time.sleep(1) ## avoid race conditions after changing global variables
    return (out,kwargs)
    
### start the calculations
if __name__ == '__main__':
    Y={}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args, kwargs = parse_options(sys.argv[1:])
    printthreadinfo(Ncores,ompthreads)
    ### set range & step sizes after parsing the commandline for options
    dtheta = np.pi/(Ntheta-2)
    theta = np.linspace(-np.pi/2-dtheta,np.pi/2+dtheta,Ntheta+1)
    beta = np.linspace(0,1,Nbeta)
    metal_kws = metal.copy()
    if len(sys.argv) > 1 and len(args)>0:
        try:
            inputdata = [readinputfile(i,init=False,theta=theta,Nphi=Nphi) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = metal_list = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError as fnameerror:
            ## only compute the metals the user has asked us to
            metal = args[0].split()
            for X in metal:
                if X not in metal_kws:
                    raise ValueError(f"One or more input files not found and {X} is not a valid keyword") from fnameerror
        
    bcc_metals = data.bcc_metals.copy()
    hcp_metals = data.hcp_metals.copy()
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        if scale_by_mu=='exp':
            isokw=False ## use single crystal elastic constants and additionally write average shear modulus of polycrystal to temp. input file
        else:
            isokw='omit' ## writeinputfile(..., iso='omit') will bypass writing ISO_c44 values to the temp. input files and missing Lame constants will always be auto-generated by averaging
        for X in metal:
            if X in bcc_metals:
                if bccslip == 'all':
                    slipkw = ['110', '112', '123']
                else:
                    slipkw=[bccslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokw,bccslip=kw)
                    metal_list.append(X+kw)
                    bcc_metals.add(X+kw)
            elif X in hcp_metals:
                if hcpslip == 'all':
                    slipkw = ['basal','prismatic','pyramidal']
                else:
                    slipkw=[hcpslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokw,hcpslip=kw)
                    metal_list.append(X+kw)
                    hcp_metals.add(X+kw)
            elif X=='ISO':
                metal_list.append(X)
            else:
                data.writeinputfile(X,X,iso=isokw) # write temporary input files for requested X of metal_data
                metal_list.append(X)
        for X in metal_list:
            if X=='ISO': ## define some isotropic elastic constants to check isotropic limit:
                Y[X] = Dislocation(sym='iso', name='ISO', b=[1,0,0], n0=[0,1,0], theta=theta, Nphi=Nphi, lat_a=1e-10)
                Y[X].burgers = 1e-10
                Y[X].c44 = 1e9
                Y[X].poisson = 1/3
                Y[X].rho = 1e3
            else:
                Y[X] = readinputfile(X,init=False,theta=theta,Nphi=Nphi)
        os.chdir("..")
        metal = metal_list
        ## list of metals symmetric in +/-theta (for the predefined slip systems):
        metal_symm = sorted(list({'ISO'}.union(data.fcc_metals).union(hcp_metals).union(data.tetr_metals).intersection(metal)))
    else:
        metal_symm = set([]) ## fall back to computing for character angles of both signs if we don't know for sure that the present slip system is symmetric

    for X in metal:
        Y[X].init_C2()
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        if scale_by_mu == 'crude':
            Y[X].mu = (Y[X].C2[0,0]-Y[X].C2[0,1]+2*Y[X].C2[3,3])/4
        #### will generate missing mu/lam by averaging over single crystal constants (improved Hershey/Kroener scheme for cubic, Hill otherwise)
        ### for hexagonal/tetragonal metals, this corresponds to the average shear modulus in the basal plane (which is the most convenient for the present calculations)
        if Y[X].mu is None:
            Y[X].compute_Lame()

    if Nbeta > 0:
        with open("theta.dat","w", encoding="utf8") as thetafile:
            thetafile.write('\n'.join("{:.6f}".format(thi) for thi in theta[1:-1]))
                           
    C2 = {}
    scaling = {}
    beta_scaled = {}
    ## compute smallest critical velocity in ratio to the scaling velocity computed from the average shear modulus mu (see above):
    ## i.e. this corresponds to the smallest velocity leading to a divergence in the dislocation field at some character angle theta
    for X in metal:
        Y[X].findvcrit_smallest()
        if Y[X].ct==0:
            Y[X].ct = np.sqrt(Y[X].mu/Y[X].rho)
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        Y[X].C2norm = UnVoigt(Y[X].C2/Y[X].mu)
        ## ct was determined from mu above and thus may not be the actual transverse sound speed (if scale_by_mu='crude')
        scaling[X] = min(1,round(Y[X].vcrit_smallest/Y[X].ct+5e-3,2))
        beta_scaled[X] = scaling[X]*beta

    
    def maincomputations(i):
        '''wrap all main computations into a single function definition to be run in a parallelized loop'''
        X = metal[i]
        with open(f"beta_{X}.dat","w", encoding="utf8") as betafile:
            betafile.write('\n'.join("{:.5f}".format(bti) for bti in beta_scaled[X]))
    
        dislocation = Y[X]
        
        ### compute dislocation displacement gradient uij and line tension LT
        def compute_lt(j):
            dislocation.computeuij(beta=beta_scaled[X][j])
            dislocation.computeEtot()
            dislocation.computeLT()
            return 4*np.pi*dislocation.LT
            
        LT = np.array([compute_lt(j) for j in range(len(beta))])
        
        # write the results to disk (and backup previous results if they exist):
        if os.access(fname:=f"LT_{X}.dat.xz", os.R_OK):
            shutil.move(fname,fname[:-3]+".bak.xz")
        with lzma.open(f"LT_{X}.dat.xz","wt") as LTfile:
            LTfile.write(f"### dimensionless line tension prefactor LT(beta,theta) for {X}, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.\n")
            LTfile.write('beta/theta[pi]\t' + '\t'.join("{:.4f}".format(thi) for thi in theta[1:-1]/np.pi) + '\n')
            for j in range(len(beta)):
                LTfile.write(f"{beta_scaled[X][j]:.4f}\t" + '\t'.join("{:.6f}".format(thi) for thi in LT[j]) + '\n')

        return 0
        
    # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
    print(f"Computing the line tension for: {metal}")
    if Ncores == 1 and Nbeta >=1:
        [maincomputations(i) for i in range(len(metal))]
    elif Nbeta<1:
        print("skipping line tension calculations, Nbeta>0 required")
    else:
        Parallel(n_jobs=Ncores)(delayed(maincomputations)(i) for i in range(len(metal)))

################## create plots ################
    if skip_plots:
        print("skipping plots as requested")
        plt_metal = []
    else:
        plt_metal = metal
    skip_plt = []

## load data from LT calculation
    LT = {}
    for X in plt_metal:
        try:
            LT[X] = read_2dresults(f"LT_{X}.dat") ## for every X, LT has Nbeta rows and Ntheta columns
        except FileNotFoundError:
            skip_plt.append(X)
            
    def mkLTplots(X):
        '''generates nice plots showing the dislocation line tension of metal X'''
        namestring = f"{X}"
        beta_trunc = [j for j in LT[X].index if j <=Y[X].vcrit_smallest/Y[X].ct]
        if X in metal_symm:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.2))
            LT_trunc = LT[X].iloc[:len(beta_trunc),int((LT[X].shape[1]-1)/2):].to_numpy()
            y_msh , x_msh = np.meshgrid(LT[X].columns[int((LT[X].shape[1]-1)/2):],beta_trunc)
            plt.yticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        else:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,4.5))
            LT_trunc = LT[X].iloc[:len(beta_trunc)].to_numpy()
            y_msh , x_msh = np.meshgrid(LT[X].columns,beta_trunc)
            plt.yticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$-\pi/2$", r"$-3\pi/8$", r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if X=='ISO':
            plt.xlabel(r'$\beta_\mathrm{t}=v/c_\mathrm{t}$',fontsize=fntsize)
            plt.title(r'Isotropic',fontsize=fntsize)
        else:
            plt.xlabel(r'$\beta_{\bar\mu}$',fontsize=fntsize)
            plt.title(namestring,fontsize=fntsize)
        plt.ylabel(r'$\vartheta$',rotation=0,fontsize=fntsize)
        colmsh = plt.pcolormesh(x_msh,y_msh, LT_trunc, vmin=-0.5, vmax=2, cmap = plt.cm.rainbow, shading='gouraud')
        plt.colorbar()
        plt.contour(x_msh,y_msh,LT_trunc, colors=('black','red','black','black','black','black'), levels=[-0.5,0,0.5,1,1.5,2], linewidths=[0.7,1.0,0.7,0.7,0.7,0.7], linestyles=['solid','solid','dashed','dashdot','dotted','solid'])
        colmsh.set_rasterized(True)
        plt.axhline(0, color='grey', linewidth=0.5, linestyle='dotted')
        plt.savefig(f"LT_{X}.pdf",format='pdf',bbox_inches='tight',dpi=450)
        plt.close()

    for X in set(plt_metal).difference(set(skip_plt)):
        mkLTplots(X)
    
################################################
    if Ntheta2==0 or Ntheta2 is None:
        sys.exit()
    
    print(f"Computing critical velocities for: {metal}")
    for X in metal:
        if X in metal_symm:
            current_symm=True
            Y[X].computevcrit(theta=np.linspace(0,np.pi/2,Ntheta2))
        else:
            current_symm=False
            Y[X].computevcrit(theta=np.linspace(-np.pi/2,np.pi/2,2*Ntheta2-1))
    
    ## write vcrit results to disk, then plot
    with open("vcrit.dat","w", encoding="utf8") as vcritfile:
        vcritfile.write("theta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*Ntheta2-1)) + '\n')
        vcritfile.write("metal / vcrit[m/s] (3 solutions per angle)\n")
        for X in sorted(list(set(metal))):
            for i in range(3):
                vcritfile.write(f"{X}\t" + '\t'.join("{:.0f}".format(thi) for thi in np.flipud(Y[X].vcrit_all[i+1,:])) + '\n')
                
    def mkvcritplot(X,Ntheta):
        '''Generates a plot showing the limiting (or critical) dislocation glide velocities as a function of character angle.'''
        fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(4.5,3.5))
        plt.tight_layout(h_pad=0.0)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        vcrit0 = Y[X].vcrit_all[1:].T
        if len(vcrit0)==Ntheta:
            plt.xticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(0,np.pi/2,Ntheta)
        else:
            plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
        ax1.axis((min(thetapoints),max(thetapoints),np.nanmin(vcrit0)*0.97,np.nanmax(vcrit0)*1.02)) ## define plot range
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.set_ylabel(r'$v_\mathrm{c}$[m/s]',fontsize=fntsize)
        ax1.set_title(f"3 vcrit solutions for {X}",fontsize=fntsize)
        for i in range(3):
            ax1.plot(thetapoints,vcrit0[:,i])
        ax1.set_xlabel(r'$\vartheta$',fontsize=fntsize)
        plt.savefig(f"vcrit_{X}.pdf",format='pdf',bbox_inches='tight')
        plt.close()
    
    for X in sorted(list(set(metal))):
        mkvcritplot(X,Ntheta2)
