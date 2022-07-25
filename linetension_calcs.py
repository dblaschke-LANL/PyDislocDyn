#!/usr/bin/env python3
# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - July 22, 2022
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
from scipy import optimize, ndimage
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
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
import metal_data as data
from elasticconstants import elasticC2, Voigt, UnVoigt
from polycrystal_averaging import metal_props, loadinputfile
from dislocations import StrohGeometry, ompthreads, printthreadinfo, elbrak1d
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/max(2,ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used) ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
    # Ncores = -2
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = Ncpus = 1 ## must be 1 without joblib
Kcores = max(Ncores,int(min(Ncpus/2,Ncores*ompthreads/2))) ## use this for parts of the code where openmp is not supported

## choose which shear modulus to use for rescaling to dimensionless quantities
## allowed values are: 'crude', 'aver', and 'exp'
## choose 'crude' for mu = (c11-c12+2c44)/4, 'aver' for mu=Hill average  (resp. improved average for cubic), and 'exp' for experimental mu supplemented by 'aver' where data are missing
## (note: when using input files, 'aver' and 'exp' are equivalent in that mu provided in that file will be used and an average is only computed if mu is missing)
scale_by_mu = 'exp'
skip_plots=False ## set to True to skip generating line tension plots from the results
write_vcrit=False ## if set to True and input files with missing vcrit are processed, append vcrit_smallest to that input file (increase Ntheta2 below for increased accuracy of vcrit_smallest)
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
OPTIONS = {"Ntheta":int, "Ntheta2":int, "Nbeta":int, "Nphi":int, "scale_by_mu":str, "skip_plots":bool, "write_vcrit":bool, "bccslip":str, "hcpslip":str, "Ncores":int}

#### input data:
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals)))
metal = sorted(metal + ['ISO']) ### test isotropic limit

## define a new method to compute critical velocities within the imported pca.metal_props class:
def computevcrit_stroh(self,Ntheta,Ncores=Kcores,symmetric=False,cache=False,theta_list=None,setvcrit=True):
    '''Computes the 'critical velocities' of a dislocation for the number Ntheta (resp. 2*Ntheta-1) of character angles in the interval [0,pi/2] (resp. [-pi/2, pi/2] if symmetric=False),
       i.e. the velocities that will lead to det=0 within the StrohGeometry.
       Optionally, an explicit list of angles in units of pi/2 may be passed via theta_list (Ntheta is a required argument, but is ignored in this case).
       Additionally, the crystal symmetry must also be specified via sym= one of 'iso', 'fcc', 'bcc', 'hcp', 'tetr', 'trig', 'orth', 'mono', 'tric'.
       Note that this function does not check for subtle cancellations that may occur in the dislocation displacement gradient at those velocities;
       use the Dislocation class and its .computevcrit(theta) method for fully automated determination of the lowest critical velocity at each character angle.'''
    cc11, cc12, cc13, cc14, cc22, cc23, cc33, cc44, cc55, cc66 = sp.symbols('cc11, cc12, cc13, cc14, cc22, cc23, cc33, cc44, cc55, cc66', real=True)
    bt2, phi = sp.symbols('bt2, phi', real=True)
    substitutions = {cc11:self.C2[0,0]/1e9, cc12:self.C2[0,1]/1e9, cc44:self.C2[3,3]/1e9}
    if self.sym in ['hcp', 'tetr', 'trig', 'orth', 'mono', 'tric']:
        substitutions.update({cc13:self.C2[0,2]/1e9, cc33:self.C2[2,2]/1e9})
    if self.sym in ['tetr', 'orth', 'mono', 'tric']:
        substitutions.update({cc66:self.C2[5,5]/1e9})
    if self.sym=='trig' or self.sym=='tric':
        substitutions.update({cc14:self.C2[0,3]/1e9})
    if self.sym in ['orth', 'mono', 'tric']:
        substitutions.update({cc22:self.C2[1,1]/1e9, cc23:self.C2[1,2]/1e9, cc55:self.C2[4,4]/1e9})
    if self.sym=='mono' or self.sym=='tric':
        cc15, cc25, cc35, cc46 = sp.symbols('cc15, cc25, cc35, cc46', real=True)
        substitutions.update({cc15:self.C2[0,4]/1e9, cc25:self.C2[1,4]/1e9, cc35:self.C2[2,4]/1e9, cc46:self.C2[3,5]/1e9})
    if self.sym=='tric':
        cc16, cc24, cc26, cc34, cc36, cc45, cc56 = sp.symbols('cc16, cc24, cc26, cc34, cc36, cc45, cc56', real=True)
        substitutions.update({cc16:self.C2[0,5]/1e9, cc24:self.C2[1,3]/1e9, cc26:self.C2[1,5]/1e9, cc34:self.C2[2,3]/1e9, cc36:self.C2[2,5]/1e9, cc45:self.C2[3,4]/1e9, cc56:self.C2[4,5]/1e9})
    if self.sym=='iso':
        C2 = elasticC2(c12=cc12,c44=cc44)
    elif self.sym=='fcc' or self.sym=='bcc':
        C2 = elasticC2(c11=cc11,c12=cc12,c44=cc44)
    elif self.sym=='hcp':
        C2 = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33)
    elif self.sym=='tetr':
        C2 = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33,c66=cc66)
    elif self.sym=='trig':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc14,cc33,cc44))
    elif self.sym=='orth':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc22,cc23,cc33,cc44,cc55,cc66))
    elif self.sym=='mono':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc15,cc22,cc23,cc25,cc33,cc35,cc44,cc46,cc55,cc66))
    elif self.sym=='tric':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc14,cc15,cc16,cc22,cc23,cc24,cc25,cc26,cc33,cc34,cc35,cc36,cc44,cc45,cc46,cc55,cc56,cc66))
    else:
        raise ValueError("sym={} not implemented".format(self.sym))
    if symmetric or Ntheta==1:
        Theta = np.multiply((sp.pi/2),np.linspace(0,1,Ntheta))
    else:
        Theta = (sp.pi/2)*np.linspace(-1,1,2*Ntheta-1)
        Ntheta=len(Theta)
    if theta_list is not None:
        Theta = np.multiply((sp.pi/2),np.asarray(theta_list))
        Ntheta = len(Theta)
    def compute_bt2(N,m0,C2,bt2,cc44=cc44):
        NC2N = np.dot(N,np.dot(N,C2))
        thedot = np.dot(N,m0)
        for a in sp.preorder_traversal(thedot):
            if isinstance(a, sp.Float):
                thedot = thedot.subs(a, round(a, 12))
        thematrix = NC2N - bt2*cc44*(thedot**2)*np.diag((1,1,1))
        thedet = sp.det(sp.Matrix(thematrix))
        out = sp.solve(thedet,bt2)
        if len(out)==2: out.append(np.nan) ## only happens in the isotropic limit
        return out
    def computevcrit(b,n0,C2,Ntheta,bt2=bt2,Ncores=Ncores):
        Ncores = min(Ncores,Ncpus) # don't over-commit and don't fail if joblib not loaded
        t = np.zeros((Ntheta,3))
        m0 = np.zeros((Ntheta,3))
        N = np.zeros((Ntheta,3),dtype=object)
        for thi in range(Ntheta):
            t[thi] = sp.cos(Theta[thi])*b + sp.sin(Theta[thi])*np.cross(b,n0)
            m0[thi] = np.cross(n0,t[thi])
            N[thi] = n0*sp.cos(phi) - m0[thi]*sp.sin(phi)
        bt2_curr = np.zeros((Ntheta,3),dtype=object)
        foundcache=False
        if cache is not False and cache is not True: ## check if cache contains needed results
            for i in range(len(cache)):
                if cache[i][3]==self.sym and abs(np.dot(cache[i][0],self.b)-1)<1e-15 and abs(np.dot(cache[i][1],self.n0)-1)<1e-15 and cache[i][2].shape==bt2_curr.shape:
                    foundcache=i
                    bt2_curr=cache[foundcache][2] ## load user-provided from previous calculation
                    break
        if cache is False or cache is True or foundcache is False:
            if Ncores == 1:
                for thi in range(Ntheta):
                    bt2_curr[thi] = compute_bt2(N[thi],m0[thi],C2,bt2)
            else:
                bt2_curr = np.array(Parallel(n_jobs=Ncores)(delayed(compute_bt2)(N[thi],m0[thi],C2,bt2) for thi in range(Ntheta)),dtype=object)
        if foundcache is False and cache is not False and cache is not True:
            cache.append((self.b,self.n0,bt2_curr,self.sym)) ## received a cache, but it didn't contain what we need
        if cache is True:
            self.cache_bt2 = bt2_curr.copy()
        vcrit = np.zeros((2,Ntheta,3))
        def findmin(bt2_curr,substitutions=substitutions,phi=phi,norm=(self.C2[3,3]/self.rho)):
            bt2_res = np.zeros((3,2),dtype=complex)
            for i in range(len(bt2_curr)):
                bt2_curr[i] = (sp.S(bt2_curr[i]).subs(substitutions))
                fphi = sp.lambdify((phi),bt2_curr[i],modules=["scipy"])
                def f(x):
                    out = fphi(x)
                    return np.real(out)
                with np.errstate(invalid='ignore'):
                    minresult = optimize.minimize_scalar(f,method='bounded',bounds=(0.0,2*np.pi))
                    if minresult.success is not True: print(f"Warning ({self.name}):\n{minresult}")
                    bt2_res[i,0] = minresult.x
                bt2_res[i,1] = bt2_curr[i].subs({phi:bt2_res[i,0]})
                if abs(np.imag(bt2_res[i,1]))>1e-6 or not minresult.success: bt2_res[i,:]=bt2_res[i,:]*float('nan') ## only keep real solutions from successful minimizations
            return np.array([np.sqrt(norm*np.real(bt2_res[:,1])),np.real(bt2_res[:,0])])
        if Ncores == 1:
            for thi in range(Ntheta):
                vcrit[:,thi] = findmin(bt2_curr[thi],substitutions,phi,(self.C2[3,3]/self.rho))
        else:
            vcrit = np.moveaxis(np.array(Parallel(n_jobs=Ncores)(delayed(findmin)(bt2_curr[thi],substitutions,phi,self.C2[3,3]/self.rho) for thi in range(Ntheta))),1,0)
        return vcrit
    
    finalresult = computevcrit(self.b,self.n0,C2,Ntheta,Ncores=Ncores)
    if setvcrit: self.vcrit = finalresult
    return finalresult[0]
metal_props.computevcrit_stroh=computevcrit_stroh

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
        self.sym = sym
        self.vcrit_all = None
        self.Rayleigh = None
    
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
    
    def findedgescrewindices(self,theta=None):
        '''Find the indices i where theta[i] is either 0 or +/-pi/2. If theta is omitted, assume theta=self.theta.'''
        if theta is None: theta=self.theta
        else: theta = np.asarray(theta)
        scrind = np.where(np.abs(theta)<1e-12)[0]
        out = [None,None]
        if len(scrind) == 1:
            out[0] = int(scrind)
        edgind = np.where(np.abs(theta-np.pi/2)<1e-12)[0]
        if len(edgind) == 1:
            out[1] = int(edgind)
        negedgind = np.where(np.abs(theta+np.pi/2)<1e-12)[0]
        if len(negedgind) == 1:
            out.append(int(negedgind))
        return out
        
    def computevcrit_barnett(self, theta_list=None, setvcrit=True, verbose=False):
        '''Computes the limiting velocities following Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5.
           All parameters are optional: unless a list of character angles is passed explicitly via theta_list,
           we calculate limiting velocities for all character angles in self.theta.
           Option setvcrit determines whether or not to overwrite attribute self.vcrit.
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
                tmpout = (-P/3 + 2*np.sqrt(-a/3)*np.cos((gamma+2*i*np.pi)/3))
                return np.abs(np.sqrt(tmpout*norm)/np.cos(phi))
            for i in range(3):
                ## default minimizer sometimes yields nan, but bounded method doesn't always find the smallest value, so run both:
                minresult1 = optimize.minimize_scalar(findvlim,bounds=(0.0,2*np.pi),args=(i))
                minresult2 = optimize.minimize_scalar(findvlim,method='bounded',bounds=(0.0,2*np.pi),args=(i))
                if verbose and not (minresult1.success and minresult2.success):
                    print(f"Warning ({self.name}, theta={theta[th]}):\n{minresult1}\n{minresult2}\n\n")
                ## always take the smaller result, ignore nan:
                choose = np.nanargmin(np.array([minresult1.fun,minresult2.fun]))
                if choose == 0: minresult = minresult1
                else: minresult = minresult2
                out[0,th,i] = minresult.fun
                out[1,th,i] = minresult.x
        if setvcrit: self.vcrit = out
        return out[0]
        
    def computevcrit_screw(self):
        '''Compute the limiting velocity of a pure screw dislocation analytically, provided the slip plane is a reflection plane, use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        scrind = self.findedgescrewindices()[0]
        A = self.C2_aligned[scrind][4,4]
        B = 2*self.C2_aligned[scrind][3,4]
        C = self.C2_aligned[scrind][3,3]
        test = np.abs(self.C2_aligned[scrind]/self.C2[3,3]) ## check for symmetry requirements
        if test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4] < 1e-12:
            self.vcrit_screw = np.sqrt((A-B**2/(4*C))/self.rho)
        return self.vcrit_screw
    
    def computevcrit_edge(self):
        '''Compute the limiting velocity of a pure edge dislocation analytically, provided the slip plane is a reflection plane (cf. L. J. Teutonico 1961, Phys. Rev. 124:1039), use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        edgind = int(np.where(np.abs(self.theta-np.pi/2)<1e-12)[0])
        test = np.abs(self.C2_aligned[edgind]/self.C2[3,3]) ## check for symmetry requirements
        if test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4] < 1e-12:
            c11=self.C2_aligned[edgind][0,0]
            c22=self.C2_aligned[edgind][1,1]
            c66=self.C2_aligned[edgind][5,5]
            c12=self.C2_aligned[edgind][0,1]
            if test[0,5] + test[1,5] < 1e-12:
                self.vcrit_edge = np.sqrt(min(c66,c11)/self.rho)
                ## cover case of q<0 (cf. Teutonico 1961 paper, eq (39); line above was only for q>0):
                if ((c11*c22-c12**2-2*c12*c66) - (c22+c66)*min(c66,c11))/(c22*c66)<0:
                    ## analytic solution to Re(lambda=0) in eq. (39) (with sp.solve); sqrt below is real because of if statement above:
                    minval = (2*np.sqrt(c22*c66*(-c11*c22 + c11*c66 + c12**2 + 2*c12*c66 + c22*c66))*(c12 + c66) - (-c11*c22**2 + c11*c22*c66 + c12**2*c22 + c12**2*c66 + 2*c12*c22*c66 + 2*c12*c66**2 + 2*c22*c66**2))/((c22 - c66)**2)
                    if minval>0:
                        # print(f"q<0: {self.vcrit_edge}, {np.sqrt(minval/self.rho)}")
                        self.vcrit_edge = min(self.vcrit_edge,np.sqrt(minval/self.rho))
            else:
                c16 = self.C2_aligned[edgind][0,5]
                c26 = self.C2_aligned[edgind][1,5]
                def theroot(y,rv2):
                    K4 = c66*c22-c26**2
                    K3 = 2*(c26*c12-c16*c22)
                    K2 = (c11*c22-c12**2-2*c12*c66+2*c16*c26) - (c22+c66)*rv2
                    K1 = 2*(c16*c12-c26*c11) + 2*rv2*(c16+c26)
                    K0 = (c11-rv2)*(c66-rv2)-c16**2
                    return K0+K1*y+K2*y**2+K3*y**3+K4*y**4
                y,rv2 = sp.symbols('y,rv2')
                ysol = sp.solve(theroot(y,rv2),y) ## 4 complex roots as fcts of rv2=rho*v**2
                yfct=sp.lambdify(rv2,ysol,modules=["scipy"])
                def f(x):
                    return np.abs(np.asarray(yfct(x)).imag.prod()) ## lambda=i*y, and any Re(lambda)=0 implies a divergence/limiting velocity
                with np.errstate(invalid='ignore'):
                    rv2limit = optimize.fsolve(f,1e5)
                    if f(rv2limit) < 1e-12: ## check if fsolve was successful
                        self.vcrit_edge = np.sqrt(float(rv2limit)/self.rho)
        return self.vcrit_edge

    def computevcrit(self,theta=None,cache=False,Ncores=Kcores,set_screwedge=True,setvcrit=True,use_bruteforce=False):
        '''Compute the lowest critical (or limiting) velocities for all dislocation character angles within list 'theta'. If theta is omitted, we fall back to attribute .theta (default).
        The list of results will be stored in method .vcrit_all, i.e. .vcrit_all[0]=theta and .vcrit[1] contains the corresponding limiting velocities.
        Option set_screwedge=True guarantees that attributes .vcrit_screw and .vcrit_edge will be set, and 'setvrit=True' will overwrite self.vcrit.
        Options 'cache' and 'Ncores' are only used if 'use_bruteforce=True' and will speed up the (much slower) calculation in that case.'''
        if theta is None:
            theta=self.theta
        indices = self.findedgescrewindices(theta)
        self.vcrit_all = np.empty((2,len(theta)))
        self.vcrit_all[0] = theta
        if self.sym=='iso':
            self.computevcrit_screw()
            self.vcrit_all[1] = self.vcrit_screw
        elif use_bruteforce:
            self.vcrit_all[1] = np.nanmin(self.computevcrit_stroh(len(theta),cache=cache,theta_list=np.asarray(theta)*2/np.pi,Ncores=Ncores,setvcrit=setvcrit),axis=1)
        else:
            self.vcrit_all[1] = np.nanmin(self.computevcrit_barnett(theta_list=np.asarray(theta),setvcrit=setvcrit),axis=1)
        if indices[0] is not None:
            self.computevcrit_screw()
            if self.vcrit_screw is not None:
                self.vcrit_all[1,indices[0]] = self.vcrit_screw
            elif set_screwedge:
                self.vcrit_screw = self.vcrit_all[1,indices[0]]
        if indices[1] is not None:
            self.computevcrit_edge()
            if self.vcrit_edge is not None:
                self.vcrit_all[1,indices[1]] = self.vcrit_edge
                if len(indices) == 3:
                    self.vcrit_all[1,indices[2]] = self.vcrit_edge
            elif set_screwedge:
                self.vcrit_edge = self.vcrit_all[1,indices[1]]
                if len(indices) == 3:
                    self.vcrit_edge = min(self.vcrit_edge,self.vcrit_all[1,indices[2]])
        return self.vcrit_all[1]
    
    def findvcrit_smallest(self,cache=False,Ncores=Kcores,xatol=1e-2,use_bruteforce=False):
        '''Computes the smallest critical velocity, which subsequently is stored as attribute .vcrit_smallest and the full result of scipy.minimize_scalar is returned
           (as type 'OptimizeResult' with its 'fun' being vcrit_smallest and 'x' the associated character angle theta).
           The absolute tolerance for theta can be passed via xatol; in order to improve accuracy and speed of this routine, we make use of computevcrit with Ntheta>=11 resolution
           in order to be able to pass tighter bounds to the subsequent call to minimize_scalar(). If .vcrit_all already exists in sufficient resolution from an earlier call,
           this step is skipped.'''
        if self.vcrit_all is None or self.vcrit_all.shape[1]<11:
            self.computevcrit(theta=np.linspace(self.theta[0],self.theta[-1],11),cache=cache,Ncores=Ncores,set_screwedge=False,setvcrit=False,use_bruteforce=use_bruteforce)
        vcrit_smallest = np.nanmin(self.vcrit_all[1])
        thind = np.where(self.vcrit_all[1]==vcrit_smallest)[0][0] ## find index of theta so that we may pass tighter bounds to minimize_scalar below for more accurate (and faster) results
        bounds=(max(-np.pi/2,self.vcrit_all[0][max(0,thind-1)]),min(np.pi/2,self.vcrit_all[0][min(thind+1,len(self.vcrit_all[0])-1)]))
        if use_bruteforce:
            def f(x):
                return np.min(self.computevcrit_stroh(1,theta_list=[x*2/np.pi],setvcrit=False)) ## cannot use cache because 1) we keep calculating for different theta values and 2) cache only checks length of theta but not actual values (so not useful for other materials either)
        else:
            def f(x):
                return np.min(self.computevcrit_barnett(theta_list=[x],setvcrit=False))
        if self.sym=='iso': result = vcrit_smallest
        else: result = optimize.minimize_scalar(f,method='bounded',bounds=bounds,options={'xatol':xatol})
        if self.sym=='iso': self.vcrit_smallest = vcrit_smallest
        elif result.success: self.vcrit_smallest = min(result.fun,vcrit_smallest)
        return result
    
    def findRayleigh(self):
        '''Computes the Rayleigh wave speed for every dislocation character self.theta.'''
        Rayleigh=np.zeros((self.Ntheta))
        norm = self.C2[3,3] # use c44
        C2norm = UnVoigt(self.C2/norm)
        if self.vcrit_all is None: self.computevcrit(set_screwedge=False) ## need it as an upper bound on the Rayleigh speed
        if len(self.vcrit_all[1])==self.Ntheta: vcrit = self.vcrit_all[1]
        else: vcrit = ndimage.interpolation.zoom(self.vcrit_all[1],self.Ntheta/len(self.vcrit_all[1]))
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
                return Rayleighcond(np.trapz(B,x=self.phi,axis=0)/(4*np.pi**2))
            bounds=(0.0,vcrit[th]*np.sqrt(self.rho/norm))
            result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=bounds,options={'xatol':1e-12})
            # if result.fun>=1e-3: print(f"{bounds}\n{result}")  ## if this failed, try enlarging the search interval slightly above vcrit (there was some numerical uncertainty there too):
            if result.success and result.fun>=1e-3: result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=(0.5*bounds[1],1.25*bounds[1]),options={'xatol':1e-12})
            if result.fun>=1e-3 or not result.success: print(f"Failed: Rayleigh not found in [{bounds[0]},{1.25*bounds[1]}]\n",result)
            if result.success and result.fun<1e-3: Rayleigh[th] = result.x * np.sqrt(norm/self.rho)
        self.Rayleigh = Rayleigh
        return Rayleigh
    
    def plotdisloc(self,beta=None,character='screw',component=[2,0],a=None,eta_kw=None,etapr_kw=None,t=None,shift=None,fastapprox=False,Nr=250,nogradient=False,cmap = plt.cm.rainbow,skipcalc=False,showplt=False,lim=(-1,1)):
        '''Generates a plot of the requested component of the dislocation displacement gradient.
           Optional arguments are: the normalized velocity 'beta'=v/self.ct (defaults to self.beta, assuming one of the .computeuij() methods were called earlier).
           'character' is either 'edge', 'screw' (default), or an index of self.theta, and 'component' is
           a list of two indices indicating which component of displacement gradient u[ij] to plot.
           The steady-state solution is plotted unless an acceleration 'a' (or a more general function eta_kw) is passed. In the latter case,
           'slipsystem' is required except for those metals where its keyword coincides with self.sym (see documentation of self.computeuij_acc()
           for details on capabilities and limitations of the current implementation of the accelerating solution).
           Option nogradient=True will plot the displacement field instead of its gradient; this option must be combined with an integer value for 'component'
           and is currently only implemented for steady-state solutions (a=None).
           Option skipcalc=True (implied when beta is not set) may be passed to plot results of an earlier calculation with the same input parameters (useful
           for plotting multiple components of the dislocation field).
           Colormap and its limits are set with options 'cmap' and 'lim', respectively.` 
           If option 'showplt' is set to 'True', the figure is shown in an interactive session in addition to being saved to a file. Warning: this will only work
           if the user sets matplotlib's backend to an interactive one after PyDislocDyn was loaded (e.g. by calling %matplotlib inline).'''
        if beta==None:
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
                self.computeuij(beta=beta, r=r, nogradient=nogradient) ## compute steady state field
                if not nogradient:
                    self.alignuij() ## self.rot was computed as a byproduct of .alignC2() above
                else:
                    self.uij_aligned = np.zeros(self.uij.shape)
                    for th in range(len(self.theta)):
                        for ri in range(Nr):
                            self.uij_aligned[:,th,ri] = np.round(np.dot(self.rot[th],self.uij[:,th,ri]),15)
            if character == 'screw':
                index = int(np.where(abs(self.theta)<1e-12)[0])
            elif character == 'edge':
                index = int(np.where(abs(self.theta-np.pi/2)<1e-12)[0])
            else:
                index=character
            if not nogradient:
                namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}{character}_{self.name}_v{beta*self.ct:.0f}.pdf"
                uijtoplot = self.uij_aligned[component[0],component[1],index]
            else:
                namestring = f"u{xylabel[component]}{character}_{self.name}_v{beta*self.ct:.0f}.pdf"
                uijtoplot = self.uij_aligned[component,index]
        elif character=='screw' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_screw(a,beta,burgers=self.burgers,fastapprox=fastapprox,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None: acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}screw_{self.name}_v{beta*self.ct:.0f}_a{acc:}.pdf"
            uijtoplot = self.uij_acc_screw_aligned[component[0],component[1]]
        elif character=='edge' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_edge(a,beta,burgers=self.burgers,fastapprox=fastapprox,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None: acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}edge{self.name}_v{beta*self.ct:.0f}_a{acc:}.pdf"
            uijtoplot = self.uij_acc_edge_aligned[component[0],component[1]]
        else:
            raise ValueError("not implemented")
        plotuij(uijtoplot,r,self.phi,lim=lim,showplt=showplt,title=namestring,savefig=namestring,fntsize=fntsize,axis=(-0.5,0.5,-0.5,0.5),figsize=(3.5,4.0))
        
    def __repr__(self):
        return  "DISLOCATION\n" + metal_props.__repr__(self) + "\n" + StrohGeometry.__repr__(self)

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
    if 'name' in inputparams.keys():
        name = inputparams['name']
    else:
        name = str(fname)
    if 'Millerb' in inputparams.keys() or 'Millern0' in inputparams.keys():
        temp = metal_props(sym,name) ## need a metal_props method to convert to Cartesian b, n0
        temp.populate_from_dict(inputparams)
        b = temp.b
        n0 = temp.n0
    else:
        b = np.asarray(inputparams['b'].split(','),dtype=float)
        n0 = np.asarray(inputparams['n0'].split(','),dtype=float)
    if theta is None:
        if 'symmetric' in inputparams.keys(): symmetric = inputparams['symmetric']
        if symmetric == True or symmetric == 'True' or Ntheta<=2: theta = np.linspace(0,np.pi/2,Ntheta)
        else: theta = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
    out = Dislocation(sym=sym, name=name, b=b, n0=n0, theta=theta, Nphi=Nphi)
    out.populate_from_dict(inputparams)
    out.filename = fname ## remember which file we read
    if isotropify and sym != 'iso': # bypass if we're already isotropic
        inputparams['sym'] = 'iso'
        inputparams['name'] = name+'_ISO'
        if 'lam' in inputparams.keys(): inputparams.pop('lam') ## ignore if read from file, use averages instead
        if 'mu' in inputparams.keys(): inputparams.pop('mu')
        out.lam = out.mu = None
        out.init_all()
        inputparams['c12'] = out.lam
        inputparams['c44'] = out.mu
        inputparams['a'] = np.cbrt(out.Vc)
        if 'c123' in inputparams.keys():
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

def plotuij(uij,r,phi,lim=(-1,1),showplt=True,title=None,savefig=False,fntsize=11,axis=(-0.5,0.5,-0.5,0.5),figsize=(3.5,4.0)):
    '''Generates a heat map plot of a 2-dim. dislocation field, where the x and y axes are in units of Burgers vectors and
    the color-encoded values are dimensionless displacement gradients.
    Required parameters are the 2-dim. array for the displacement gradient field, uij, as well as arrays r and phi for 
    radius (in units of Burgers vector) and polar angle; note that the pot will be converted to Cartesian coordinates.
    Options include, the colorbar limits "lim", whether or not to call plt.show(), an optional title for the plot,
    which filename (if any) to save it as, the fontsize to be used, and the plot range to be passed to plt.axis().'''
    phi_msh , r_msh = np.meshgrid(phi,r)
    x_msh = r_msh*np.cos(phi_msh)
    y_msh = r_msh*np.sin(phi_msh)
    plt.figure(figsize=figsize)
    plt.axis(axis)
    plt.xticks(np.linspace(*axis[:2],5),fontsize=fntsize)
    plt.yticks(np.linspace(*axis[2:],5),fontsize=fntsize)
    plt.xlabel(r'$x[b]$',fontsize=fntsize)
    plt.ylabel(r'$y[b]$',fontsize=fntsize)
    if title is not None: plt.title(title,fontsize=fntsize)
    if np.all(uij==0): raise ValueError('Dislocation field contains only zeros, forgot to calculate?')
    if uij.shape != (len(r),len(phi)):
        uij = np.outer(1/r,uij)
    colmsh = plt.pcolormesh(x_msh, y_msh, uij, vmin=lim[0], vmax=lim[-1], cmap = plt.cm.rainbow, shading='gouraud')
    colmsh.set_rasterized(True)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize = fntsize)
    if showplt: plt.show()
    if savefig is not False: plt.savefig(savefig,format='pdf',bbox_inches='tight')
    plt.close()

def read_2dresults(fname):
    '''Read results (such as line tension or drag coefficient) from file fname and return a Pandas DataFrame where index=beta and columns=theta.'''
    if os.access((newfn:=fname+'.xz'), os.R_OK): fname = newfn # new default
    elif os.access((newfn:=fname), os.R_OK): pass # old default
    elif os.access((newfn:=fname+'.gz'), os.R_OK): fname = newfn
    else: raise FileNotFoundError(f'tried {fname}.xz, {fname}, and {fname}.gz')
    out = pd.read_csv(fname,skiprows=1,index_col=0,sep='\t')
    out.columns = pd.to_numeric(out.columns)*np.pi
    out.index.name='beta'
    out.columns.name='theta'
    return out

def parse_options(arglist,optionlist=OPTIONS,globaldict=globals()):
    '''Search commandline arguments passed to this script for known options to set by comparing to a list of keyword strings "optionlist".
    These will then override default variables set above in this script. This function also returns a copy of 'arglist' stripped of all 
    option calls for further processing (e.g. opening input files that were passed etc.).'''
    out = arglist
    setoptions = [i for i in out if "--" in i and "=" in i and i[:2]=="--"]
    for i in setoptions:
        out.remove(i)
        key,val = i[2:].split("=")
        if key in optionlist:
            globaldict[key] = optionlist[key](val)
            print(f"setting {key}={globaldict[key]}")
    time.sleep(1) ## avoid race conditions after changing global variables
    return out
    
### start the calculations
if __name__ == '__main__':
    printthreadinfo(Ncores,ompthreads)
    Y={}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args = parse_options(sys.argv[1:])
    ### set range & step sizes after parsing the commandline for options
    dtheta = np.pi/(Ntheta-2)
    theta = np.linspace(-np.pi/2-dtheta,np.pi/2+dtheta,Ntheta+1)
    beta = np.linspace(0,1,Nbeta)
    if len(sys.argv) > 1 and len(args)>0:
        try:
            inputdata = [readinputfile(i,init=False,theta=theta,Nphi=Nphi) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = metal_list = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError:
            ## only compute the metals the user has asked us to
            metal = args[0].split()
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
        with open("theta.dat","w") as thetafile:
            thetafile.write('\n'.join("{:.6f}".format(thi) for thi in theta[1:-1]))
                           
    C2 = {}
    scaling = {}
    beta_scaled = {}
    ## compute smallest critical velocity in ratio to the scaling velocity computed from the average shear modulus mu (see above):
    ## i.e. this corresponds to the smallest velocity leading to a divergence in the dislocation field at some character angle theta
    writesmallest = {}
    writeedge={}
    writescrew={}
    for X in metal:
        writeedge[X]=False
        writescrew[X]=False
        if Y[X].vcrit_screw is None:
            writescrew[X]=True ## remember for later
            Y[X].computevcrit_screw()
        if Y[X].vcrit_edge is None:
            writeedge[X]=True
            Y[X].computevcrit_edge()
        writesmallest[X] = False
        if Y[X].vcrit_smallest is None:
            Y[X].findvcrit_smallest()
            writesmallest[X] = True
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
        with open("beta_{}.dat".format(X),"w") as betafile:
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
            LTfile.write("### dimensionless line tension prefactor LT(beta,theta) for {}, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
            LTfile.write('beta/theta[pi]\t' + '\t'.join("{:.4f}".format(thi) for thi in theta[1:-1]/np.pi) + '\n')
            for j in range(len(beta)):
                LTfile.write("{:.4f}".format(beta_scaled[X][j]) + '\t' + '\t'.join("{:.6f}".format(thi) for thi in LT[j]) + '\n')

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
        namestring = "{}".format(X)
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
        plt.savefig("LT_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=450)
        plt.close()

    for X in set(plt_metal).difference(set(skip_plt)):
        mkLTplots(X)
    
################################################
    if Ntheta2==0 or Ntheta2 is None:
        sys.exit()
    
    print(f"Computing critical velocities for: {metal}")
    # if Kcores > 1: print(f"(using joblib parallelization with {Kcores} cores)")
    vcrit = {} # values contain critical velocities and associated polar angles
    for X in metal:
        if X in metal_symm:
            current_symm=True
            Y[X].computevcrit(theta=np.linspace(0,np.pi/2,Ntheta2))
            scrind=0
        else:
            current_symm=False
            Y[X].computevcrit(theta=np.linspace(-np.pi/2,np.pi/2,2*Ntheta2-1))
            scrind = Ntheta2-1
        if Y[X].sym=='iso': print(f"skipping isotropic {X}, vcrit=ct")
        else:
            vcrit[X]=np.empty(Y[X].vcrit.shape)
            for th in range(vcrit[X].shape[1]):
                tmplist=[]
                for i in range(3):
                    tmplist.append(tuple(Y[X].vcrit[:,th,i]))
                vcrit[X][:,th,:] = np.array(sorted(tmplist)).T
    
    ## write vcrit results to disk, then plot
    with open("vcrit.dat","w") as vcritfile:
        vcritfile.write("theta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*Ntheta2-1)) + '\n')
        vcritfile.write("metal / vcrit[m/s] (3 solutions per angle)\n")
        for X in sorted(list(set(metal).intersection(vcrit.keys()))):
            for i in range(3):
                vcritfile.write("{}\t".format(X) + '\t'.join("{:.0f}".format(thi) for thi in np.flipud(vcrit[X][0,:,i])) + '\n')
        vcritfile.write("\ntheta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*Ntheta2-1)) + '\n')
        vcritfile.write("metal / phi(vcrit)[pi] - polar angle at which vcrit occurs (3 solutions per theta)\n")
        for X in sorted(list(set(metal).intersection(vcrit.keys()))):
            for i in range(3):
                vcritfile.write("{}\t".format(X) + '\t'.join("{:.4f}".format(thi) for thi in np.flipud(vcrit[X][1,:,i]/np.pi)) + '\n')
                
    def mkvcritplot(X,vcrit,Ntheta):
        '''Generates a plot showing the velocities where det(nn)=0, which most of the time leads to divergences in the dislocation field.'''
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(5.5,6.0))
        plt.tight_layout(h_pad=0.0)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        vcrit0 = vcrit[0]
        vcrit1 = vcrit[1]/np.pi
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
        ax1.set_title("3 vcrit solutions for {}".format(X),fontsize=fntsize)
        plt.setp(ax1.get_xticklabels(), visible=False)
        for i in range(3):
            ax1.plot(thetapoints,vcrit0[:,i])
        ##
        ax2.axis((min(thetapoints),max(thetapoints),np.nanmin(vcrit1)*0.97,np.nanmax(vcrit1)*1.02)) ## define plot range
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.set_xlabel(r'$\vartheta$',fontsize=fntsize)
        ax2.set_ylabel(r'$\phi(v_\mathrm{c})/\pi$',fontsize=fntsize)
        for i in range(3):
            ax2.plot(thetapoints,vcrit1[:,i])
        plt.savefig("vcrit_{}.pdf".format(X),format='pdf',bbox_inches='tight')
        plt.close()
    
    for X in sorted(list(set(metal).intersection(vcrit.keys()))):
        mkvcritplot(X,vcrit[X],Ntheta2)
        if write_vcrit and not use_metaldata:
            with open(Y[X].filename,"a") as outf:
                if writesmallest[X] or writeedge[X] or writescrew[X]:
                    outf.write("## limiting velocities as computed by PyDislocDyn:\n")
                if writesmallest[X]:
                    print(f"writing vcrit_smallest to {X} ...")
                    outf.write(f"vcrit_smallest = {Y[X].vcrit_smallest:.0f}\n")
                if writeedge[X]:
                    print(f"writing vcrit_edge to {X}")
                    if Y[X].vcrit_edge is None: ## the case if the slip plane is not a reflection plane
                        if scrind>0:
                            outf.write(f"vcrit_edge = {min(np.min(vcrit[X][0][0]),np.min(vcrit[X][0][-1])):.0f}\n")
                        else:
                            outf.write(f"vcrit_edge = {np.min(vcrit[X][0][-1]):.0f}\n")
                    else:
                        outf.write(f"vcrit_edge = {Y[X].vcrit_edge:.0f}\n")
                if writescrew[X]:
                    print(f"writing vcrit_screw to {X}")
                    if Y[X].vcrit_screw is None: ## the case if the slip plane is not a reflection plane
                        outf.write(f"vcrit_screw = {np.min(vcrit[X][0][scrind]):.0f}\n")
                    else:
                        outf.write(f"vcrit_screw = {Y[X].vcrit_screw:.0f}\n")
        
