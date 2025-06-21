#!/usr/bin/env python3
# Compute averages of elastic constants for polycrystals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - June 21, 2025
'''This submodule defines the metal_props class which is one of the parents of the Dislocation class defined in linetension_calcs.py.
   Additional classes available in this module are IsoInvariants and IsoAverages which inherits from the former and is used to
   calculate averages of elastic constants. We also define a function, readinputfile, which reads a PyDislocDyn input file and
   returns an instance of the metal_props class.'''
#################################
import numpy as np
import sympy as sp
from sympy.solvers import solve
from scipy import optimize
import pandas as pd
##
from .elasticconstants import elasticC2, elasticC3, elasticS2, elasticS3, Voigt, UnVoigt, \
    convert_SOECiso, convert_TOECiso, strain_poly
from .utilities import str_to_array, loadinputfile

### compute various contractions/invariants of elastic constant/compliance tensors:
def invI1(C2):
    '''Computes the trace C_iijj of SOEC.'''
    return np.trace(np.trace(C2))
    ### einsum is slightly faster, but works only with numbers, not symbols for numpy<1.25

def invI2(C2):
    '''Computes the trace C_ijij of SOEC.'''
    return np.trace(np.trace(C2,axis1=0,axis2=2))

def invI3(C3):
    '''Computes the trace C_iijjkk of TOEC.'''
    return np.trace(np.trace(np.trace(C3)))

def invI4(C3):
    '''Computes the trace C_ijijkk of TOEC.'''
    return np.trace(np.trace(np.trace(C3,axis1=0,axis2=2)))

def invI5(C3):
    '''Computes the trace C_ijkijk of TOEC.'''
    return np.trace(np.trace(np.trace(C3,axis1=0,axis2=3),axis1=0,axis2=2))

# ### define some symbols and functions of symbols for the effective isotropic side of our calculations
lam, mu = sp.symbols('lam mu') ## Symbols for Lame constants
Murl, Murm, Murn = sp.symbols('Murl Murm Murn') ## Symbols for Murnaghan constants

class IsoInvariants:
    '''This class initializes isotropic second and third order elastic tensors and their invariants depending on the provided Lame and Murnaghan constants,
    which may be sympy symbols or numbers.'''
    def __init__(self,lam,mu,Murl,Murm,Murn):
        self.C2 = elasticC2(c12=lam,c44=mu)
        self.invI1 = sp.factor(invI1(self.C2))
        self.invI2 = sp.factor(invI2(self.C2))
        
        self.C3 = elasticC3(l=Murl,m=Murm,n=Murn)
        self.invI3 = sp.factor(invI3(self.C3))
        self.invI4 = sp.factor(invI4(self.C3))
        self.invI5 = sp.factor(invI5(self.C3))
        
        self.S2 = elasticS2(self.C2)
        self.invI1b = sp.factor(invI1(self.S2))
        self.invI2b = sp.factor(invI2(self.S2))
        
        self.S3 = elasticS3(self.S2,self.C3)
        self.invI3b = sp.factor(invI3(self.S3))
        self.invI4b = sp.factor(invI4(self.S3))
        self.invI5b = sp.factor(invI5(self.S3))

######### Voigt, Reuss, & Hill averages, etc.
class IsoAverages(IsoInvariants):
    '''This class computes a number of polycrystal averages of second and third order elastic constants.
    These include, Voigt, Reuss, Hill, in general, and "improved" averages for cubic crystals.
    Required inputs upon initialization are five sympy symbols representing the Lame and Murnaghan constants.'''
    def __init__(self,lam,mu,Murl,Murm,Murn):
        IsoInvariants.__init__(self,lam,mu,Murl,Murm,Murn)
        self.lam=lam
        self.mu=mu
        self.l = Murl
        self.m = Murm
        self.n = Murn
        self.voigt = None
        self.reuss = None
        self.hill = None
        self.improved = None
    
    def voigt_average(self,C2,C3=None):
        '''Computes the Voigt averages of elastic constants using the tensor(s) C2 and C3 of SOEC and TOEC (optional). The output is a dictionary of average values for the Lame and Murnaghan constants.
        If C3 is omitted (or 'None'), only the average Lame constants are computed.'''
        out = solve([self.invI1-invI1(C2),self.invI2-invI2(C2)],[self.lam,self.mu],dict=True)[0]
        ## combine all five Voigt-averaged answers into one dictionary for each metal (i.e. Voigt is a dictionary of dictionaries)
        if C3 is not None:
            out.update(solve([self.invI3-invI3(C3),self.invI4-invI4(C3),self.invI5-invI5(C3)],[self.l,self.m,self.n],dict=True)[0])
        self.voigt = out
        return out

    def reuss_average(self,S2,S3=None):
        '''Computes the Reuss averages of elastic constants using the second and third (optional) order compliances tensor(s) S2 and S3. The output is a dictionary of average values for the Lame and Murnaghan constants.
        If S3 is omitted (or 'None'), only the average Lame constants are computed.'''
        out = solve([self.invI1b-invI1(S2),self.invI2b-invI2(S2)],[self.lam,self.mu],dict=True)[0]
        ## plug in results for lam and mu to compute l,m,n, then combine the results into one dictionary:
        if S3 is not None:
            out.update(solve([self.invI3b.subs(out)-invI3(S3),self.invI4b.subs(out)-invI4(S3),self.invI5b.subs(out)-invI5(S3)],[self.l,self.m,self.n],dict=True)[0])
        self.reuss = out
        return out

    def hill_average(self):
        '''Computes the Hill averages of the Lame and Murnaghan (optional) constants using dictionaries of previously computed Voigt and Reuss averages.
        Requirements: run methods voigt_average() and reuss_average() first.'''
        voigt = self.voigt
        reuss = self.reuss
        out = {self.lam: (voigt[self.lam] + reuss[self.lam])/2}
        out.update({self.mu: (voigt[self.mu] + reuss[self.mu])/2})
        if len(reuss.keys())==5:
            out.update({self.l: (voigt[self.l] + reuss[self.l])/2})
            out.update({self.m: (voigt[self.m] + reuss[self.m])/2})
            out.update({self.n: (voigt[self.n] + reuss[self.n])/2})
        self.hill = out
        return out
    
    ############################# for improved method, see Lubarda 1997 and Blaschke 2017
    def improved_average(self,C2,C3=None):
        '''Compute an improved average for elastic constants of polycrystals whose single crystals are of cubic I symmetry.
        For the Lame constants, this function will use the self-consistent method of Hershey and Kroener.
        For the Murnaghan constants, a (far from perfect) generalization thereof is used following Blaschke 2017.'''
        C11 = C2[0,0,0,0]
        C12 = C2[0,0,1,1]
        lam = self.lam
        mu = self.mu
        hdenominator = (3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
        hfactor = (C11 + 2*C12 + 6*mu)*(C11 - C12 - 2*mu)/(3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
        Sh = sp.Symbol('Sh', real=True)
        ### Amat is (x_i x_j x_k x_l + y_i y_j y_k y_l + z_i z_j z_k z_l) where x,y,z are the unit vectors along the crystal axes
        ### one may easily check that in Voigt notation this construction leads to the diagonal matrix below
        Amat = np.diag([1,1,1,0,0,0])
        Hmat = Voigt(elasticC2(c12=Sh,c44=Sh+1/sp.S(2))) -5*Sh*Amat
        Hm = np.dot(Hmat,np.diag([1,1,1,2,2,2]))
        C2hat = UnVoigt(np.array(sp.factor(sp.expand(sp.Matrix(np.dot(Hm,np.dot(Voigt(C2),Hm.T)))).subs(Sh**2,0)).subs(Sh,hfactor/2)))
        ###
        tmplam = solve(self.invI1-sp.factor(invI1(C2hat)),lam,dict=True)[0][lam]
        tmpeqn = sp.factor(hdenominator*(self.invI2-sp.factor(invI2(C2hat))).subs(lam,tmplam))
        if C2.dtype==object:
            tmpmu = solve(tmpeqn,mu)
            print("WARNING: Kroener polynomial of improved averaging scheme for mu has 3 solutions;\
                  without numbers I do not know which one is positive and real. Returning all solutions for mu\
                  and returning the other elastic constants in terms of mu.")
        else:
            tmpmu = np.array(solve(tmpeqn,mu),dtype=complex)
            tmpmu = np.real(tmpmu[tmpmu>0])[0]
            tmplam = tmplam.subs(mu,tmpmu)
        out = {lam:tmplam, mu:tmpmu}
        
        if C3 is not None:
            C3hat = np.dot(Hm,np.dot(Hm,np.dot(Voigt(C3),Hm.T)))
            for i in range(6):
                C3hat[i] = np.array(((sp.expand(sp.Matrix(C3hat[i])).subs(Sh**2,0)).subs(Sh**3,0)).subs(Sh,hfactor/2))
            C3hat = UnVoigt(C3hat)
            if C2.dtype==object:
                out2 = {lam:tmplam, mu:mu}
                out.update(solve([(self.invI3-invI3(C3hat)).subs(out2),(self.invI4-invI4(C3hat)).subs(out2),(self.invI5-invI5(C3hat)).subs(out2)],[self.l,self.m,self.n],dict=True)[0])
            else:
                out.update(solve([(self.invI3-invI3(C3hat)).subs(out),(self.invI4-invI4(C3hat)).subs(out),(self.invI5-invI5(C3hat)).subs(out)],[self.l,self.m,self.n],dict=True)[0])
        
        self.improved = out
        return out

########### re-organize metal data within a class (so that we can perform calculations also on data read from input files, not just from metal_data dicts):
class metal_props:
    '''This class stores various metal properties; needed input sym must be one of 'iso', 'fcc', 'bcc', 'cubic', 'hcp', 'tetr', 'tetr2', 'trig', 'orth', 'mono', 'tric',
    and this will define which input parameters (such as elastic constants) are set/expected by the various initializing functions.
    In particular, symmetries 'iso'--'tetr' require individual elastic constants, such as .c44 and .c123, to be set, whereas lower symmetries ('tetr2'--'tric') require a
    list of all independent elastic constants to be stored in attribute .cij.
    Methods .init_C2() and .init_C3() (which are called by .init_all()) will compute the tensors of elastic constants and store them in Voigt notation as .C2 and .C3.
    Other attributes of this class are: temperature .T, density .rho, thermal expansion coefficient .alpha_a, and lattice constants .ac, .bc, .cc where the latter two are required only if they differ from .ac.
    The slip system to be studied is passed via Burgers vector length .burgers, unit Burgers vector .b, and slip plane normal .n0 in Cartesian coordinates,
    Additional optional attributes are polycrystal averages for Lame constants .mu and .lam (these are calculated via .init_all(), which calls .compute_Lame(), if not given).
    Finally, .init_sound() (called by .init_all()) computes the average transverse/longitudinal sound speeds of the polycrystal, .ct and .cl.
    Method .computesound(v) computes the sound speeds along vector v.'''
    def __init__(self,sym='iso', name='some_crystal'):
        self.sym=sym
        self.name = name
        self.filename = None ## if initialized from an inputfile, this attribute will store that file's name
        self.T=300
        self.Tm = None # melting temperature at low pressure
        self.ac = self.cc = self.bc = None ## lattice constants
        self.alphac = None ## angle between bc and cc, etc.; only needed for low symmetries like triclinic (None = determined from sym)
        self.betac = None
        self.gammac = None
        if self.sym not in ['tric']: self.alphac=np.pi/2
        if self.sym not in ['tric','mono']: self.betac=np.pi/2
        if self.sym not in ['tric','trig','hcp']: self.gammac=np.pi/2
        elif self.sym in ['trig','hcp']: self.gammac=2*np.pi/3 ## assume hexagonal lattice
        self.rho = 0
        self.c11=self.c12=self.c44=0
        self.c13=self.c33=self.c66=0
        self.cij=None ## list of 2nd order elastic constants for lower symmetries (trigonal/rhombohedral I, orthorhombic, monoclinic, and triclinic)
        self.cp = None
        self.Zener = None
        self.cijk=None ## list of 3rd order elastic constants for lower symmetries
        self.c111=self.c112=self.c123=0
        self.c144=self.c166=self.c456=0
        self.c113=self.c133=self.c155=0
        self.c222=self.c333=self.c344=self.c366=0
        self.Murl = self.Murm = self.Murn = None
        self.C2=np.zeros((6,6)) #tensor of SOEC in Voigt notation
        self.C3=np.zeros((6,6,6)) # tensor of TOEC in Voigt notation
        self.mu=self.lam=self.bulk=self.poisson=self.young=None ## polycrystal averages
        self.ct=self.cl=self.ct_over_cl=0 ## polycrystal averages
        self.Vc=0 ## unit cell volume
        self.qBZ=0 ## edge of Brillouin zone in isotropic approximation
        self.burgers=0 # length of Burgers vector
        self.b=np.zeros((3)) ## unit Burgers vector
        self.n0=np.zeros((3)) ## slip plane normal
        self.Millerb = None ## Miller indices for burgers vector
        self.Millern0 = None ## Miller indices for slip plane
        self.alpha_a=0 # thermal expansion coefficient at temperature self.T, set to 0 for constant rho calculations
        if sym=='tetr':
            self.c222=None
        elif sym=='hcp':
            self.c66=self.c166=self.c366=self.c456=None
        elif sym in ('fcc', 'bcc', 'cubic', 'iso'):
            self.c13=self.c33=self.c66=None
            self.c113=self.c133=self.c155=self.c222=self.c333=self.c344=self.c366=None
        if sym=='iso':
            self.c11=self.c111=self.c112=self.c166=None
            
    def __repr__(self):
        out = f" name:\t {self.name}\n sym:\t {self.sym}\n T:\t {self.T}\n ac:\t {self.ac}\n bc:\t {self.bc}\n cc:\t {self.cc}\n Vc:\t {self.Vc:.6e}\n rho:\t {self.rho}"
        if isinstance(self.rho, sp.Expr) or isinstance(self.ct, sp.Expr) or isinstance(self.cl, sp.Expr):
            out += "\n\t using sympy symbols"
        else:
            out += f"\n ct:\t {self.ct:.2f}\n cl:\t {self.cl:.2f}"
        return out
    
    def init_symbols(self):
        '''populates material density self.rho and elastic constants with sympy symbols'''
        poly = strain_poly(sym=self.sym)
        self.rho = sp.symbols(r'\rho',positive=True)
        if self.sym=='tetr':
            self.c66 = poly.C2[5,5]
        if self.sym in ('hcp','tetr'):
            self.c13 = poly.C2[0,2]
            self.c33 = poly.C2[2,2]
        if self.sym in ('fcc','bcc','cubic','hcp','tetr'):
            self.c11 = poly.C2[0,0]
        if self.sym in ('iso','fcc','bcc','cubic','hcp','tetr'):
            self.c12 = poly.C2[0,1]
            self.c44 = poly.C2[3,3]
        else:
            self.cij = poly.cij
            self.cijk = poly.cijk
        self.init_C2()
        self.C3 = poly.C3

    def init_C2(self):
        '''initializes the tensor of second order elastic constants'''
        if self.sym=='iso' and self.c12==0 and self.poisson is not None:
            self.c12 = self.c44*2/(1/self.poisson-2)
        self.C2=elasticC2(c11=self.c11,c12=self.c12,c13=self.c13,c33=self.c33,c44=self.c44,c66=self.c66,cij=self.cij,voigt=True)
        if self.sym in ('fcc', 'bcc', 'cubic'):
            self.cp = (self.c11-self.c12)/2
            self.Zener = self.c44/self.cp
        elif self.sym=='iso':
            self.cp = self.c44
            self.Zener = 1
    
    def init_C3(self):
        '''initializes the tensor of third order elastic constants'''
        self.C3=elasticC3(c111=self.c111,c112=self.c112,c113=self.c113,c123=self.c123,c133=self.c133,c144=self.c144,c155=self.c155,c166=self.c166,c222=self.c222,c333=self.c333,c344=self.c344,c366=self.c366,c456=self.c456,cijk=self.cijk,voigt=True)
    
    def compute_Lame(self, roundto=-8, include_TOEC=False, scheme='auto',simplify=True):
        '''Computes the Lame constants by averaging over the second order elastic constants.
           If option include_TOEC=True, Hill averages for the Murnaghan constants are calculated as well,
           but the user should be aware that the latter are not reliable as there is no good averaging scheme for TOECs.
           By default, the best available averaging scheme is used for SOECs, i.e. Kroener's method for cubic
           and Hill averages for all other crystals. However, keyword "scheme" (default: "auto") can be set to force
           one of "voigt", "reuss", "hill" averages to be used instead. Other common representations of SOECs are also
           determined from the averaged Lame constants, i.e. bulk and Young moduli as well as Poisson's ratio.
           Computed results are stored as class attributes and additionally we return the results as a pandas.Series;
           the latter includes additional representations of the TOEC (if include_TOEC) such as the Toupin/Bernstein
           constants nui and the standard repr. of cijk. Keyword "simplify" is only used if the single crystal
           elastic constants are sympy symbols, in which case it controls whether the results will be simplified.'''
        C2 = UnVoigt(self.C2)
        if include_TOEC:
            C3 = UnVoigt(self.C3)
            aver = IsoAverages(lam,mu,Murl,Murm,Murn)
        else:
            C3 = None
            aver = IsoAverages(lam,mu,0,0,0)
        if include_TOEC or self.sym not in ['iso','fcc','bcc', 'cubic'] or scheme!='auto':
            S2 = elasticS2(C2)
            if include_TOEC:
                S3 = elasticS3(S2,C3)
            else: S3 = None
            aver.voigt_average(C2,C3)
            if scheme != 'voigt':
                aver.reuss_average(S2,S3)
                HillAverage = aver.hill_average()
        if self.sym=='iso':
            self.lam=self.C2[0,1]
            self.mu=self.C2[3,3]
            self.Murl, self.Murm, self.Murn = tuple(convert_TOECiso(c123=self.C3[0,1,2],c144=self.C3[0,3,3],c456=self.C3[3,4,5]))[:3]
        elif scheme=='voigt':
            self.lam = aver.voigt[lam]
            self.mu = aver.voigt[mu]
        elif scheme=='reuss':
            self.lam = aver.reuss[lam]
            self.mu = aver.reuss[mu]
        elif self.sym in ('fcc', 'bcc', 'cubic') and scheme!='hill':
            ImprovedAv = aver.improved_average(C2,C3)
            self.lam = ImprovedAv[lam]
            self.mu = ImprovedAv[mu]
        else:
            ### use Hill average for Lame constants for non-cubic metals, as we do not have a better scheme at the moment
            self.lam = HillAverage[lam]
            self.mu = HillAverage[mu]
        if self.sym!='iso' and roundto is not None and not C2.dtype==object:
            self.lam = round(float(self.lam),roundto)
            self.mu = round(float(self.mu),roundto)
        if include_TOEC and self.sym != 'iso':
            if scheme=='voigt':
                self.Murl = aver.voigt[Murl]
                self.Murm = aver.voigt[Murm]
                self.Murn = aver.voigt[Murn]
            elif scheme=='reuss':
                self.Murl = aver.reuss[Murl]
                self.Murm = aver.reuss[Murm]
                self.Murn = aver.reuss[Murn]
            elif scheme=='blaschke':
                self.Murl = ImprovedAv[Murl]
                self.Murm = ImprovedAv[Murm]
                self.Murn = ImprovedAv[Murn]
            else:
                self.Murl = HillAverage[Murl]
                self.Murm = HillAverage[Murm]
                self.Murn = HillAverage[Murn]
            if roundto is not None and not C2.dtype==object:
                self.Murl = round(float(self.Murl),roundto)
                self.Murm = round(float(self.Murm),roundto)
                self.Murn = round(float(self.Murn),roundto)
        if C2.dtype==object and self.sym in ['cubic', 'fcc', 'bcc'] and scheme not in ['voigt','reuss','hill']:
            self.bulk,self.young,self.poisson = tuple(convert_SOECiso(c12=self.lam,c44=sp.symbols('mu')))[-3:]
        else:
            self.bulk,self.young,self.poisson = tuple(convert_SOECiso(c12=self.lam,c44=self.mu))[-3:]
        out = {"lambda":self.lam, "mu":self.mu, "bulk":self.bulk, "young":self.young, "poisson":self.poisson}
        if include_TOEC:
            nu1,nu2,nu3,c111,c112,c123,c144,c166,c456 = tuple(convert_TOECiso(l=self.Murl,m=self.Murm,n=self.Murn))[-9:]
            out |= {"l":self.Murl, "m":self.Murm, "n":self.Murn}
            out |= {"nu1":nu1, "nu2":nu2, "nu3":nu3}
            out |= {"c111":c111, "c112":c112, "c123":c123, "c144":c144, "c166":c166, "c456":c456}
        if simplify and self.sym!='iso' and C2.dtype==object:
            for key,val in out.items():
                if not isinstance(out[key], list):
                    out[key] = sp.simplify(val)
        return pd.Series(out)
    
    def init_sound(self):
        '''Computes the effective sound speeds of a polycrystal from its averaged Lame constants.'''
        if self.C2.dtype==object:
            self.ct = sp.sqrt(self.mu/self.rho)
            self.cl = sp.sqrt((self.lam+2*self.mu)/self.rho)
        else:
            self.ct = np.sqrt(self.mu/self.rho)
            self.cl = np.sqrt((self.lam+2*self.mu)/self.rho)
        self.ct_over_cl = self.ct/self.cl
        
    def init_qBZ(self):
        '''computes the radius of the first Brillouin zone for a sphere of equal volume as the unit cell
           after determining the latter'''
        if self.sym in ('iso', 'fcc', 'bcc', 'cubic'):
            self.Vc = self.ac**3
        elif self.sym=='hcp':
            self.Vc = self.ac*self.ac*self.cc*(3/2)*np.sqrt(3) ## 3*sin(pi/3)
        elif self.sym in ('tetr','tetr2'):
            self.Vc = self.ac*self.ac*self.cc
        elif self.sym=='orth': ## orthorhombic
            self.Vc = self.ac*self.bc*self.cc
        elif self.sym=='trig' and self.Vc<=0: ## trigonal I
            self.Vc = self.ac*self.ac*self.cc*np.sqrt(3)/2
            if abs(self.alphac-self.betac)<1e-15 and abs(self.alphac-self.gammac)<1e-15:
                self.Vc = self.ac*self.ac*self.cc*np.sqrt(1-3*np.cos(self.alphac)**2+2*np.cos(self.alphac)**3)
        elif self.Vc<=0 and self.sym=='mono':
            self.Vc = self.ac*self.bc*self.cc*np.sin(self.betac)
        elif self.Vc<=0 and self.sym=='tric':
            self.Vc = self.ac*self.bc*self.cc*np.sqrt(1 - np.cos(self.alphac)**2 - np.cos(self.betac)**2\
                 - np.cos(self.gammac)**2 + 2*np.cos(self.alphac)*np.cos(self.betac)*np.cos(self.gammac))
        self.qBZ = (6*np.pi**2/self.Vc)**(1/3)
    
    def init_all(self):
        '''call all other initializing functions for the data currently available'''
        self.init_C2()
        if self.c123!=0 or self.cijk is not None:
            self.init_C3()
        if self.mu is None:
            self.compute_Lame()
        else:
            self.bulk=self.lam + 2*self.mu/3
        self.init_sound()
        if self.ac is not None and self.ac>0: self.init_qBZ()
        
    def computesound(self,v,Miller=False,reziprocal=False):
        '''Computes the sound speeds of the crystal propagating in the direction of unit vector v.'''
        bt2 = sp.symbols('bt2')
        if Miller or len(v)==4:
            v = self.Miller_to_Cart(v,reziprocal=reziprocal)
        elif self.C2.dtype == np.dtype('O'): ## force purely analytical calculation if C2 contains non-floats
            v = np.asarray(v,dtype=object)
            v = v/sp.sqrt(np.dot(v, v))
        else:
            v = np.asarray(v)
            v = v/np.sqrt(np.dot(v, v))
        c44 = self.C2[3,3]
        thematrix = np.dot(v, np.dot(UnVoigt(self.C2/c44), v)) - bt2* np.diag([1,1,1]) ## compute normalized bt2 = v^2 /( c44/rho )
        thedet = sp.det(sp.Matrix(thematrix))
        solution = sp.solve(thedet,bt2)
        for i, si in enumerate(solution):
            si = si*c44/self.rho
            if si.free_symbols == set():
                solution[i] = np.sqrt(float(sp.re(si)))
            else:
                solution[i] = sp.sqrt(si)
        return solution
    
    def find_wavespeed(self,which='l',verbose=False,accuracy=1e-4,maxfun=1000):
        '''For keyword which in ["l","h","hs"], this method determines the [lowest, highest, or highest (quasi-)shear] wave speed
           in the crystal. We use scipy's optimize.direct() algorithm to search over all directions in the crystal (2 angles).
           Keywords accuracy and maxfun are passed to optimize.direct, where f_min_rtol=accuracy=len_tol and  vol_tol=accuracy**2.
           See the docs of optimize.direct() for further information on those parameters (which we reduce here to speed up
           the calculation).'''
        if which=='l':
            if self.sym=='iso':
                return self.ct
            def select(x):
                return min(x)
        elif which=='h':
            if self.sym=='iso':
                return self.cl
            def select(x):
                return 1/max(x)
        elif which=='hs':
            if self.sym=='iso':
                return self.ct
            def select(x):
                if len(x)<3:
                    return 1/min(x)
                return 1/max(np.sort(x)[:2])
        else:
            raise ValueError("keyword which must be one of ['l','h','hs']")
        def f(x):
            v = [np.sin(x[0])*np.cos(x[1]),np.sin(x[0])*np.sin(x[1]),np.cos(x[0])]
            return select(self.computesound(v))
        bounds = [(0,2*np.pi),(0,2*np.pi)]
        result = optimize.direct(f,bounds,f_min_rtol=accuracy,maxfun=maxfun,vol_tol=accuracy**2,len_tol=accuracy)
        if not result.success or verbose:
            print(result)
            x = result.x
            v = [np.sin(x[0])*np.cos(x[1]),np.sin(x[0])*np.sin(x[1]),np.cos(x[0])]
            print(f"{v=}")
        if which =='l':
            return result.fun
        return 1/result.fun
    
    def anisotropy_index(self):
        '''Computes a number quantifying the anisotropy of a crystal following the
           recommendation of Kube 2016. In particular, we compute a measure of the
           difference between Voigt and Reuss averages of shear and bulk modulus, 
           known also as the universal log-Euclidean anisotropy index:
           A_L = sqrt([ln(B_V/B_R)]^2 + 5*[ln(G_V/G_R)]^2), see AIP Advances 6, 095209 (2016).'''
        C2 = UnVoigt(self.C2)
        aver = IsoAverages(lam,mu,0,0,0)
        aver.voigt_average(C2)
        S2 = elasticS2(C2)
        aver.reuss_average(S2)
        muV = aver.voigt[mu]
        muR = aver.reuss[mu]
        bulkV = aver.voigt[lam] + 2*muV/3
        bulkR = aver.reuss[lam] + 2*muR/3
        if C2.dtype==object:
            out = sp.sqrt((sp.log(bulkV/bulkR))**2 + 5*(sp.log(muV/muR))**2)
        else:
            out = float(np.round(np.sqrt((np.log(float(bulkV/bulkR)))**2 + 5*(np.log(float(muV/muR)))**2),12))
        return out
    
    def Miller_to_Cart(self,v,normalize=True,reziprocal=False):
        '''Converts vector v from Miller indices to Cartesian coordinates (very small numbers are rounded to 0). If normalize=True, a unit vector is returned.
        See Luscher et al., Modelling Simul. Mater. Sci. Eng. 22 (2014) 075008 for details on the method.
        By default, this function expects real space Miller indices, set reziprocal=True for reziprocal space.
        Warning: all lattice constants are assumed a=b=c=1 unless explicitly set by the user.'''
        if self.ac is None or self.ac==0:
            a=1
        else: a=self.ac
        if self.bc is None or self.bc==0:
            b=a
        else: b=self.bc
        if self.cc is None or self.cc==0:
            c=a
        else:c=self.cc
        if isinstance(a+b+c, sp.Expr):
            v = np.asarray(v)
            alphac = int(round(self.alphac*180/np.pi))
            betac = int(round(self.betac*180/np.pi))
            gammac = int(round(self.gammac*180/np.pi))
            return Miller_to_Cart(v,lattice=((a,b,c),(alphac,betac,gammac)),normalize=normalize,reziprocal=reziprocal)
        v = np.asarray(v).astype(dtype=float)
        d = c*(np.cos(self.alphac)-np.cos(self.gammac)*np.cos(self.betac))/np.sin(self.gammac)
        T = np.array([[a,b*np.cos(self.gammac),c*np.cos(self.betac)],
                      [0,b*np.sin(self.gammac),d],
                      [0,0,np.sqrt((c*np.sin(self.betac))**2-d**2)]])
        if reziprocal:
            ## real space basis vectors a_i = T[:,i]
            V = np.dot(np.cross(T[:,0],T[:,1]),T[:,2])
            R = np.empty(T.shape)
            R[:,0] = np.cross(T[:,1],T[:,2])/V
            R[:,1] = np.cross(T[:,2],T[:,0])/V
            R[:,2] = np.cross(T[:,0],T[:,1])/V
            if len(v)==4 and abs(v[0]+v[1]+v[2])<1e-12: v = [v[0]+v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
            out = np.dot(R,v)
        else:
            if len(v)==4 and abs(v[0]+v[1]+v[2])<1e-12: v = [v[0]-v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
            out = np.dot(T,v)
        accuracy = 1e-15
        if normalize:
            if self.sym in ['cubic', 'fcc', 'bcc']:
                out = v # minimize rounding errors
            out = out/np.sqrt(np.dot(out,out))
        else:
            accuracy *= self.ac
        for i,outi in enumerate(out):
            if abs(outi)<accuracy:
                out[i] = round(outi) # round tiny numbers to zero
        return out
        
    def populate_from_dict(self,inputparams):
        '''Assigns values to various attributes of this class by reading a dictionary 'inputparams'. Keywords unknown to this function are ignored.'''
        keys = inputparams.keys()
        sym = self.sym
        self.name = inputparams.get('name',self.name)
        if 'T' in keys:
            self.T=float(inputparams['T'])
        if 'Tm' in keys:
            self.Tm=float(inputparams['Tm'])
        if 'lam' in keys and 'mu' in keys:
            self.lam=float(inputparams['lam'])
            self.mu=float(inputparams['mu'])
            self.bulk,self.young,self.poisson = tuple(convert_SOECiso(c12=self.lam,c44=self.mu))[-3:]
        self.ac=float(inputparams['a'])
        self.rho = float(inputparams['rho'])
        if 'c11' in keys and sym != 'iso':
            self.c11 = float(inputparams['c11'])
        if 'c12' in keys:
            self.c12 = float(inputparams['c12'])
        if 'c44' in keys:
            self.c44 = float(inputparams['c44'])
        if sym in ('hcp', 'tetr'):
            self.cc = float(inputparams['c'])
            self.c13 = float(inputparams['c13'])
            self.c33 = float(inputparams['c33'])
            if 'c113' in keys:
                self.c113 = float(inputparams['c113'])
                self.c133 = float(inputparams['c133'])
                self.c155 = float(inputparams['c155'])
                self.c333 = float(inputparams['c333'])
                self.c344 = float(inputparams['c344'])
        if sym=='tetr':
            self.c66 = float(inputparams['c66'])
            if 'c366' in keys:
                self.c366 = float(inputparams['c366'])
        if sym not in ['iso', 'fcc', 'bcc', 'cubic', 'hcp', 'tetr']: ## support for other/lower crystal symmetries
            self.cc = float(inputparams['c'])
            if 'lcb' in keys:
                self.bc = float(inputparams['lcb'])
            if 'Vc' in keys: ## TODO: init_qBZ() will currently overwrite this by auto-computed Vc; do we want to support a manual override?
                self.Vc = float(inputparams['Vc'])
            self.cij = np.asarray(inputparams['cij'].split(','),dtype=float) ## expect a list of cij in ascending order
            if 'cijk' in keys:
                self.cijk = np.asarray(inputparams['cijk'].split(','),dtype=float) ## expect a list of cijk in ascending order
        if 'c111' in keys and sym != 'iso':
            self.c111 = float(inputparams['c111'])
            self.c112 = float(inputparams['c112'])
        if 'c123' in keys:
            self.c123 = float(inputparams['c123'])
            self.c144 = float(inputparams['c144'])
            if sym not in ('hcp', 'iso'):
                self.c166 = float(inputparams['c166'])
            if sym !='hcp':
                self.c456 = float(inputparams['c456'])
            if sym =='hcp':
                self.c222 = float(inputparams['c222'])
        if 'alpha' in keys:
            self.alphac=float(inputparams['alpha'])*np.pi/180 ## convert degrees to radians
        if 'beta' in keys:
            self.betac=float(inputparams['beta'])*np.pi/180
        if 'gamma' in keys:
            self.gammac=float(inputparams['gamma'])*np.pi/180
        if 'Millerb' in keys:
            self.Millerb = str_to_array(inputparams['Millerb'])
            if 'burgers' in keys:
                self.burgers=float(inputparams['burgers'])
            else:
                burg = self.Miller_to_Cart(self.Millerb,normalize=False)
                self.burgers=np.sqrt(burg @ burg)
            self.b = self.Miller_to_Cart(self.Millerb)
        elif 'b' in keys:
            self.b=np.asarray(inputparams['b'].split(','),dtype=float)
            if 'burgers' in keys:
                self.burgers=float(inputparams['burgers'])
            else:
                self.burgers=np.sqrt(self.b @ self.b)
            self.b = self.b/np.sqrt(np.dot(self.b,self.b))
        if 'Millern0' in keys:
            self.Millern0 = str_to_array(inputparams['Millern0'])
            self.n0 = self.Miller_to_Cart(self.Millern0,reziprocal=True)
        elif 'n0' in keys:
            self.n0=np.asarray(inputparams['n0'].split(','),dtype=float)
            self.n0 = self.n0/np.sqrt(np.dot(self.n0,self.n0))
        if abs(np.dot(self.b,self.n0))>1e-15:
            print("ERROR: Burgers vector does not lie in the slip plane; .b and .n0 must be normal to each other!")
        if self.burgers<0 or self.burgers > 100*self.ac:
            print("ERROR: Burgers vector length is much larger than lattice constant a, check units!")
        if 'alpha_a' in keys:
            self.alpha_a = float(inputparams['alpha_a'])

def readinputfile(fname,init=True):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the metal_props class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.'''
    inputparams = loadinputfile(fname)
    sym = inputparams['sym']
    name = inputparams.get('name',str(fname))
    out = metal_props(sym,name)
    out.populate_from_dict(inputparams)
    out.filename = fname ## remember which file we read
    if init:
        out.init_all()
    return out

def Miller_to_Cart(v,lattice=None,normalize=True,reziprocal=False):
    '''Converts vector v from Miller indices to Cartesian coordinates. If normalize=True, a unit vector is returned.
    See Luscher et al., Modelling Simul. Mater. Sci. Eng. 22 (2014) 075008 for details on the method.
    By default, this function expects real space Miller indices, set reziprocal=True for reziprocal space.
    Keyword "lattice", if set, must have the format ((a,b,c),(alpha,beta,gamma)), where a,b,c are lattice vector lengths
    and alpha,beta,gamma, are angles in units of degrees (not radians).
    Warning: all lattice constants are assumed a=b=c=1 and normal to each other (cubic symmetry) unless explicitly
    set by the user.'''
    v = list(v)
    sym = None
    if lattice is None:
        a=b=c=1
        alphac=betac=gammac=sp.pi/sp.S(2)
        sym = 'cubic'
    else:
        a,b,c = lattice[0]
        alphac,betac,gammac = lattice[1]
        alphac *= sp.pi/sp.S(180)
        betac *= sp.pi/sp.S(180)
        gammac *= sp.pi/sp.S(180)
    d = c*(sp.cos(alphac)-sp.cos(gammac)*sp.cos(betac))/sp.sin(gammac)
    T = [sp.Matrix([a,b*sp.cos(gammac),c*sp.cos(betac)]),
         sp.Matrix([0,b*sp.sin(gammac),d]),
         sp.Matrix([0,0,sp.sqrt((c*sp.sin(betac))**2-d**2)])]
    # TT = [sp.Matrix([a,0,0]),
    #      sp.Matrix([b*sp.cos(gammac),b*sp.sin(gammac),0]),
    #      sp.Matrix([c*sp.cos(betac),d,sp.sqrt((c*sp.sin(betac))**2-d**2)])]
    hcpsum = v[0]+v[1]+v[2]
    checkhcp = (len(v)==4 and ((not isinstance(hcpsum, sp.Expr) or len(hcpsum.free_symbols)==0) and abs(v[0]+v[1]+v[2])<1e-12))
    if reziprocal:
        ## real space basis vectors a_i = T[i]
        V = (T[0].cross(T[1])).T
        V = V.dot(T[2])
        R = [T[1].cross(T[2])/V, T[2].cross(T[0])/V, T[0].cross(T[1])/V]
        if checkhcp:
            v = [v[0]+v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
        out = sp.Matrix([list(Ri) for Ri in R])
        out = out @ sp.Matrix(v)
    else:
        if checkhcp:
            v = [v[0]-v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
        out = sp.Matrix([list(Ti) for Ti in T]) @ sp.Matrix(v)
    if normalize:
        if sym=='cubic':
            out = sp.Matrix(v) # minimize rounding errors
        norm = out.T @ out
        if len(norm)==1:
            norm = norm[0]
        out = out/sp.sqrt(norm)
    return [sp.simplify(outi) for outi in out]
