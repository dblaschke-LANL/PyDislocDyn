# Compute averages of elastic constants for polycrystals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - July 15, 2021
'''This module defines the metal_props class which is one of the parents of the Dislocation class defined in linetension_calcs.py.
   Additional classes available in this module are IsoInvariants and IsoAverages which inherits from the former and is used to
   calculate averages of elastic constants. We also define a function, readinputfile, which reads a PyDislocDyn input file and
   returns an instance of the metal_props class.
   If run as a script, this file will compute polycrystal averages of second and third order elastic constants, either for
   all metals predefined in metal_data.py, or for those input files passed as arguments to the script; results are written
   to a text file 'averaged_elastic_constants.txt'.'''
#################################
import sys
from sympy.solvers import solve 
import sympy as sp
import numpy as np
## workaround for spyder's runfile() command when cwd is somewhere else:
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
##
from elasticconstants import elasticC2, elasticC3, elasticS2, elasticS3, Voigt, UnVoigt
import metal_data as data

### generate a list of those fcc, bcc, hcp metals for which we have sufficient data
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(data.CRC_c11.keys())))
### compute various contractions/invariants of elastic constant/compliance tensors:
def invI1(C2):
    '''Computes the trace C_iijj of SOEC.'''
    return np.trace(np.trace(C2))
    ### einsum is slightly faster, but works only with numbers, not symbols
    
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
    
### define some symbols and functions of symbols for the effective isotropic side of our calculations
lam, mu = sp.symbols('lam mu') ## Symbols for Lame constants
Murl, Murm, Murn = sp.symbols('Murl Murm Murn') ## Symbols for Murnaghan constants

class IsoInvariants(object):
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
    
    def voigt_average(self,C2,C3=None):
        '''Computes the Voigt averages of elastic constants using the tensor(s) C2 and C3 of SOEC and TOEC (optional). The output is a dictionary of average values for the Lame and Murnaghan constants.
        If C3 is omitted (or 'None'), only the average Lame constants are computed.'''
        out = solve([self.invI1-invI1(C2),self.invI2-invI2(C2)],[lam,mu],dict=True)[0]
        ## combine all five Voigt-averaged answers into one dictionary for each metal (i.e. Voigt is a dictionary of dictionaries)
        if np.asarray(C3).all()!=None:
            out.update(solve([self.invI3-invI3(C3),self.invI4-invI4(C3),self.invI5-invI5(C3)],[Murl,Murm,Murn],dict=True)[0])
        self.voigt = out
        return out

    def reuss_average(self,S2,S3=None):
        '''Computes the Reuss averages of elastic constants using the second and third (optional) order compliances tensor(s) S2 and S3. The output is a dictionary of average values for the Lame and Murnaghan constants.
        If S3 is omitted (or 'None'), only the average Lame constants are computed.'''
        out = solve([self.invI1b-invI1(S2),self.invI2b-invI2(S2)],[lam,mu],dict=True)[0]
        ## plug in results for lam and mu to compute l,m,n, then combine the results into one dictionary:
        if np.asarray(S3).all()!=None:
            out.update(solve([self.invI3b.subs(out)-invI3(S3),self.invI4b.subs(out)-invI4(S3),self.invI5b.subs(out)-invI5(S3)],[Murl,Murm,Murn],dict=True)[0])
        self.reuss = out
        return out

    def hill_average(self):
        '''Computes the Hill averages of the Lame and Murnaghan (optional) constants using dictionaries of previously computed Voigt and Reuss averages.
        Requirements: run methods voigt_average() and reuss_average() first.'''
        voigt = self.voigt
        reuss = self.reuss
        out = {lam : (voigt[lam] + reuss[lam])/2}
        out.update({mu : (voigt[mu] + reuss[mu])/2})
        if len(reuss.keys())==5:
            out.update({Murl : (voigt[Murl] + reuss[Murl])/2})
            out.update({Murm : (voigt[Murm] + reuss[Murm])/2})
            out.update({Murn : (voigt[Murn] + reuss[Murn])/2})
        self.hill = out
        return out
    
    ############################# for improved method, see Lubarda 1997 and Blaschke 2017
    def improved_average(self,C2,C3=None):
        '''Compute an improved average for elastic constants of polycrystals whose single crystals are of cubic I symmetry.
        For the Lame constants, this function will use the self-consistent method of Hershey and Kroener.
        For the Murnaghan constants, a (far from perfect) generalization thereof is used following Blaschke 2017.'''
        C11 = C2[0,0,0,0]
        C12 = C2[0,0,1,1]
        hdenominator = (3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
        hfactor = (C11 + 2*C12 + 6*mu)*(C11 - C12 - 2*mu)/(3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
        Sh = sp.Symbol('Sh', real=True)
        ### Amat is (x_i x_j x_k x_l + y_i y_j y_k y_l + z_i z_j z_k z_l) where x,y,z are the unit vectors along the crystal axes
        ### one may easily check that in Voigt notation this construction leads to the diagonal matrix below
        Amat = np.diag([1,1,1,0,0,0])
        Hmat = Voigt(elasticC2(c12=Sh,c44=Sh+1/2)) -5*Sh*Amat
        Hm = np.dot(Hmat,np.diag([1,1,1,2,2,2]))
        C2hat = UnVoigt(np.array(sp.factor(sp.expand(sp.Matrix(np.dot(Hm,np.dot(Voigt(C2),Hm.T)))).subs(Sh**2,0)).subs(Sh,hfactor/2)))
        ###
        tmplam = solve(self.invI1-sp.factor(invI1(C2hat)),lam,dict=True)[0][lam]
        tmpeqn = sp.factor(hdenominator*(self.invI2-sp.factor(invI2(C2hat))).subs(lam,tmplam))
        tmpmu = np.array(solve(tmpeqn,mu),dtype=complex)
        tmpmu = np.real(tmpmu[tmpmu>0])[0]
        tmplam = tmplam.subs(mu,tmpmu)
        out = {lam:tmplam, mu:tmpmu}
        
        if np.asarray(C3).all()!=None:
            C3hat = np.dot(Hm,np.dot(Hm,np.dot(Voigt(C3),Hm.T)))
            for i in range(6):
                C3hat[i] = np.array(((sp.expand(sp.Matrix(C3hat[i])).subs(Sh**2,0)).subs(Sh**3,0)).subs(Sh,hfactor/2))
            C3hat = UnVoigt(C3hat)
            out.update(solve([(self.invI3-invI3(C3hat)).subs(out),(self.invI4-invI4(C3hat)).subs(out),(self.invI5-invI5(C3hat)).subs(out)],[Murl,Murm,Murn],dict=True)[0])
        
        self.improved = out
        return out

########### re-organize metal data within a class (so that we can perform calculations also on data read from input files, not just from metal_data dicts):
class metal_props:
    '''This class stores various metal properties; needed input sym must be one of 'iso', 'fcc', 'bcc', 'hcp', 'tetr', 'trig', 'orth', 'mono', 'tric',
    and this will define which input parameters (such as elastic constants) are set/expected by the various initializing functions.
    In particular, symmetries 'iso'--'tetr' require individual elastic constants, such as .c44 and .c123, to be set, whereas lower symmetries require a
    list of all independent elastic constants to be stored in attribute .cij.
    Methods .init_C2() and .init_C3() (which are called by .init_all()) will compute the tensors of elastic constants and store them in Voigt notation as .C2 and .C3.
    Other attributes of this class are: temperature .T, density .rho, thermal expansion coefficient .alpha_a, and lattice constants .ac, .bc, .cc where the latter two are required only if they differ from .ac.
    For monoclinic and triclinic crystals, the unit cell volume .Vc must also be given.
    The slip system to be studied is passed via Burgers vector length .burgers, unit Burgers vector .b, and slip plane normal .n0 in Cartesian coordinates,
    Additional optional attributes are polycrystal averages for Lame constants .mu and .lam (these are calculated via .init_all(), which calls .compute_Lame(), if not given), 
    as well as critical velocities .vcrit_smallest, .vcrit_screw, .vcrit_edge. Finally, .init_sound() (called by .init_all()) computes the average transverse/longitudinal sound speeds of the polycrystal, .ct and .cl.
    Method .computesound(v) computes the sound speeds along vector v.'''
    def __init__(self,sym='iso', name='some_crystal'):
        self.sym=sym
        self.name = name
        self.T=300
        self.ac = self.cc = self.bc = None ## lattice constants
        self.alphac = None ## angle between bc and cc, etc.; only needed for low symmetries like triclinic (None = determined from sym)
        self.betac = None
        self.gammac = None
        if self.sym not in ['tric']: self.alphac=np.pi/2
        if self.sym not in ['tric','mono']: self.betac=np.pi/2
        if self.sym not in ['tric','trig','hcp']: self.gammac=np.pi/2
        elif self.sym in ['trig','hcp']: self.gammac=2*np.pi/3
        self.rho = 0
        self.c11=self.c12=self.c44=0
        self.c13=self.c33=self.c66=0
        self.cij=None ## list of 2nd order elastic constants for lower symmetries (trigonal/rhombohedral I, orthorhombic, monoclinic, and triclinic)
        self.cijk=None ## list of 3rd order elastic constants for lower symmetries
        self.c111=self.c112=self.c123=0
        self.c144=self.c166=self.c456=0
        self.c113=self.c133=self.c155=0
        self.c222=self.c333=self.c344=self.c366=0
        self.C2=np.zeros((6,6)) #tensor of SOEC in Voigt notation
        self.C3=np.zeros((6,6,6)) # tensor of TOEC in Voigt notation
        self.mu=self.lam=self.bulk=self.poisson=self.young=None ## polycrystal averages
        self.ct=self.cl=self.ct_over_cl=0 ## polycrystal averages
        self.Vc=0 ## unit cell volume
        self.qBZ=0 ## edge of Brillouin zone in isotropic approximation
        self.burgers=0 # length of Burgers vector
        self.b=np.zeros((3)) ## unit Burgers vector 
        self.n0=np.zeros((3)) ## slip plane normal
        self.vcrit_smallest=self.vcrit_screw=self.vcrit_edge=None
        self.alpha_a=0 # thermal expansion coefficient at temperature self.T, set to 0 for constant rho calculations
        if sym=='tetr':
            self.c222=None
        elif sym=='hcp':
            self.c66=self.c166=self.c366=self.c456=None
        elif sym=='fcc' or sym=='bcc'or sym=='iso':
            self.c13=self.c33=self.c66=None
            self.c113=self.c133=self.c155=self.c222=self.c333=self.c344=self.c366=None
        if sym=='iso':
            self.c11=self.c111=self.c112=self.c166=None
            
    def __repr__(self):
        return  "{}".format({'name':self.name, 'sym':self.sym, 'T':self.T, 'ac':self.ac, 'bc':self.bc, 'cc':self.cc, 'Vc':self.Vc, 'rho':self.rho, 'ct':self.ct, 'cl':self.cl})
            
    def init_C2(self):
        '''initializes the tensor of second order elastic constants'''
        self.C2=elasticC2(c11=self.c11,c12=self.c12,c13=self.c13,c33=self.c33,c44=self.c44,c66=self.c66,cij=self.cij,voigt=True)
        if self.cij is not None:
            self.c44 = self.C2[3,3] ## some legacy code expect these to be set
            self.c11 = self.C2[0,0]
            self.c12 = self.C2[0,1]
    
    def init_C3(self):
        '''initializes the tensor of third order elastic constants'''
        self.C3=elasticC3(c111=self.c111,c112=self.c112,c113=self.c113,c123=self.c123,c133=self.c133,c144=self.c144,c155=self.c155,c166=self.c166,c222=self.c222,c333=self.c333,c344=self.c344,c366=self.c366,c456=self.c456,cijk=self.cijk,voigt=True)
        if self.cijk is not None:
            self.c123 = self.C3[0,1,2] ## some legacy code expect these to be set
    
    def compute_Lame(self, roundto=-8):
        '''computes the Lame constants by averaging over the second order elastic constants'''
        aver = IsoAverages(lam,mu,0,0,0) # don't need Murnaghan constants
        C2 = UnVoigt(self.C2)
        if self.sym=='iso':
            self.lam=self.c12
            self.mu=self.c44
        elif self.sym=='fcc' or self.sym=='bcc':
            ImprovedAv = aver.improved_average(C2)
            self.lam = round(float(ImprovedAv[lam]),roundto)
            self.mu = round(float(ImprovedAv[mu]),roundto)
        else:
            S2 = elasticS2(C2)
            aver.voigt_average(C2)
            aver.reuss_average(S2)
            HillAverage = aver.hill_average()
            ### use Hill average for Lame constants for non-cubic metals, as we do not have a better scheme at the moment
            self.lam = round(float(HillAverage[lam]),roundto)
            self.mu = round(float(HillAverage[mu]),roundto)
        self.bulk = self.lam + 2*self.mu/3
        self.poisson = self.lam/(2*(self.lam+self.mu)) ## average Poisson ratio nu
        self.young = 2*self.mu*(1+self.poisson)
    
    def init_sound(self):
        '''Computes the effective sound speeds of a polycrystal from its averaged Lame constants.'''
        self.ct = np.sqrt(self.mu/self.rho)
        self.cl = np.sqrt((self.lam+2*self.mu)/self.rho)
        self.ct_over_cl = self.ct/self.cl
        
    def init_qBZ(self):
        '''computes the radius of the first Brillouin zone for a sphere of equal volume as the unit cell
           after determining the latter'''
        if self.sym=='iso' or self.sym=='fcc' or self.sym=='bcc':
            self.Vc = self.ac**3
        elif self.sym=='hcp':
            self.Vc = self.ac*self.ac*self.cc*(3/2)*np.sqrt(3) ## 3*sin(pi/3)
        elif self.sym=='tetr':
            self.Vc = self.ac*self.ac*self.cc
        elif self.sym=='orth': ## orthorhombic
            self.Vc = self.ac*self.bc*self.cc
        elif self.sym=='trig': ## trigonal/rhombohedral I
            self.Vc = self.ac*self.ac*self.cc*np.sqrt(3)/2
        elif self.Vc<=0 and self.sym in ['mono', 'tric']:
            raise ValueError("need unit cell volume Vc")
        self.qBZ = ((6*np.pi**2/self.Vc)**(1/3))
    
    def init_all(self):
        '''call all other initializing functions for the data currently available'''
        self.init_C2()
        if self.c123!=0 or self.cijk is not None:
            self.init_C3()
        if self.mu==None:
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
        else:
            v = np.asarray(v)
            v = v/np.sqrt(np.dot(v , v))
        c44 = self.C2[3,3]
        thematrix = np.dot(v , np.dot(UnVoigt(self.C2/c44) , v)) - bt2* np.diag([1,1,1]) ## compute normalized bt2 = v^2 /( c44/rho )
        thedet = sp.det(sp.Matrix(thematrix))
        solution = sp.solve(thedet,bt2)
        for i in range(len(solution)):
            solution[i] = solution[i]  * c44/self.rho
            if solution[i].free_symbols == set():
                solution[i] = np.sqrt(float(sp.re(solution[i])))
            else:
                solution[i] = sp.sqrt(solution[i])
        return solution
    
    def Miller_to_Cart(self,v,normalize=True,reziprocal=False,accuracy=15):
        '''Converts vector v from Miller indices to Cartesian coordinates rounded to 'accuracy' digits (15 by default). If normalize=True, a unit vector is returned.
        See Luscher et al., Modelling Simul. Mater. Sci. Eng. 22 (2014) 075008 for details on the method.
        By default, this function expects real space Miller indices, set reziprocal=True for reziprocal space.'''
        if self.ac==None or self.ac==0: a=1
        else: a=self.ac
        if self.bc==None or self.bc==0: b=a
        else: b=self.bc
        if self.cc==None or self.cc==0 : c=a
        else:c=self.cc
        d = c*(np.cos(self.alphac)-np.cos(self.gammac)*np.cos(self.betac))/np.sin(self.gammac)
        T = np.array([[a,b*np.cos(self.gammac),c*np.cos(self.betac)],\
                      [0,b*np.sin(self.gammac),d],\
                      [0,0,np.sqrt((c*np.sin(self.betac))**2-d**2)]])
        if reziprocal:
            ## real space basis vectors a_i = T[:,i]
            V = np.dot(np.cross(T[:,0],T[:,1]),T[:,2])
            R = np.empty(T.shape)
            R[:,0] = np.cross(T[:,1],T[:,2])/V
            R[:,1] = np.cross(T[:,2],T[:,0])/V
            R[:,2] = np.cross(T[:,0],T[:,1])/V
            if len(v)==4 and abs(v[0]+v[1]+v[2])<1e-15: v = [v[0]+v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
            out = np.dot(R,v)
        else:
            if len(v)==4 and abs(v[0]+v[1]+v[2])<1e-15: v = [v[0]-v[2],v[1]-v[2],v[3]] ## convert from 4 to 3 indices
            out = np.dot(T,v)
        if normalize:
            out = out/np.sqrt(np.dot(out,out))
        return np.round(out,accuracy)
        
    def populate_from_dict(self,inputparams):
        '''Assigns values to various attributes of this class by reading a dictionary 'inputparams'. Keywords unknown to this function are ignored.'''
        keys = inputparams.keys()
        sym = self.sym
        if 'name' in keys:
            self.name = inputparams['name']
        if 'T' in keys:
            self.T=float(inputparams['T'])
        self.burgers=float(inputparams['burgers'])
        if 'lam' in keys and 'mu' in keys:
            self.lam=float(inputparams['lam'])
            self.mu=float(inputparams['mu'])
            self.bulk = self.lam + 2*self.mu/3
            self.poisson = self.lam/(2*(self.lam+self.mu)) ## average Poisson ratio nu
            self.young = 2*self.mu*(1+self.poisson)
        self.ac=float(inputparams['a'])
        self.rho = float(inputparams['rho'])
        if 'c11' in inputparams.keys():
            self.c11 = float(inputparams['c11'])
        if 'c12' in inputparams.keys():
            self.c12 = float(inputparams['c12'])
        if 'c44' in inputparams.keys():
            self.c44 = float(inputparams['c44'])
        if sym=='hcp' or sym=='tetr':
            self.cc = float(inputparams['c'])
            self.c13 = float(inputparams['c13'])
            self.c33 = float(inputparams['c33'])
            if 'c113' in inputparams.keys():
                self.c113 = float(inputparams['c113'])
                self.c133 = float(inputparams['c133'])
                self.c155 = float(inputparams['c155'])
                self.c333 = float(inputparams['c333'])
                self.c344 = float(inputparams['c344'])
        if sym=='tetr':
            self.c66 = float(inputparams['c66'])
            if 'c366' in inputparams.keys():
                self.c366 = float(inputparams['c366'])
        if sym not in ['iso', 'fcc', 'bcc', 'hcp', 'tetr']: ## support for other/lower crystal symmetries
            self.cc = float(inputparams['c'])
            if 'lcb' in inputparams.keys():
                self.bc = float(inputparams['lcb'])
            if 'Vc' in inputparams.keys():
                self.Vc = float(inputparams['Vc'])
            self.cij = np.asarray(inputparams['cij'].split(','),dtype=float) ## expect a list of cij in ascending order
            if 'cijk' in inputparams.keys():
                self.cijk = np.asarray(inputparams['cijk'].split(','),dtype=float) ## expect a list of cijk in ascending order
        if 'c111' in inputparams.keys() and sym != 'iso':
            self.c111 = float(inputparams['c111'])
            self.c112 = float(inputparams['c112'])
        if 'c123' in inputparams.keys():
            self.c123 = float(inputparams['c123'])
            self.c144 = float(inputparams['c144'])
            if sym != 'hcp' and sym != 'iso':
                self.c166 = float(inputparams['c166'])
            if sym !='hcp':
                self.c456 = float(inputparams['c456'])
            if sym =='hcp':
                self.c222 = float(inputparams['c222'])
        if 'alpha' in keys: self.alphac=inputparams['alpha']
        if 'beta' in keys: self.betac=inputparams['beta']
        if 'gamma' in keys: self.gammac=inputparams['gamma']
        if 'Millerb' in keys:
            self.Millerb = np.asarray(inputparams['Millerb'].split(','),dtype=float)
            self.b = self.Miller_to_Cart(self.Millerb)
        else:
            self.b=np.asarray(inputparams['b'].split(','),dtype=float)
            self.b = self.b/np.sqrt(np.dot(self.b,self.b))
        if 'Millern0' in keys:
            self.Millern0 = np.asarray(inputparams['Millern0'].split(','),dtype=float)
            self.n0 = self.Miller_to_Cart(self.Millern0,reziprocal=True)
        else:
            self.n0=np.asarray(inputparams['n0'].split(','),dtype=float)
            self.n0 = self.n0/np.sqrt(np.dot(self.n0,self.n0))
        if abs(np.dot(self.b,self.n0))>1e-15:
            print("ERROR: Burgers vector does not lie in the slip plane; .b and .n0 must be normal to each other!")
        if 'vcrit_smallest' in keys:
            self.vcrit_smallest = float(inputparams['vcrit_smallest'])
        if 'vcrit_screw' in keys:
            self.vcrit_screw = float(inputparams['vcrit_screw'])
        if 'vcrit_edge' in keys:
            self.vcrit_edge = float(inputparams['vcrit_edge'])
        if 'alpha_a' in keys:
            self.alpha_a = float(inputparams['alpha_a'])
        
def loadinputfile(fname):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py and returns its data as a dictionary.'''
    inputparams={}
    with open(fname,"r") as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            if "#" != line[0]:
                currentline = line.lstrip().rstrip().split()
                if len(currentline) > 2:
                    key  = currentline[0]
                    if len(currentline)==3 or currentline[3]=='#':
                        value = currentline[2]
                        if value[-1] == '#':
                            value = value[:-1]
                    else:
                        value = currentline[2]
                        for i in range(len(currentline)-3):
                            addval = currentline[i+3]
                            if addval[0] == '#':
                                break
                            elif value[-1] == '#':
                                value = value[:-1]
                                break
                            else:
                                value += addval
                    inputparams[key] = value
    return inputparams

def readinputfile(fname,init=True):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the metal_props class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.'''
    inputparams = loadinputfile(fname)
    sym = inputparams['sym']
    if 'name' in inputparams.keys():
        name = inputparams['name']
    else:
        name = str(fname)
    out = metal_props(sym,name)
    out.populate_from_dict(inputparams)
    if init:
        out.init_all()
    return out

############################################################

if __name__ == '__main__':
    Y={}
    inputdata = {}
    metal_list = []
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        try:
            for i in range(len(args)):
                inputdata[i]=readinputfile(args[i])
                X = inputdata[i].name
                metal_list.append(X)
                Y[X] = inputdata[i]
            metal_symm = metal = set([])
            print(f"success reading input files {args}")
        except FileNotFoundError:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = sys.argv[1].split()
    
    sym={}
    for X in data.fcc_metals.intersection(metal): sym[X]='fcc'
    for X in data.bcc_metals.intersection(metal): sym[X]='bcc'
    for X in data.hcp_metals.intersection(metal): sym[X]='hcp'
    for X in data.tetr_metals.intersection(metal): sym[X]='tetr'
    for X in metal:
        Y[X] = metal_props(sym[X],name=X)
        # 2nd order elastic constants taken from the CRC handbook:
        Y[X].c11 = data.CRC_c11[X]
        Y[X].c12 = data.CRC_c12[X]
        Y[X].c44 = data.CRC_c44[X]
        Y[X].c13 = data.CRC_c13[X]
        Y[X].c33 = data.CRC_c33[X]
        Y[X].c66 = data.CRC_c66[X]
        ## numbers from Thomas:1968, Hiki:1966, Leese:1968, Powell:1984, and Graham:1968
        ## (these were used for arXiv:1706.07132)
        # if X in data.THLPG_c11.keys():
        #     Y[X].c11 = data.THLPG_c11[X]
        #     Y[X].c12 = data.THLPG_c12[X]
        #     Y[X].c44 = data.THLPG_c44[X]
        Y[X].init_C2()
        ## TOEC from various refs.
        if X in data.c123.keys():
            Y[X].c111 = data.c111[X]
            Y[X].c112 = data.c112[X]
            Y[X].c123 = data.c123[X]
            Y[X].c144 = data.c144[X]
            Y[X].c166 = data.c166[X]
            Y[X].c456 = data.c456[X]
            Y[X].c113 = data.c113[X]
            Y[X].c133 = data.c133[X]
            Y[X].c155 = data.c155[X]
            Y[X].c222 = data.c222[X]
            Y[X].c333 = data.c333[X]
            Y[X].c344 = data.c344[X]
            Y[X].c366 = data.c366[X]
            Y[X].init_C3()
    
    if metal == set([]):
        metal = metal_list ## triggers only if user provided one or more inputdata files
        
    metal_cubic = []
    metal_toec = []
    metal_toec_cubic = []
    for X in metal:
        ### will compute improved averages for cubic metals only:
        if Y[X].sym == 'fcc' or Y[X].sym == 'bcc':
            metal_cubic.append(X)
        if abs(Y[X].c123)>0: ### subset for which we have TOEC
            metal_toec.append(X)
            if Y[X].sym == 'fcc' or Y[X].sym == 'bcc':
                metal_toec_cubic.append(X)
    print(f"Computing for: {metal} (={len(metal)} metals)")

    # results to be stored in the following dictionaries (for various metals)
    VoigtAverage = {}
    ReussAverage = {}
    HillAverage = {}
    ImprovedAv = {}
    
    aver = IsoAverages(lam,mu,Murl,Murm,Murn) ### initialize isotropic quantities first
    print(f"Computing Voigt and Reuss averages for SOEC of {len(metal)} metals and for TOEC of {len(metal_toec)} metals ...")
    # do the calculations for various metals:
    C2 = {}
    C3 = {}
    S2 = {}
    S3 = {}
    for X in metal:
        #### devide by 1e9 to get the results in units of GPa
        C2[X] = UnVoigt(Y[X].C2/1e9)
        S2[X] = elasticS2(C2[X])
        C3[X] = None
        S3[X] = None
    
    for X in metal_toec:
        C3[X] = UnVoigt(Y[X].C3/1e9)
        S3[X] = elasticS3(S2[X],C3[X])

    for X in metal:
        VoigtAverage[X] = aver.voigt_average(C2[X],C3[X])
        ReussAverage[X] = aver.reuss_average(S2[X],S3[X])
        HillAverage[X] = aver.hill_average()
    
    print(f"Computing improved averages for SOEC of {len(metal_cubic)} cubic metals and for TOEC of {len(metal_toec_cubic)} cubic metals ...")
    
    for X in metal_cubic:
        ImprovedAv[X] = aver.improved_average(C2[X],C3[X])
    
    ##### write results to files (as LaTeX tables):
    stringofnames = " & $\lambda$ & $\mu$ \\\\ \hline"+"\n"
    def stringofresults(dictionary,X):
        return X+" & "+"{:.1f}".format(float(dictionary[X][lam]))+" & "+"{:.1f}".format(float(dictionary[X][mu]))+" \\\\"+"\n"

    stringofnames_toec = " & $\lambda$ & $\mu$ & $l$ & $m$ & $n$ \\\\ \hline"+"\n"
    def stringofresults_toec(dictionary,X):
        return X+" & "+"{:.1f}".format(float(dictionary[X][lam]))+" & "+"{:.1f}".format(float(dictionary[X][mu]))+" & "+"{:.0f}".format(float(dictionary[X][Murl]))+" & "+"{:.0f}".format(float(dictionary[X][Murm]))+" & "+"{:.0f}".format(float(dictionary[X][Murn]))+" \\\\"+"\n"
    
    with open("averaged_elastic_constants.txt","w") as averfile:
        averfile.write("Voigt averages [GPa]:\n"+stringofnames)
        for X in metal:
            averfile.write(stringofresults(VoigtAverage,X))
        if len(metal_toec)>0: averfile.write("\n\n"+stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(VoigtAverage,X))
        averfile.write("\n\n")
        
        averfile.write("Reuss averages [GPa]:\n"+stringofnames)
        for X in metal:
            averfile.write(stringofresults(ReussAverage,X))
        if len(metal_toec)>0: averfile.write("\n\n"+stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(ReussAverage,X)) 
        averfile.write("\n\n")
        
        averfile.write("Hill averages [GPa]:\n"+stringofnames)
        for X in metal:
            averfile.write(stringofresults(HillAverage,X))
        if len(metal_toec)>0: averfile.write("\n\n"+stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(HillAverage,X)) 
        averfile.write("\n\n") 
        
        if len(metal_cubic)>0:
            averfile.write("improved averages [GPa]:\n"+stringofnames)
            for X in metal_cubic:
                averfile.write(stringofresults(ImprovedAv,X))
            if len(metal_toec)>0: averfile.write("\n\n"+stringofnames_toec)
            for X in metal_toec_cubic:
                averfile.write(stringofresults_toec(ImprovedAv,X))
            averfile.write("\n\n")
    print("done.")
    
