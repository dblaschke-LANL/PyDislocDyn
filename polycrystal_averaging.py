# Compute averages of elastic constants for polycrystals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - Apr. 24, 2019
#################################
from __future__ import division
from __future__ import print_function

import sys
### make sure we are running a recent version of python
# assert sys.version_info >= (3,6)
from sympy.solvers import solve 
import sympy as sp
import numpy as np
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
        tmpmu = tmpmu[tmpmu>0]
        if np.imag(tmpmu)/np.real(tmpmu)<1e-15 and len(tmpmu)==1:
            tmpmu = np.real(tmpmu)[0]
        else:
            print("ERROR: found mu={}; unable to determine solution for mu! Setting to 0.".format(tmpmu))
            tmpmu = 0
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
    '''This class stores various metal properties; needed input sym must be one of 'iso', 'fcc', 'bcc', 'hcp', 'tetr'
    and this will define which input parameters (such as elastic constants) are set/expected by the various initializing functions.'''
    def __init__(self,sym='iso', name='some_crystal'):
        self.sym=sym
        self.name = name
        self.T=300
        self.ac = self.cc = 0
        self.rho = 0
        self.c11=self.c12=self.c44=0
        self.c13=self.c33=self.c66=0
        self.c111=self.c112=self.c123=0
        self.c144=self.c166=self.c456=0
        self.c113=self.c133=self.c155=0
        self.c222=self.c333=self.c344=self.c366=0
        self.C2=np.zeros((6,6)) #tensor of SOEC in Voigt notation
        self.C3=np.zeros((6,6,6)) # tensor of TOEC in Voigt notation
        self.mu=self.lam=self.bulk=None ## polycrystal averages
        self.ct=self.cl=self.ct_over_cl=0 ## polycrystal averages
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
            self.cc=self.c13=self.c33=self.c66=None
            self.c113=self.c133=self.c155=self.c222=self.c333=self.c344=self.c366=None
        if sym=='iso':
            self.c11=self.c111=self.c112=self.c166=None
            
    def __repr__(self):
        return  "{}".format({'name':self.name, 'sym':self.sym, 'T':self.T, 'ac':self.ac, 'cc':self.cc, 'rho':self.rho, 'ct':self.ct, 'cl':self.cl})
            
    def init_C2(self):
        self.C2=elasticC2(c11=self.c11,c12=self.c12,c13=self.c13,c33=self.c33,c44=self.c44,c66=self.c66,voigt=True)
    
    def init_C3(self):
        self.C3=elasticC3(c111=self.c111,c112=self.c112,c113=self.c113,c123=self.c123,c133=self.c133,c144=self.c144,c155=self.c155,c166=self.c166,c222=self.c222,c333=self.c333,c344=self.c344,c366=self.c366,c456=self.c456,voigt=True)
    
    def compute_Lame(self, roundto=-8):
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
    
    def init_sound(self):
        '''Computes the effective sound speeds of a polycrystal from its averaged Lame constants.'''
        self.ct = np.sqrt(self.mu/self.rho)
        self.cl = np.sqrt((self.lam+2*self.mu)/self.rho)
        self.ct_over_cl = self.ct/self.cl
        
    def init_qBZ(self):
        if self.sym=='iso' or self.sym=='fcc' or self.sym=='bcc':
            self.qBZ = ((6*np.pi**2)**(1/3))/self.ac
        elif self.sym=='hcp':
            self.qBZ = ((4*np.pi**2/(self.ac*self.ac*self.cc*np.sqrt(3)))**(1/3))
        elif self.sym=='tetr':
            self.qBZ = ((6*np.pi**2/(self.ac*self.ac*self.cc))**(1/3))
    
    def init_all(self):
        self.init_C2()
        self.init_C3()      
        if self.mu==None:
            self.compute_Lame()
        else:
            self.bulk=self.lam + 2*self.mu/3
        self.init_sound()
        if self.ac>0: self.init_qBZ()
        
    def computesound(self,v):
        '''Computes the sound speeds of the crystal propagating in the direction of unit vector v.'''
        bt2 = sp.symbols('bt2')
        v = np.asarray(v)
        v = v/np.sqrt(np.dot(v , v))
        thematrix = np.dot(v , np.dot(UnVoigt(self.C2/self.c44) , v)) - bt2* np.diag([1,1,1]) ## compute normalized bt2 = v^2 /( c44/rho )
        thedet = sp.det(sp.Matrix(thematrix))
        solution = sp.solve(thedet,bt2)
        for i in range(len(solution)):
            solution[i] = solution[i]  * self.c44/self.rho
            if solution[i].free_symbols == set():
                solution[i] = np.sqrt(float(sp.re(solution[i])))
            else:
                solution[i] = sp.sqrt(solution[i])
        return solution
        
def readinputfile(fname,init=True):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the metal_props class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.'''
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
    sym = inputparams['sym']
    out = metal_props(sym)
    keys=inputparams.keys()
    if 'name' in keys:
        out.name = inputparams['name']
    else:
        out.name = str(fname)
    if 'T' in keys:
        out.T=float(inputparams['T'])
    out.b=np.asarray(inputparams['b'].split(','),dtype=float)
    out.b = out.b/np.sqrt(np.dot(out.b,out.b))
    out.burgers=float(inputparams['burgers'])
    out.n0=np.asarray(inputparams['n0'].split(','),dtype=float)
    out.n0 = out.n0/np.sqrt(np.dot(out.n0,out.n0))
    if 'lam' in keys and 'mu' in keys:
        out.lam=float(inputparams['lam'])
        out.mu=float(inputparams['mu'])
    out.ac=float(inputparams['a'])
    out.rho = float(inputparams['rho'])
    if sym !='iso':
        out.c11 = float(inputparams['c11'])
    out.c12 = float(inputparams['c12'])
    out.c44 = float(inputparams['c44'])
    if sym=='hcp' or sym=='tetr':
        out.cc = float(inputparams['c'])
        out.c13 = float(inputparams['c13'])
        out.c33 = float(inputparams['c33'])
        if 'c113' in inputparams.keys():
            out.c113 = float(inputparams['c113'])
            out.c133 = float(inputparams['c133'])
            out.c155 = float(inputparams['c155'])
            out.c333 = float(inputparams['c333'])
            out.c344 = float(inputparams['c344'])
    if sym=='tetr':
        out.c66 = float(inputparams['c66'])
        if 'c366' in inputparams.keys():
            out.c366 = float(inputparams['c366'])
    if 'c111' in inputparams.keys() and sym != 'iso':
        out.c111 = float(inputparams['c111'])
        out.c112 = float(inputparams['c112'])
    if 'c123' in inputparams.keys():
        out.c123 = float(inputparams['c123'])
        out.c144 = float(inputparams['c144'])
        if sym != 'hcp' and sym != 'iso':
            out.c166 = float(inputparams['c166'])
        if sym !='hcp':
            out.c456 = float(inputparams['c456'])
        if sym =='hcp':
            out.c222 = float(inputparams['c222'])
    if 'vcrit_smallest' in keys:
        out.vcrit_smallest = float(inputparams['vcrit_smallest'])
    if 'vcrit_screw' in keys:
        out.vcrit_screw = float(inputparams['vcrit_screw'])
    if 'vcrit_edge' in keys:
        out.vcrit_edge = float(inputparams['vcrit_edge'])
    if 'alpha_a' in keys:
        out.alpha_a = float(inputparams['alpha_a'])
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
            print("success reading input files ",args)
        except FileNotFoundError:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = sys.argv[1].split()
    
    sym={}
    for X in data.fcc_metals.intersection(metal): sym[X]='fcc'
    for X in data.bcc_metals.intersection(metal): sym[X]='bcc'
    for X in data.hcp_metals.intersection(metal): sym[X]='hcp'
    for X in data.tetr_metals.intersection(metal): sym[X]='tetr'
    for X in metal:
        Y[X] = metal_props(sym[X])
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
    print("Computing for: {0} (={1} metals)".format(metal,len(metal)))

    # results to be stored in the following dictionaries (for various metals)
    VoigtAverage = {}
    ReussAverage = {}
    HillAverage = {}
    ImprovedAv = {}
    
    aver = IsoAverages(lam,mu,Murl,Murm,Murn) ### initialize isotropic quantities first
    print("Computing Voigt and Reuss averages for SOEC of {0} metals and for TOEC of {1} metals ...".format(len(metal),len(metal_toec)))
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
    
    print("Computing improved averages for SOEC of {0} cubic metals and for TOEC of {1} cubic metals ...".format(len(metal_cubic),len(metal_toec_cubic)))
    
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
    
