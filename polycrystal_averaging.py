# Compute averages of elastic constants for polycrystals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - Sept. 18, 2019
#################################
from __future__ import division
from __future__ import print_function

import sys
### make sure we are running a recent version of python
# assert sys.version_info >= (3,5)
from sympy import Symbol
from sympy.solvers import solve 
import sympy as sp
import numpy as np
from elasticconstants import Voigt, UnVoigt
import elasticconstants as ec
import metal_data as data

### vectorize some sympy fcts to make them work on numpy arrays
### (as an alternative to loops and/or converting to and from sympy matrices):
Expand = np.vectorize(sp.expand)
Factor = np.vectorize(sp.factor)
Simplify = np.vectorize(sp.simplify)
###############

delta = ec.delta
Delta = sp.KroneckerDelta

#### input data:
## numbers from the CRC handbook
c11 = data.CRC_c11
c12 = data.CRC_c12
c44 = data.CRC_c44
c13 = data.CRC_c13
c33 = data.CRC_c33
c66 = data.CRC_c66

## numbers from Thomas:1968, Hiki:1966, Leese:1968, Powell:1984, and Graham:1968
## (these were used for arXiv:1706.07132)
# c11 = data.THLPG_c11
# c12 = data.THLPG_c12
# c44 = data.THLPG_c44

c111 = data.c111
c112 = data.c112
c123 = data.c123
c144 = data.c144
c166 = data.c166
c456 = data.c456

c113 = data.c113
c133 = data.c133
c155 = data.c155
c222 = data.c222
c333 = data.c333
c344 = data.c344
c366 = data.c366

### generate a list of those fcc, bcc, hcp metals for which we have sufficient data
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals).intersection(c11.keys())))

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
    
## can use these same fcts for the invariants of S2 and S3

### define some symbols and functions of symbols for the effective isotropic side of our calculations
# Symbols for Lame constants:
lam = Symbol('lam')
mu = Symbol('mu')
# Symbols for Murnaghan constants:
Murl = Symbol('Murl')
Murm = Symbol('Murm')
Murn = Symbol('Murn')

class IsoInvariants(object):
    def __init__(self,lam,mu,Murl,Murm,Murn):
        '''This class initializes isotropic second and third order elastic tensors and their invariants depending on the provided Lame and Murnaghan constants
        (which may be sympy symbols).'''
        self.C2 = ec.elasticC2(c12=lam,c44=mu)
        self.invI1 = sp.factor(invI1(self.C2))
        self.invI2 = sp.factor(invI2(self.C2))
        
        self.C3 = ec.elasticC3(l=Murl,m=Murm,n=Murn)
        self.invI3 = sp.factor(invI3(self.C3))
        self.invI4 = sp.factor(invI4(self.C3))
        self.invI5 = sp.factor(invI5(self.C3))
        
        self.S2 = ec.elasticS2(self.C2)
        self.invI1b = sp.factor(invI1(self.S2))
        self.invI2b = sp.factor(invI2(self.S2))
        
        self.S3 = ec.elasticS3(self.S2,self.C3)
        self.invI3b = sp.factor(invI3(self.S3))
        self.invI4b = sp.factor(invI4(self.S3))
        self.invI5b = sp.factor(invI5(self.S3))

######### Voigt, Reuss, & Hill averages
def voigt_average(C2,C3=None,iso=None):
    '''Computes the Voigt averages of elastic constants using the tensor(s) C2 and C3 of SOEC and TOEC (optional). The output is a dictionary of average values for the Lame and Murnaghan constants.
       If C3 is omitted (or 'None'), only the average Lame constants are computed.
       If 'iso', an object of class IsoInvariants(), is passed explicitly, computation is faster since initialization of that class can be bypassed.'''
    if iso==None:
        iso = IsoInvariants(lam,mu,Murl,Murm,Murn)
    out = solve([iso.invI1-invI1(C2),iso.invI2-invI2(C2)],[lam,mu],dict=True)[0]
    ## combine all five Voigt-averaged answers into one dictionary for each metal (i.e. Voigt is a dictionary of dictionaries)
    if np.asarray(C3).all()!=None:
        out.update(solve([iso.invI3-invI3(C3),iso.invI4-invI4(C3),iso.invI5-invI5(C3)],[Murl,Murm,Murn],dict=True)[0])
    return out

def reuss_average(S2,S3=None,iso=None):
    '''Computes the Reuss averages of elastic constants using the second and third (optional) order compliances tensor(s) S2 and S3. The output is a dictionary of average values for the Lame and Murnaghan constants.
       If S3 is omitted (or 'None'), only the average Lame constants are computed.
       If 'iso', an object of class IsoInvariants(), is passed explicitly, computation is faster since initialization of that class can be bypassed.'''
    if iso==None:
        iso = IsoInvariants(lam,mu,Murl,Murm,Murn)
    out = solve([iso.invI1b-invI1(S2),iso.invI2b-invI2(S2)],[lam,mu],dict=True)[0]
    ## plug in results for lam and mu to compute l,m,n, then combine the results into one dictionary:
    if np.asarray(S3).all()!=None:
        out.update(solve([iso.invI3b.subs(out)-invI3(S3),iso.invI4b.subs(out)-invI4(S3),iso.invI5b.subs(out)-invI5(S3)],[Murl,Murm,Murn],dict=True)[0])
    return out

def hill_average(voigt,reuss):
    '''Computes the Hill averages of the Lame and Murnaghan (optional) constants using dictionaries of Voigt and Reuss averages as input.'''
    out = {lam : (voigt[lam] + reuss[lam])/2}
    out.update({mu : (voigt[mu] + reuss[mu])/2})
    if len(reuss.keys())==5:
        out.update({Murl : (voigt[Murl] + reuss[Murl])/2})
        out.update({Murm : (voigt[Murm] + reuss[Murm])/2})
        out.update({Murn : (voigt[Murn] + reuss[Murn])/2})
    return out
	
    
############################# for improved method, see Lubarda 1997 and Blaschke 2017
Sh = Symbol('Sh', real=True)
def Hmat(Sh):
    '''helper function for improved_average().'''
    ### Amat is (x_i x_j x_k x_l + y_i y_j y_k y_l + z_i z_j z_k z_l) where x,y,z are the unit vectors along the crystal axes
    ### one may easily check that in Voigt notation this construction leads to the diagonal matrix below
    Amat = UnVoigt(np.diag([1,1,1,0,0,0]))
    if isinstance(Sh,Symbol) or isinstance(Sh,float):
        result = ec.elasticC2((Sh,Sh+1/2)) -5*Sh*Amat
    else:
        ## play it safe (and slower):
        Sh2 = Symbol('Sh2', real=True)
        ### need to convert to sympy matrix in order to use the subs method
        result = UnVoigt(np.array(sp.Matrix(Voigt(ec.elasticC2(c12=Sh2,c44=Sh2+1/2))).subs(Sh2,Sh))) -5*Sh*Amat
    return result

def elasticC2hat(C2,Sh):
    '''helper function for improved_average().'''
    correction = np.diag([1,1,1,2,2,2])
    Sh2 = Symbol('Sh2', real=True)
    Hm = np.dot(Voigt(Hmat(Sh2/2)),correction)
    return UnVoigt(np.array(sp.factor(sp.expand(sp.Matrix(np.dot(Hm,np.dot(Voigt(C2),Hm.T)))).subs(Sh2**2,0)).subs(Sh2,Sh)))
    
def elasticC3hat(C3,Sh):
    '''helper function for improved_average().'''
    correction = np.diag([1,1,1,2,2,2])
    Sh2 = Symbol('Sh2', real=True)
    Hm = np.dot(Voigt(Hmat(Sh2/2)),correction)
    result = np.dot(Hm,np.dot(Hm,np.dot(Voigt(C3),Hm.T)))
    for i in range(6):
        result[i] = np.array(((sp.expand(sp.Matrix(result[i])).subs(Sh2**2,0)).subs(Sh2**3,0)).subs(Sh2,Sh))
    return UnVoigt(result)

def improved_average(C2,C3=None,iso=None):
    '''Compute an improved average for elastic constants of polycrystals whose single crystals are of cubic I symmetry.
       For the Lame constants, this function will use the self-consistent method of Hershey and Kroener. For the Murnaghan constants, a generalization thereof is used following Blaschke 2017.
       If 'iso', an object of class IsoInvariants(), is passed explicitly, computation is faster since initialization of that class can be bypassed.'''
    if iso==None:
        iso = IsoInvariants(lam,mu,Murl,Murm,Murn)
    C11 = C2[0,0,0,0]
    C12 = C2[0,0,1,1]
    hdenominator = (3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
    hfactor = (C11 + 2*C12 + 6*mu)*(C11 - C12 - 2*mu)/(3*(8*mu**2 + 9*C11*mu + (C11 - C12)*(C11 + 2*C12)))
    C2hat = elasticC2hat(C2,hfactor)
    tmplam = solve(iso.invI1-sp.factor(invI1(C2hat)),lam,dict=True)[0][lam]
    tmpeqn = sp.factor(hdenominator*(iso.invI2-sp.factor(invI2(C2hat))).subs(lam,tmplam))
    tmpmu = np.array(solve(tmpeqn,mu),dtype=complex)
    tmpmu = tmpmu[tmpmu>0]
    if np.imag(tmpmu)/np.real(tmpmu)<1e-15 and len(tmpmu)==1:
        tmpmu = np.real(tmpmu)[0]
    else:
        print("ERROR: found mu[{0}]={1}; unable to determine solution for mu[{0}]! Setting to 0.".format(X,tmpmu))
        tmpmu = 0
    tmplam = tmplam.subs(mu,tmpmu)
    out = {lam:tmplam, mu:tmpmu}
    
    if np.asarray(C3).all()!=None:
        C3hat = elasticC3hat(C3,hfactor)
        out.update(solve([(iso.invI3-invI3(C3hat)).subs(out),(iso.invI4-invI4(C3hat)).subs(out),(iso.invI5-invI5(C3hat)).subs(out)],[Murl,Murm,Murn],dict=True)[0])

    return out

############################################################

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
        metal = sys.argv[1].split()
    print("Computing for: {0} (={1} metals)".format(metal,len(metal)))
    ### subset for which we have TOEC:
    metal_toec = sorted(list(set(metal).intersection(c111.keys())))

    # results to be stored in the following dictionaries (for various metals)
    VoigtAverage = {}
    ReussAverage = {}
    HillAverage = {}
    ImprovedAv = {}
    
    iso = IsoInvariants(lam,mu,Murl,Murm,Murn) ### compute isotropic quantities first    
    print("Computing Voigt and Reuss averages for SOEC of {0} metals and for TOEC of {1} metals ...".format(len(metal),len(metal_toec)))
    # do the calculations for various metals:
    C2 = {}
    C3 = {}
    S2 = {}
    S3 = {}
    for X in metal:
        #### devide by 1e9 to get the results in units of GPa
        C2[X] = ec.elasticC2(c11=c11[X], c12=c12[X], c44=c44[X], c13=c13[X], c33=c33[X], c66=c66[X])/1e9    
        S2[X] = ec.elasticS2(C2[X])
        C3[X] = None
        S3[X] = None
    
    for X in metal_toec:
        C3[X] = ec.elasticC3(c111=c111[X], c112=c112[X], c113=c113[X], c123=c123[X], c133=c133[X], c144=c144[X], c155=c155[X], c166=c166[X], c222=c222[X], c333=c333[X], c344=c344[X], c366=c366[X], c456=c456[X])/1e9
        S3[X] = ec.elasticS3(S2[X],C3[X])

    for X in metal:
        VoigtAverage[X] = voigt_average(C2[X],C3[X],iso)
        ReussAverage[X] = reuss_average(S2[X],S3[X],iso)
        HillAverage[X] = hill_average(VoigtAverage[X], ReussAverage[X])
    
    ### compute improved averages for cubic metals only:
    metal_cubic = sorted(list(data.fcc_metals.union(data.bcc_metals).intersection(set(metal))))
    metal_toec_cubic = sorted(list(data.fcc_metals.union(data.bcc_metals).intersection(set(metal_toec))))
    print("Computing improved averages for SOEC of {0} cubic metals and for TOEC of {1} cubic metals ...".format(len(metal_cubic),len(metal_toec_cubic)))
    
    for X in metal_cubic:
        ImprovedAv[X] = improved_average(C2[X],C3[X],iso)
    
    ##### write results to files (as LaTeX tables):
    stringofnames = " & $\lambda$ & $\mu$ \\\\ \hline"+"\n"
    def stringofresults(dictionary,X):
        return X+" & "+"{:.1f}".format(float(dictionary[X][lam]))+" & "+"{:.1f}".format(float(dictionary[X][mu]))+" \\\\"+"\n"

    stringofnames_toec = " & $\lambda$ & $\mu$ & $l$ & $m$ & $n$ \\\\ \hline"+"\n"
    def stringofresults_toec(dictionary,X):
        return X+" & "+"{:.1f}".format(float(dictionary[X][lam]))+" & "+"{:.1f}".format(float(dictionary[X][mu]))+" & "+"{:.0f}".format(float(dictionary[X][Murl]))+" & "+"{:.0f}".format(float(dictionary[X][Murm]))+" & "+"{:.0f}".format(float(dictionary[X][Murn]))+" \\\\"+"\n"
    
    with open("averaged_elastic_constants.txt","w") as averfile:
        averfile.write("Voigt averages [GPa]:\n")
        averfile.write(stringofnames)
        for X in metal:
            averfile.write(stringofresults(VoigtAverage,X))
        averfile.write("\n\n")    
        averfile.write(stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(VoigtAverage,X))
        averfile.write("\n\n")
        
        averfile.write("Reuss averages [GPa]:\n")
        averfile.write(stringofnames)
        for X in metal:
            averfile.write(stringofresults(ReussAverage,X))
        averfile.write("\n\n")
        averfile.write(stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(ReussAverage,X)) 
        averfile.write("\n\n")
        
        averfile.write("Hill averages [GPa]:\n")
        averfile.write(stringofnames)
        for X in metal:
            averfile.write(stringofresults(HillAverage,X))
        averfile.write("\n\n")
        averfile.write(stringofnames_toec)
        for X in metal_toec:
            averfile.write(stringofresults_toec(HillAverage,X)) 
        averfile.write("\n\n") 
    
        averfile.write("improved averages [GPa]:\n")
        averfile.write(stringofnames)
        for X in metal_cubic:
            averfile.write(stringofresults(ImprovedAv,X))
        averfile.write("\n\n")
        averfile.write(stringofnames_toec)
        for X in metal_toec_cubic:
            averfile.write(stringofresults_toec(ImprovedAv,X))
        averfile.write("\n\n")

    print("done.")
        
