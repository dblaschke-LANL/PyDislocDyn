# setup elastic constants and compliances, including Voigt notation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - Feb. 27, 2019
#################################
from __future__ import division
from __future__ import print_function

# from sys import version_info
### make sure we are running a recent version of python
# assert version_info >= (3,5)
from sympy import Symbol
import sympy as sp
import numpy as np
### makes numerical versions faster, but symbolic versions slightly slower
from numba import jit

### vectorize some sympy fcts to make them work on numpy arrays
### (as an alternative to loops and/or converting to and from sympy matrices):
Expand = np.vectorize(sp.expand)
Factor = np.vectorize(sp.factor)
Simplify = np.vectorize(sp.simplify)
###############

delta = np.diag((1,1,1))
Delta = sp.KroneckerDelta


### generate tensors of elastic constants
@jit
def elasticC2(c12, c44, c11=None, c13=None, c33=None, c66=None):
    '''Generates the tensor of second order elastic constants for tetragonal I, hexagonal, cubic and isotropic symmetries using c11, c12, c44, c13, c33, and c66 as input data
    (assuming the third axis is perpendicular to the basal plane).
    If only c66 is omitted (or 'None'), hexagonal symmetry is assumed and this function will set c66=(c11-c12)/2.
    If c13 or c33 are omitted (or 'None'), cubic symmetry is assumed and the according tensor is generated with c13=c12, c33=c11, and c66=c44.
    If in addition c11 is omitted (or 'None'), an isotropic tensor is generated with c11 = c12+2*c44.'''
    if c13==None or c33==None:
        c13=c12
        c33=c11
        c66=c44
        if c11==None:
            c11 = c12+2*c44
    elif c66==None:
        c66 = (c11-c12)/2
    if isinstance(c11, Symbol) or isinstance(c12, Symbol) or isinstance(c44, Symbol) or isinstance(c33, Symbol) or isinstance(c13, Symbol):
        C2 = np.empty((3,3,3,3), dtype=object)
    else:
        C2 = np.empty((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C2[i,j,k,l] = c12*delta[i,j]*delta[k,l]+c44*(delta[i,k]*delta[j,l]+delta[i,l]*delta[j,k])-(2*c44+c12-c11)*delta[i,k]*delta[j,l]*delta[i,j]
    if c13!=c12:
        C2 = Voigt(C2)
        C2[2,2] = c33
        C2[5,5] = c66
        C2[0,2] = c13
        C2[2,0] = c13
        C2[1,2] = c13
        C2[2,1] = c13
        C2 = UnVoigt(C2)
    return C2
        
@jit
def elasticC3_cubic(c111=None, c112=None, c123=None, c144=None, c166=None, c456=None, l=None, m=None, n=None):
    '''Generates the tensor of third order elastic constants for solids of cubic I symmetry using c111, c112, c123, c144, c166, and c456 as input data.
    If either c111, c112, or c166 are omitted (or 'None'), an isotropic tensor is generated using c123, c144, and c456 as the input.
    Alternatively in the latter case, three Murnaghan l, m, n constants may be given instead.
    (Warning: this function does not support giving any other combination of constants than the ones described above.)'''
    if c111==None or c112==None or c166==None:
        if c123==None or c144==None or c456==None:
            if l==None or m==None or n==None:
                raise ValueError("ERROR: unsupported input.")
                
            c123 = 2*l - 2*m + n
            c144 = m - n/2
            c456 = n/4
            if isinstance(l, Symbol) or isinstance(m, Symbol) or isinstance(n, Symbol):
                C3 = np.empty((3,3,3,3,3,3), dtype=object)
            else:
                C3 = np.empty((3,3,3,3,3,3))
        else:
            if isinstance(c123, Symbol) or isinstance(c144, Symbol) or isinstance(c456, Symbol):
                C3 = np.empty((3,3,3,3,3,3), dtype=object)
            else:
                C3 = np.empty((3,3,3,3,3,3))
        H1 = 0
        H2 = 0
        H3 = 0
    else:
        if isinstance(c111, Symbol) or isinstance(c112, Symbol) or isinstance(c123, Symbol) or isinstance(c144, Symbol) or isinstance(c166, Symbol) or isinstance(c456, Symbol):
            C3 = np.empty((3,3,3,3,3,3), dtype=object)
        else:
            C3 = np.empty((3,3,3,3,3,3))
        H2 = c123 + 2*c144 - c112
        H3 = c144 + 2*c456 - c166
        H1 = (c123 + 6*c144 + 8*c456 - 3*H2 - 12*H3 - c111)
    for i in range(3):
        for ii in range(3):
            for j in range(3):
                for jj in range(3):
                    for k in range(3):
                        for kk in range(3):
                            C3[i,ii,j,jj,k,kk] = c123*delta[i,ii]*delta[j,jj]*delta[k,kk] \
                                                + c144*(delta[i,ii]*(delta[j,k]*delta[jj,kk]+delta[j,kk]*delta[jj,k]) + delta[j,jj]*(delta[i,k]*delta[ii,kk]+delta[i,kk]*delta[ii,k]) + delta[k,kk]*(delta[j,i]*delta[jj,ii]+delta[j,ii]*delta[jj,i])) \
                                                + c456*(delta[i,j]*(delta[ii,k]*delta[jj,kk]+delta[ii,kk]*delta[jj,k]) + delta[ii,jj]*(delta[i,k]*delta[j,kk]+delta[i,kk]*delta[j,k]) \
                                                    + delta[i,jj]*(delta[ii,k]*delta[j,kk]+delta[ii,kk]*delta[j,k]) + delta[ii,j]*(delta[i,k]*delta[jj,kk]+delta[i,kk]*delta[jj,k])) \
                                                - H1*delta[i,ii]*delta[j,jj]*delta[k,kk]*delta[i,j]*delta[i,k] \
                                                - H2*delta[i,ii]*delta[j,jj]*delta[k,kk]*(delta[i,j] + delta[j,k] + delta[k,i]) \
                                                - H3*(delta[i,ii]*(delta[j,k]*delta[jj,kk]+delta[j,kk]*delta[jj,k])*(delta[i,j] + delta[i,jj]) \
                                                    + delta[j,jj]*(delta[i,k]*delta[ii,kk]+delta[i,kk]*delta[ii,k])*(delta[j,k] + delta[j,kk]) \
                                                    + delta[k,kk]*(delta[j,i]*delta[jj,ii]+delta[j,ii]*delta[jj,i])*(delta[i,k] + delta[ii,k]))
    return C3

### we follow the conventions of Brugger 1964
def elasticC3(c111=None, c112=None, c113=None, c123=None, c133=None, c144=None, c155=None, c166=None, c222=None, c333=None, c344=None, c366=None, c456=None, l=None, m=None, n=None):
    '''Generates the tensor of third order elastic constants for solids using c111, c112, c113, c123, c133, c144, c155, c166, c222, c333, c344, c366, and c456 as input data.
    If c333 is omitted (or 'None'), cubic I symmetry is assumed.
    If in addition either c111, c112, or c166 are omitted (or 'None'), an isotropic tensor is generated using c123, c144, and c456 as the input.
    Alternatively in the latter case, three Murnaghan l, m, n constants may be given instead.
    If c333 is given, but either c456, c166, or c366 are omitted (or 'None'), hexagonal I symmetry is assumed (with the 3rd axis perpendicular to the basal plane).
    If c333 is given, but c222 is omitted (or 'None'), tetragonal I symmetry is assumed.
    (Other symmetries are not yet implemented.)'''
    if c333==None: ### assume cubic I or isotropic
        C3 = elasticC3_cubic(c111=c111, c112=c112, c123=c123, c144=c144, c166=c166, c456=c456, l=l, m=m, n=n)
    else:
        Cdict = {'C111':c111, 'C112':c112, 'C113':c113, 'C114':0, 'C115':0, 'C116':0, 'C122':0, 'C123':c123, 'C124':0, 'C125':0, 'C126':0, 'C133':c133, 'C134':0, 'C135':0, 'C136':0, 'C144':c144, 'C145':0, 'C146':0, 'C155':c155, 'C156':0, 'C166':c166,\
            'C222':c222, 'C223':0, 'C224':0, 'C225':0, 'C226':0, 'C233':0, 'C234':0, 'C235':0, 'C236':0, 'C244':0, 'C245':0, 'C246':0, 'C255':0, 'C256':0, 'C266':0,\
             'C333':c333, 'C334':0, 'C335':0, 'C336':0, 'C344':c344, 'C345':0, 'C346':0, 'C355':0, 'C356':0, 'C366':c366, 'C444':0, 'C445':0, 'C446':0, 'C455':0, 'C456':c456, 'C466':0, 'C555':0, 'C556':0, 'C566':0, 'C666':0, }
        if c456==None or c166==None or c366==None: ### assume hexagonal I
            Cdict['C456'] = (c155-c144)/2
            Cdict['C122'] = c111+c112-c222
            Cdict['C166'] = (3*c222-c112-2*c111)/4
            Cdict['C266'] = (2*c111-c112-c222)/4
            Cdict['C366'] = (c113-c123)/2
            Cdict['C223'] = c113
            Cdict['C233'] = c133
            Cdict['C244'] = c155
            Cdict['C255'] = c144
            Cdict['C355'] = c344
        elif c222==None: ### assume tetragonal I
            Cdict['C222']=c111
            Cdict['C122']=c112
            Cdict['C223']=c113
            Cdict['C233']=c133
            Cdict['C244']=c155
            Cdict['C255']=c144
            Cdict['C266']=c166
            Cdict['C355']=c344
        else:
            raise ValueError("ERROR: not implemented.")
        if isinstance(sum(Cdict.values()),sp.Expr):
            C3 = np.empty((6,6,6), dtype=object)
        else:
            C3 = np.empty((6,6,6))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    ii,jj,kk = tuple(sorted([i+1,j+1,k+1]))
                    C3[i,j,k] = Cdict["C"+str(ii)+str(jj)+str(kk)]
        C3 = UnVoigt(C3)
    return C3
    
### Convert to and from Voigt notation:
VoigtIndices = [0,4,8,5,2,1]
UnVoigtIndices = [0,5,4,5,1,3,4,3,2]

def Voigt(elasticC):
    '''Converts Voigt-symmetric tensors of ranks 2, 4, and 6 (such as strain/stress tensors or 2nd and 3rd order elastic constants) into Voigt notation.
    Both input and output are numpy arrays. Warning: this function does not check the input for Voigt symmetry, use CheckVoigt() instead.'''
    CVoigt = np.asarray(elasticC)
    if CVoigt.ndim == 2:
        CVoigt = np.reshape(CVoigt,(9))[VoigtIndices]
    elif CVoigt.ndim == 4:
        CVoigt = np.reshape(CVoigt,(9,9))[VoigtIndices][:,VoigtIndices]
    elif CVoigt.ndim == 6:
        CVoigt = np.reshape(CVoigt,(9,9,9))[VoigtIndices][:,VoigtIndices][:,:,VoigtIndices]
    elif CVoigt.ndim == 8:
        CVoigt = np.reshape(CVoigt,(9,9,9,9))[VoigtIndices][:,VoigtIndices][:,:,VoigtIndices][:,:,:,VoigtIndices]
    else:
        print('not implemented for array of dimension',CVoigt.ndim,', returning input')
    return CVoigt

def UnVoigt2(CVoigt):
    '''Deprecaded, call UnVoigt() instead.'''
    print("Warning: UnVoigt2() is deprecated, call UnVoigt() instead!")
    return UnVoigt(CVoigt)
    
def UnVoigt3(CVoigt):
    '''Deprecaded, call UnVoigt() instead.'''
    print("Warning: UnVoigt3() is deprecated, call UnVoigt() instead!")
    return UnVoigt(CVoigt)

def UnVoigt(CVoigt):
    '''Converts tensors of ranks 1, 2, and 3 (such as strain/stress tensors or 2nd and 3rd order elastic constants) from Voigt to conventional tensor notation.
    Both input and output are numpy arrays.'''
    elasticC = np.asarray(CVoigt)
    if elasticC.ndim == 1:
        elasticC = np.reshape(elasticC[UnVoigtIndices],(3,3))
    elif elasticC.ndim == 2:
        elasticC = np.reshape(elasticC[UnVoigtIndices][:,UnVoigtIndices],(3,3,3,3))
    elif elasticC.ndim == 3:
        elasticC = np.reshape(elasticC[UnVoigtIndices][:,UnVoigtIndices][:,:,UnVoigtIndices],(3,3,3,3,3,3))
    elif elasticC.ndim == 4:
        elasticC = np.reshape(elasticC[UnVoigtIndices][:,UnVoigtIndices][:,:,UnVoigtIndices][:,:,:,UnVoigtIndices],(3,3,3,3,3,3,3,3))
    else:
        print('not implemented for array of dimension',elasticC.ndim,', returning input')
    return elasticC
    
## check for Voigt symmetry
def CheckVoigt(tensor):
    '''Quick and dirty check for Voigt symmetry: only checks UnVoigt(Voigt(tensor).T)==tensor, so don't rely on it entirely.
    Input must be a (numpy array) tensor of rank 2, 4, or 6.'''
    return np.all(UnVoigt(Voigt(tensor).T)==tensor)
    
##### generate tensors of elastic compliances
def elasticS2(elasticC2):
    '''Generates the tensor of second order elastic compliances using a second order elastic constants tensor, elasticC2, as input data.'''
    sprimetos = np.diag([1,1,1,0.5,0.5,0.5])
    if elasticC2.dtype.kind == 'O':
        ### need to convert to sympy matrix in order to call sympy's symbolic matrix inversion
        result = np.array(sp.simplify(sp.Matrix(Voigt(elasticC2)).inv()))
    else:
        result = np.linalg.inv(Voigt(np.asarray(elasticC2, dtype=float)))
    return UnVoigt(np.dot(sprimetos,np.dot(result,sprimetos)))
        
def elasticS3(elasticS2,elasticC3):
    '''Generates the tensor of third order elastic compliances using the second order elastic compliances tensor, elasticS2, and the third order elastic constants tensor, elasticC3, as input data.'''
    if elasticS2.dtype.kind == 'O':
        S3 = np.zeros((6,6,6),dtype=object)
    else:
        S3 = np.zeros((6,6,6))
    # need to sum indices 3,4,5 twice below, therefore:
    S2P = np.dot(Voigt(elasticS2),np.diag([1,1,1,2,2,2]))
    S3 = -np.dot(S2P,np.dot(S2P,np.dot(Voigt(elasticC3),S2P.T)))
    # for i in range(6):
    #             ### simplify takes too long, factor is somewhat faster but perhaps still an unnecessary delay
    #             S3[i] = np.array(sp.factor(sp.Matrix(S3[i])))
    return UnVoigt(S3)
    
