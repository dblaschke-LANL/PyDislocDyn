# setup elastic constants and compliances, including Voigt notation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - Apr. 30, 2024
'''This module contains functions to generate elastic constant and compliance tensors, as well as class to help with calculating ECs.
   In particular, it contains the following functions:
       elasticC2(), elasticC3(),
       elasticS2(), elasticS3(),
       Voigt(), UnVoigt(), CheckVoigt(), CheckReflectionSymmetry(),
       and the class strain_poly.'''
#################################
import sympy as sp
import numpy as np
from pydislocdyn.utilities import delta, roundcoeff

### generate tensors of elastic constants
def elasticC2(c12=None, c44=None, c11=None, c13=None, c33=None, c66=None, c22=None, c23=None, c55=None, cij=None, voigt=False):
    '''Generates the tensor of second order elastic constants using c11, c12, c13, c22, c23, c33, c44, c55, and c66 as input data
    (assuming the third axis is perpendicular to the basal plane).
    If only c22, c23, and c55 are omitted (or 'None'), tetragonal I symmetry is assumed and this function will set c22=c11, c23=c12, and c55=c44.
    If additionally c66 is omitted (or 'None'), hexagonal symmetry is assumed and this function will set c66=(c11-c12)/2.
    If c13 or c33 are omitted (or 'None'), cubic symmetry is assumed and the according tensor is generated with c13=c12, c33=c11, and c66=c44.
    If in addition c11 is omitted (or 'None'), an isotropic tensor is generated with c11 = c12+2*c44.
    For lower/other symmetries, input must be given via keyword cij as a tuple or list containing all required elastic constants
    (21 for triclinic, 13 for monoclinic, 9 for orthorhombic, 7 for tetragonal II, and 6 for trigonal/rhombohedral I) in ascending order where i<=j
    in cij, i.e. cij=(c11, c12, c13, ...); see D. C. Wallace, Solid State Physics 25 (1970) 301 and K. Brugger, J. Appl. Phys. 36 (1965) 759.
    Boolean option 'voigt' determines whether the output is generated in Voigt or Cartesian (default) notation.'''
    if cij is None:
        if c22 is None or c23 is None or c55 is None:
            if c13 is None or c33 is None:
                if c11 is None:
                    c11 = c12+2*c44
                c13=c12
                c33=c11
                c66=c44
            elif c66 is None:
                c66 = (c11-c12)/2
            c22=c11
            c23=c13
            c55=c44
        Cdict = {'C11':c11, 'C12':c12, 'C13':c13, 'C14':0, 'C15':0, 'C16':0, 'C22':c22, 'C23':c23, 'C24':0, 'C25':0, 'C26':0,
                 'C33':c33, 'C34':0, 'C35':0, 'C36':0, 'C44':c44, 'C45':0, 'C46':0, 'C55':c55, 'C56':0, 'C66':c66}
    elif len(cij)==21:
        Cdict = {'C11':cij[0], 'C12':cij[1], 'C13':cij[2], 'C14':cij[3], 'C15':cij[4], 'C16':cij[5], 'C22':cij[6], 'C23':cij[7], 'C24':cij[8], 'C25':cij[9], 'C26':cij[10],
                 'C33':cij[11], 'C34':cij[12], 'C35':cij[13], 'C36':cij[14], 'C44':cij[15], 'C45':cij[16], 'C46':cij[17], 'C55':cij[18], 'C56':cij[19], 'C66':cij[20]}
    elif len(cij)==13:
        Cdict = {'C11':cij[0], 'C12':cij[1], 'C13':cij[2], 'C14':0, 'C15':cij[3], 'C16':0, 'C22':cij[4], 'C23':cij[5], 'C24':0, 'C25':cij[6], 'C26':0,
                 'C33':cij[7], 'C34':0, 'C35':cij[8], 'C36':0, 'C44':cij[9], 'C45':0, 'C46':cij[10], 'C55':cij[11], 'C56':0, 'C66':cij[12]}
    elif len(cij)==9:
        Cdict = {'C11':cij[0], 'C12':cij[1], 'C13':cij[2], 'C14':0, 'C15':0, 'C16':0, 'C22':cij[3], 'C23':cij[4], 'C24':0, 'C25':0, 'C26':0,
                 'C33':cij[5], 'C34':0, 'C35':0, 'C36':0, 'C44':cij[6], 'C45':0, 'C46':0, 'C55':cij[7], 'C56':0, 'C66':cij[8]}
    elif len(cij)==7: ## tetragonal II
        Cdict = {'C11':cij[0], 'C12':cij[1], 'C13':cij[2], 'C14':0, 'C15':0, 'C16':cij[3], 'C22':cij[0], 'C23':cij[2], 'C24':0, 'C25':0, 'C26':-cij[3],
                 'C33':cij[4], 'C34':0, 'C35':0, 'C36':0, 'C44':cij[5], 'C45':0, 'C46':0, 'C55':cij[5], 'C56':0, 'C66':cij[6]}
    elif len(cij)==6: ## trigonal (rhombohedral) I
        Cdict = {'C11':cij[0], 'C12':cij[1], 'C13':cij[2], 'C14':cij[3], 'C15':0, 'C16':0, 'C22':cij[0], 'C23':cij[2], 'C24':-cij[3], 'C25':0, 'C26':0,
                 'C33':cij[4], 'C34':0, 'C35':0, 'C36':0, 'C44':cij[5], 'C45':0, 'C46':0, 'C55':cij[5], 'C56':cij[3], 'C66':(cij[0]-cij[1])/2}
    else:
        raise ValueError(f"len(cij)={len(cij)}, expected 21 (triclinic), 13 (monoclinic), 9 (orthorhombic), 7 (tetragonal II), or 6 (trigonal/rhombohedral I) values")
    if isinstance(sum(Cdict.values()),sp.Expr):
        C2 = np.empty((6,6), dtype=object)
    else:
        C2 = np.empty((6,6))
    for i in range(6):
        for j in range(6):
            ii,jj = tuple(sorted([i+1,j+1]))
            C2[i,j] = Cdict["C"+str(ii)+str(jj)]
    if not voigt:
        C2 = UnVoigt(C2)
    return C2

def elasticC3(c111=None, c112=None, c113=None, c123=None, c133=None, c144=None, c155=None, c166=None, c222=None, c333=None, c344=None, c366=None, c456=None, l=None, m=None, n=None, cijk=None, voigt=False):
    '''Generates the tensor of third order elastic constants for solids using c111, c112, c113, c123, c133, c144, c155, c166, c222, c333, c344, c366, and c456 as input data.
    If c333 is omitted (or 'None'), cubic I symmetry is assumed.
    If in addition either c111, c112, or c166 are omitted (or 'None'), an isotropic tensor is generated using c123, c144, and c456 as the input.
    Alternatively in the latter case, three Murnaghan l, m, n constants may be given instead.
    If c333 is given, but either c456, c166, or c366 are omitted (or 'None'), hexagonal I symmetry is assumed (with the 3rd axis perpendicular to the basal plane).
    If c333 is given, but c222 is omitted (or 'None'), tetragonal I symmetry is assumed.
    For lower/other symmetries, input must be given via keyword cijk as a tuple or list containing all required elastic constants
    (56 for triclinic, 32 for monoclinic, 20 for orthorhombic, 16 for tetragonal II, and 14 for trigonal/rhombohedral I) in ascending order where i<=j<=k in cijk,
    i.e. cij=(c111, c112, ..., c122, ...); K. Brugger, J. Appl. Phys. 36 (1965) 759.
    Boolean option 'voigt' determines whether the output is generated in Voigt or Cartesian (default) notation.'''
    if cijk is None and c333 is None: ### assume cubic I or isotropic
        if c111 is None or c112 is None or c166 is None:
            if c123 is None or c144 is None or c456 is None:
                if l is None or m is None or n is None:
                    raise ValueError("ERROR: unsupported input.")
                c123 = 2*l - 2*m + n
                c144 = m - n/2
                c456 = n/4
            c112 = c123 + 2*c144
            c166 = c144 + 2*c456
            c111 = c123 + 6*c144 + 8*c456
        c113=c122=c133=c223=c233=c112
        c155=c244=c266=c344=c355=c166
        c222=c333=c111
        c255=c366=c144
    elif cijk is None and (c456 is None or c166 is None or c366 is None): ### assume hexagonal I
        c456 = (c155-c144)/2
        c122 = c111+c112-c222
        c166 = (3*c222-c112-2*c111)/4
        c266 = (2*c111-c112-c222)/4
        c366 = (c113-c123)/2
        c223 = c113
        c233 = c133
        c244 = c155
        c255 = c144
        c355 = c344
    elif cijk is None and c222 is None: ### assume tetragonal I
        c222=c111
        c122=c112
        c223=c113
        c233=c133
        c244=c155
        c255=c144
        c266=c166
        c355=c344
    elif cijk is None:
        raise ValueError("ERROR: not implemented.")
    if cijk is None:
        Cdict = {'C111':c111, 'C112':c112, 'C113':c113, 'C114':0, 'C115':0, 'C116':0, 'C122':c122, 'C123':c123, 'C124':0, 'C125':0, 'C126':0, 'C133':c133, 'C134':0, 'C135':0, 'C136':0, 'C144':c144, 'C145':0, 'C146':0, 'C155':c155, 'C156':0, 'C166':c166,
                 'C222':c222, 'C223':c223, 'C224':0, 'C225':0, 'C226':0, 'C233':c233, 'C234':0, 'C235':0, 'C236':0, 'C244':c244, 'C245':0, 'C246':0, 'C255':c255, 'C256':0, 'C266':c266,
                 'C333':c333, 'C334':0, 'C335':0, 'C336':0, 'C344':c344, 'C345':0, 'C346':0, 'C355':c355, 'C356':0, 'C366':c366, 'C444':0, 'C445':0, 'C446':0, 'C455':0, 'C456':c456, 'C466':0, 'C555':0, 'C556':0, 'C566':0, 'C666':0}
    elif len(cijk)==56:
        Cdict = {'C111':cijk[0], 'C112':cijk[1], 'C113':cijk[2], 'C114':cijk[3], 'C115':cijk[4], 'C116':cijk[5], 'C122':cijk[6], 'C123':cijk[7], 'C124':cijk[8], 'C125':cijk[9], 'C126':cijk[10], 'C133':cijk[11], 'C134':cijk[12], 'C135':cijk[13],
                 'C136':cijk[14], 'C144':cijk[15], 'C145':cijk[16], 'C146':cijk[17], 'C155':cijk[18], 'C156':cijk[19], 'C166':cijk[20], 'C222':cijk[21], 'C223':cijk[22], 'C224':cijk[23], 'C225':cijk[24], 'C226':cijk[25], 'C233':cijk[26], 'C234':cijk[27],
                 'C235':cijk[28], 'C236':cijk[29], 'C244':cijk[30], 'C245':cijk[31], 'C246':cijk[32], 'C255':cijk[33], 'C256':cijk[34], 'C266':cijk[35], 'C333':cijk[36], 'C334':cijk[37], 'C335':cijk[38], 'C336':cijk[39], 'C344':cijk[40], 'C345':cijk[41],
                 'C346':cijk[42], 'C355':cijk[43], 'C356':cijk[44], 'C366':cijk[45], 'C444':cijk[46], 'C445':cijk[47], 'C446':cijk[48], 'C455':cijk[49], 'C456':cijk[50], 'C466':cijk[51], 'C555':cijk[52], 'C556':cijk[53], 'C566':cijk[54], 'C666':cijk[55]}
    elif len(cijk)==32:
        Cdict = {'C111':cijk[0], 'C112':cijk[1], 'C113':cijk[2], 'C114':0, 'C115':cijk[3], 'C116':0, 'C122':cijk[4], 'C123':cijk[5], 'C124':0, 'C125':cijk[6], 'C126':0, 'C133':cijk[7], 'C134':0, 'C135':cijk[8],
                 'C136':0, 'C144':cijk[9], 'C145':0, 'C146':cijk[10], 'C155':cijk[11], 'C156':0, 'C166':cijk[12], 'C222':cijk[13], 'C223':cijk[14], 'C224':0, 'C225':cijk[15], 'C226':0, 'C233':cijk[16], 'C234':0,
                 'C235':cijk[17], 'C236':0, 'C244':cijk[18], 'C245':0, 'C246':cijk[19], 'C255':cijk[20], 'C256':0, 'C266':cijk[21], 'C333':cijk[22], 'C334':0, 'C335':cijk[23], 'C336':0, 'C344':cijk[24], 'C345':0,
                 'C346':cijk[25], 'C355':cijk[26], 'C356':0, 'C366':cijk[27], 'C444':0, 'C445':cijk[28], 'C446':0, 'C455':0, 'C456':cijk[29], 'C466':0, 'C555':cijk[30], 'C556':0, 'C566':cijk[31], 'C666':0}
    elif len(cijk)==20:
        Cdict = {'C111':cijk[0], 'C112':cijk[1], 'C113':cijk[2], 'C114':0, 'C115':0, 'C116':0, 'C122':cijk[3], 'C123':cijk[4], 'C124':0, 'C125':0, 'C126':0, 'C133':cijk[5], 'C134':0, 'C135':0,
                 'C136':0, 'C144':cijk[6], 'C145':0, 'C146':0, 'C155':cijk[7], 'C156':0, 'C166':cijk[8], 'C222':cijk[9], 'C223':cijk[10], 'C224':0, 'C225':0, 'C226':0, 'C233':cijk[11], 'C234':0,
                 'C235':0, 'C236':0, 'C244':cijk[12], 'C245':0, 'C246':0, 'C255':cijk[13], 'C256':0, 'C266':cijk[14], 'C333':cijk[15], 'C334':0, 'C335':0, 'C336':0, 'C344':cijk[16], 'C345':0,
                 'C346':0, 'C355':cijk[17], 'C356':0, 'C366':cijk[18], 'C444':0, 'C445':0, 'C446':0, 'C455':0, 'C456':cijk[19], 'C466':0, 'C555':0, 'C556':0, 'C566':0, 'C666':0}
    elif len(cijk)==16: ## tetragonal II
        Cdict = {'C111':cijk[0], 'C112':cijk[1], 'C113':cijk[2], 'C114':0, 'C115':0, 'C116':cijk[3], 'C122':cijk[1], 'C123':cijk[4], 'C124':0, 'C125':0, 'C126':0, 'C133':cijk[5], 'C134':0, 'C135':0,
                 'C136':cijk[6], 'C144':cijk[7], 'C145':cijk[8], 'C146':0, 'C155':cijk[9], 'C156':0, 'C166':cijk[10], 'C222':cijk[0], 'C223':cijk[2], 'C224':0, 'C225':0, 'C226':-cijk[3], 'C233':cijk[5], 'C234':0,
                 'C235':0, 'C236':-cijk[6], 'C244':cijk[9], 'C245':-cijk[8], 'C246':0, 'C255':cijk[7], 'C256':0, 'C266':cijk[10], 'C333':cijk[11], 'C334':0, 'C335':0, 'C336':0, 'C344':cijk[12], 'C345':0,
                 'C346':0, 'C355':cijk[12], 'C356':0, 'C366':cijk[13], 'C444':0, 'C445':0, 'C446':cijk[14], 'C455':0, 'C456':cijk[15], 'C466':0, 'C555':0, 'C556':-cijk[14], 'C566':0, 'C666':0}
    elif len(cijk)==14: ## trigonal (rhombohedral) I
        Cdict = {'C111':cijk[0], 'C112':cijk[1], 'C113':cijk[2], 'C114':cijk[3], 'C115':0, 'C116':0, 'C122':(cijk[0]+cijk[1]-cijk[10]), 'C123':cijk[4], 'C124':cijk[5], 'C125':0, 'C126':0, 'C133':cijk[6], 'C134':cijk[7], 'C135':0,
                 'C136':0, 'C144':cijk[8], 'C145':0, 'C146':0, 'C155':cijk[9], 'C156':(cijk[3]+3*cijk[5])/2, 'C166':(3*cijk[10]-2*cijk[0]-cijk[1])/4, 'C222':cijk[10], 'C223':cijk[2], 'C224':(-cijk[3]-2*cijk[5]), 'C225':0, 'C226':0, 'C233':cijk[6], 'C234':(-cijk[7]),
                 'C235':0, 'C236':0, 'C244':cijk[9], 'C245':0, 'C246':0, 'C255':cijk[8], 'C256':(cijk[3]-cijk[5])/2, 'C266':(2*cijk[0]-cijk[1]-cijk[10])/4, 'C333':cijk[11], 'C334':0, 'C335':0, 'C336':0, 'C344':cijk[12], 'C345':0,
                 'C346':0, 'C355':cijk[12], 'C356':cijk[7], 'C366':(cijk[2]-cijk[4])/2, 'C444':cijk[13], 'C445':0, 'C446':0, 'C455':(-cijk[13]), 'C456':(cijk[9]-cijk[8])/2, 'C466':cijk[5], 'C555':0, 'C556':0, 'C566':0, 'C666':0}
    else:
        raise ValueError(f"len(cijk)={len(cijk)}, expected 56 (triclinic), 32 (monoclinic), 20 (orthorhombic), 16 (tetragonal II), or 14 (trigonal/rhombohedral I) values")
    if isinstance(sum(Cdict.values()),sp.Expr):
        C3 = np.empty((6,6,6), dtype=object)
    else:
        C3 = np.empty((6,6,6))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                ii,jj,kk = tuple(sorted([i+1,j+1,k+1]))
                C3[i,j,k] = Cdict["C"+str(ii)+str(jj)+str(kk)]
    if not voigt:
        C3 = UnVoigt(C3)
    return C3
        
### Convert to and from Voigt notation:
VoigtIndices = [0,4,8,5,2,1]
UnVoigtIndices = [0,5,4,5,1,3,4,3,2]

def Voigt(elasticC):
    '''Converts Voigt-symmetric tensors of ranks 2, 4, 6, and 8 (such as strain/stress tensors or 2nd, 3rd, and 4th order elastic constants) into Voigt notation.
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
        print(f'not implemented for array of dimension {CVoigt.ndim}, returning input')
    return CVoigt

def UnVoigt(CVoigt):
    '''Converts tensors of ranks 1, 2, 3, and 4 (such as strain/stress tensors or 2nd, 3rd, and 4th order elastic constants) from Voigt to conventional tensor notation.
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
        print(f'not implemented for array of dimension {elasticC.ndim}, returning input')
    return elasticC
    
## check for Voigt symmetry
def CheckVoigt(tensor):
    '''Quick and dirty check for Voigt symmetry: only checks UnVoigt(Voigt(tensor).T)==tensor, so don't rely on it entirely.
    Input must be a (numpy array) tensor of rank 2, 4, 6, or 8.'''
    return np.all(UnVoigt(Voigt(tensor).T)==tensor)
    
def CheckReflectionSymmetry(elasticC2,strict=False):
    '''Check for reflection symmetry of the z-plane assuming the tensor of second order elastic constants provided has been rotated into
     the coordinates to be checked. By default, we check for the slightly weaker condition where non-vanishing c34 and c35 are allowed
     (since they drop out of the differential equations for screw/edge dislocations. Set strict=True to check for true reflection symmetry.'''
    if len(elasticC2)==3:
        C2=Voigt(elasticC2)
    else:
        C2=elasticC2
    test = np.abs(C2/C2[3,3])
    testsum = test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4]
    if strict:
        testsum += test[2,3] + test[2,4]
    out = False
    if testsum < 1e-12:
        out = True
    return out

##### generate tensors of elastic compliances
def elasticS2(elasticC2,voigt=False):
    '''Generates the tensor of second order elastic compliances using a second order elastic constants tensor, elasticC2, as input data.
    Input may be in Voigt or in Cartesian notation, output notation is controlled by Boolean option 'voigt' (default=False, i.e. Cartesian).'''
    sprimetos = np.diag([1,1,1,0.5,0.5,0.5])
    if len(elasticC2)==3:
        C2=Voigt(elasticC2)
    else:
        C2=elasticC2
    if elasticC2.dtype.kind == 'O':
        ### need to convert to sympy matrix in order to call sympy's symbolic matrix inversion
        result = np.array(sp.simplify(sp.Matrix(C2).inv()))
    else:
        result = np.linalg.inv(np.asarray(C2, dtype=float))
    S2 = np.dot(sprimetos,np.dot(result,sprimetos))
    if not voigt:
        S2 = UnVoigt(S2)
    return S2
        
def elasticS3(elasticS2,elasticC3,voigt=False):
    '''Generates the tensor of third order elastic compliances using the second order elastic compliances tensor, elasticS2, and the third order elastic constants tensor, elasticC3, as input data.
    Input may be in Voigt or in Cartesian notation, output notation is controlled by Boolean option 'voigt' (default=False, i.e. Cartesian).'''
    if len(elasticS2)==3:
        S2=Voigt(elasticS2)
    else:
        S2=elasticS2
    if len(elasticC3)==3:
        C3=Voigt(elasticC3)
    else:
        C3=elasticC3
    if elasticS2.dtype.kind == 'O':
        S3 = np.zeros((6,6,6),dtype=object)
    else:
        S3 = np.zeros((6,6,6))
    # need to sum indices 3,4,5 twice below, therefore:
    S2P = np.dot(S2,np.diag([1,1,1,2,2,2]))
    S3 = -np.dot(S2P,np.dot(S2P,np.dot(C3,S2P.T)))
    if not voigt:
        S3 = UnVoigt(S3)
    return S3
    
class strain_poly:
    '''This class computes polynomials in y which depend on elastic constants and where y parametrizes infinitesimal deformations/strains.
    These can subsequently be used to compute elastic constants by fitting these polynomials to results from DFT calculations done in some third party software package.
    To initialize the class, a sympy symbol y (=sp.symbols('y') by default) as well as a keyword specifying the crystal symmetry must be provided, i.e. one of 'iso', 'cubic', 'hcp', 'tetr', 'tetr2', 'trig', 'orth', 'mono', or 'tric';
    'fcc' and 'bcc' are synonymous with 'cubic' (default). Tensors of 2nd and 3rd order elastic constants in symbolic Voigt notation are subsequently generated as attributes C2, C3.
    Method generate_poly([a1,a2,a3,a4,a5,a6],order=3) then computes the polynomial to the specified order, where a1-a6 represent infinitesimal strain in Voigt notation
    and they must be expressions in symbol y.'''
    def __init__(self,y=sp.symbols('y'),sym='cubic'):
        self.y = y
        self.sym = sym
        C11, C12, C44, C13, C33, C66 = sp.symbols('C11 C12 C44 C13 C33 C66')
        C111, C112, C123, C144, C166, C456, C113, C133, C155, C222, C333, C344, C366 = sp.symbols('C111 C112 C123 C144 C166 C456 C113 C133 C155 C222 C333 C344 C366')
        if sym=='iso':
            self.C2 = elasticC2(c12=C12,c44=C44,voigt=True)
            self.C3 = elasticC3(c123=C123,c144=C144,c456=C456,voigt=True)
        elif sym in ('cubic', 'fcc', 'bcc'):
            self.C2 = elasticC2(c11=C11,c12=C12,c44=C44,voigt=True)
            self.C3 = elasticC3(c111=C111,c112=C112,c123=C123,c144=C144,c166=C166,c456=C456,voigt=True)
        elif sym == 'hcp':
            self.C2 = elasticC2(c11=C11,c12=C12,c44=C44, c13=C13, c33=C33,voigt=True)
            self.C3 = elasticC3(c111=C111,c112=C112,c123=C123,c144=C144,c113=C113,c133=C133,c155=C155,c222=C222,c333=C333,c344=C344,voigt=True)
        elif sym == 'tetr':
            self.C2 = elasticC2(c11=C11,c12=C12,c44=C44, c13=C13, c33=C33, c66=C66,voigt=True)
            self.C3 = elasticC3(c111=C111,c112=C112,c123=C123,c144=C144,c166=C166,c456=C456,c113=C113,c133=C133,c155=C155,c333=C333,c344=C344, c366=C366,voigt=True)
        elif sym=='trig':
            self.C2 = elasticC2(cij=sp.symbols('C11,C12,C13,C14,C33,C44'),voigt=True)
            self.C3 = elasticC3(cijk=sp.symbols('C111,C112,C113,C114,C123,C124,C133,C134,C144,C155,C222,C333,C344,C444'),voigt=True)
        elif sym=='tetr2':
            self.C2 = elasticC2(cij=sp.symbols('C11,C12,C13,C16,C33,C44,C66'),voigt=True)
            self.C3 = elasticC3(cijk=sp.symbols('C111,C112,C113,C116,C123,C133,C136,C144,C145,C155,C166,C333,C344,C366,C446,C456'),voigt=True)
        elif sym=='orth':
            self.C2 = elasticC2(cij=sp.symbols('C11,C12,C13,C22,C23,C33,C44,C55,C66'),voigt=True)
            self.C3 = elasticC3(cijk=sp.symbols('C111,C112,C113,C122,C123,C133,C144,C155,C166,C222,C223,C233,C244,C255,C266,C333,C344,C355,C366,C456'),voigt=True)
        elif self.sym=='mono':
            self.C2 = elasticC2(cij=sp.symbols('C11,C12,C13,C15,C22,C23,C25,C33,C35,C44,C46,C55,C66'),voigt=True)
            self.C3 = elasticC3(cijk=sp.symbols('C111,C112,C113,C115,C122,C123,C125,C133,C135,C144,C146,C155,C166,C222,C223,C225,C233,C235,C244,C246,C255,C266,C333,C335,C344,C346,C355,C366,C445,C456,C555,C566'),voigt=True)
        elif self.sym=='tric':
            self.C2 = elasticC2(cij=sp.symbols('C11,C12,C13,C14,C15,C16,C22,C23,C24,C25,C26,C33,C34,C35,C36,C44,C45,C46,C55,C56,C66'),voigt=True)
            self.C3 = elasticC3(cijk=sp.symbols('C111,C112,C113,C114,C115,C116,C122,C123,C124,C125,C126,C133,C134,C135,C136,C144,C145,C146,C155,C156,C166,C222,C223,C224,C225,C226,C233,C234,C235,C236,C244,C245,C246,C255,C256,C266,C333,C334,C335,C336,C344,C345,C346,C355,C356,C366,C444,C445,C446,C455,C456,C466,C555,C556,C566,C666'),voigt=True)
        else:
            raise ValueError(f"sym={self.sym} not implemented")
        self.alpha=delta
        self.strain=np.zeros((3,3))
        self.poly=0
        
    def __repr__(self):
        return f"{({'y':self.y, 'sym':self.sym})}"
            
    def generate_alpha(self,epsilon=[0,0,0,0,0,0],preserve_volume=False):
        '''Computes the (symmetric) deformation matrix in Cartesian coordinates that would lead to infinitesimal strain epsilon,i.e.
        alpha=1+epsilon, where epsilon is first converted to Cartesian notation. If preserve_volume is set to True, alpha will subsequently be rescaled by its determinant.'''
        alpha = UnVoigt(np.asarray(epsilon)+np.array([1,1,1,0,0,0]))
        if preserve_volume:
            determinant=sp.det(sp.Matrix(alpha))
            if isinstance(determinant,sp.Float):
                alpha=alpha/(float(determinant)**(1/3))
            else:
                alpha=alpha/sp.cbrt(sp.factor(determinant))
        self.alpha=alpha
        return alpha
        
    def generate_strain(self,alpha=None):
        '''Computes strain from deformation matrix alpha (both in Cartesian notation).'''
        if alpha is None:
            alpha=self.alpha
        strain = (np.dot(alpha.T,alpha) - delta)/2
        self.strain=strain
        return strain
    
    def generate_poly(self,epsilon=[0,0,0,0,0,0],order=3,preserve_volume=False,make_eta=True,P=0):
        '''Computes the polynomial in y whose coefficients are the elastic constants, where epsilon must depend on y parametrizing infinitesimal deformations/strains.
        If option preserve_volume is set to True, the provided strain will be changed to preserve volume like so:
            epsilon_new=((1+epsilon)/(det(1+epsilon))**(1/3) -1), where epsilon is first converted to Cartesian notation and epsilon_new is converted back to Voigt notation.
        If option make_eta is set to False, no alpha is computed and we assume the users input epsilon is already the finite strain (thereby ignoring the preserve_volume option).
        By default, the polynomial is computed for the Gibbs free energy (option P=0); set pressure P to compute for the Helmholtz free energy.'''
        if make_eta:
            alpha = self.generate_alpha(epsilon,preserve_volume)
            etac = Voigt(self.generate_strain(alpha))*np.array([1,1,1,2,2,2])
        else:
            etac = np.asarray(epsilon)*np.array([1,1,1,2,2,2])
            alpha = np.zeros((3,3))
        phi1 = -P*sum(etac[:3])
        phi2 = np.dot(np.dot(self.C2,etac),etac)/2 + phi1
        phi2 = sp.simplify(sp.series(phi2,self.y,n=order+1))
        if order>2 and np.any(alpha!=delta): ## skip phi3 if order<3
            phi3 = np.dot(np.dot(np.dot(self.C3,etac),etac),etac)/6
            phi3 = sp.simplify(sp.series(phi3,self.y,n=order+1))
            out = phi2 + phi3
        elif order>1:
            out = phi2
        else:
            raise ValueError("Error: expected 'order'>=2")
        self.poly = roundcoeff(out) ## get rid of tiny numbers due to rounding errors, i.e. set numerical coefficients <10^-12 to 0.0
        return self.poly
