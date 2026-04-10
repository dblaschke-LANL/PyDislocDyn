#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - Apr. 10, 2026
'''This script implements several unit tests for PyDislocyn meant to be called by pytest.'''
import os
import sys
import pathlib
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
testpath = pathlib.Path(__file__).resolve().parents[0]
tmppydislocdyn = pathlib.Path(testpath,"temp_pydislocdyn")
tmppydislocdyn.mkdir(exist_ok=True)

import pydislocdyn as pydis
if pydis.usefortran:
    from pydislocdyn.subroutines import elastic_constants, utilities, various_subroutines
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import sympy as sp

def initialize_metals(metal_list=None):
    '''writes all input files to subfolder "temp_pydislocdyn", then reads a subset defined by the keywords in "metal_list",
       and returns a dictionary of the resulting metal_props class instances.'''
    os.chdir(tmppydislocdyn)
    pydis.writeallinputfiles()
    Y = {}
    if metal_list is None:
        metal_list = sorted(tmppydislocdyn.glob("*"))
    for X in metal_list:
        tmpY = pydis.crystals.readinputfile(X)
        Y[tmpY.name] = tmpY
    return Y

def initialize_dislocs(metal_list=None,Ntheta=2):
    '''writes all input files to subfolder "temp_pydislocdyn", then reads a subset defined by the keywords in "metal_list",
       and returns a dictionary of the resulting Dislocation class instances.'''
    os.chdir(tmppydislocdyn)
    pydis.writeallinputfiles(iso=True)
    pydis.writeallinputfiles()
    Y = {}
    if metal_list is None:
        metal_list = sorted(tmppydislocdyn.glob("*"))
        ## make sure we wrote all expected files: iso+3 slip systems for bcc and hcp, fcc and tetr
        ## are overwritten by anisotropic version; also missing iso data for K, so -1
        assert len(metal_list)>=len(pydis.metal_data.fcc_metals)+len(pydis.metal_data.tetr_metals)\
                +4*len(pydis.metal_data.hcp_metals)+4*len(pydis.metal_data.bcc_metals)-1
    for X in metal_list:
        tmpY = pydis.readinputfile(X,Ntheta=Ntheta)
        Y[tmpY.name] = tmpY
    return Y

def test_sound(metal_list=None):
    '''verify properties of calculated sound speeds'''
    Y = initialize_metals(metal_list=metal_list)
    for X in Y:
        Y[X].sumofsounds = 0
        for v in ([1,0,0],[0,1,0],[0,0,1]):
            sound = Y[X].computesound(v)
            if len(sound)==2:
                Y[X].sumofsounds += 2*min(sound)**2+max(sound)**2
            else:
                Y[X].sumofsounds += sum(np.array(sound)**2)
        Y[X].C2tracerho = np.trace(np.trace(pydis.UnVoigt(Y[X].C2),axis1=1,axis2=2))/Y[X].rho
        assert np.isclose(Y[X].C2tracerho,Y[X].sumofsounds) ## see Fitzgerald 1967 for details on this relation

def test_elasticconstants(metal_list=None,Ntheta=2):
    '''verify properties of elastic constants of anisotropic crystals'''
    Y = initialize_dislocs(metal_list=metal_list,Ntheta=Ntheta)
    for X in Y:
        if Y[X].Zener is not None:
            Y[X].AL = Y[X].anisotropy_index()
            Y[X].AL_Z = np.sqrt(5)*np.log((2+3*Y[X].Zener)*(3+2*Y[X].Zener)/(25*Y[X].Zener))
            assert np.isclose(Y[X].AL, Y[X].AL_Z)

        if X in pydis.metal_data.fcc_metals: # could include bcc here, but don't spend too much time on this test
            Y[X].clowest1 = round(Y[X].find_wavespeed(accuracy=1e-2)) # due to reduced accuracy, only expect correct to 1 m/s
            Y[X].clowest2 = np.sqrt(min(Y[X].cp,Y[X].c44)/Y[X].rho)
            Y[X].findvcrit_smallest()
            assert (np.isclose(Y[X].clowest1, round(Y[X].clowest2)) and np.isclose(Y[X].clowest2, Y[X].vcrit_smallest))

        testC = (12.3e9,4.5e9,6e9)
        a1 = pydis.convert_SOECiso(*testC[:2])
        assert (np.allclose(a1,pydis.convert_SOECiso(bulk=a1['bulk'],c44=a1['c44']))
                and np.allclose(a1,pydis.convert_SOECiso(lam=a1['c12'],bulk=a1['bulk']))
                and np.allclose(a1,pydis.convert_SOECiso(c12=a1['c12'],young=a1['young']))
                and np.allclose(a1,pydis.convert_SOECiso(c12=a1['c12'],poisson=a1['poisson']))
                and np.allclose(a1,pydis.convert_SOECiso(bulk=a1['bulk'],young=a1['young']))
                and np.allclose(a1,pydis.convert_SOECiso(bulk=a1['bulk'],poisson=a1['poisson']))
                and np.allclose(a1,pydis.convert_SOECiso(c44=a1['c44'],young=a1['young']))
                and np.allclose(a1,pydis.convert_SOECiso(c44=a1['c44'],poisson=a1['poisson']))
                and np.allclose(a1,pydis.convert_SOECiso(poisson=a1['poisson'],young=a1['young'])))

        a2 = pydis.convert_TOECiso(*testC)
        assert (np.allclose(a2,pydis.convert_TOECiso(l=a2['l'],m=a2['m'],n=a2['n']))
                and np.allclose(a2,pydis.convert_TOECiso(nu1=a2['nu1'],nu2=a2['nu2'],nu3=a2['nu3'])))

def test_isotropic():
    '''tests some propoerties in the isotropic limit'''
    iso = pydis.Dislocation(b=[1,0,0],n0=[0,1,0])
    iso.init_symbols()
    iso.vcrit=iso.computevcrit()
    iso.sound =iso.computesound([1,0,0])
    iso.compute_Lame()
    iso.init_sound()
    assert iso.rho*iso.vcrit['screw']**2-iso.c44==0
    assert sp.simplify(sp.Matrix(iso.sound)-sp.Matrix([iso.ct,iso.cl]))==sp.Matrix([0,0])
    assert np.prod(iso.vcrit['edge']*sp.sqrt(iso.rho))**2-iso.c44*iso.cl**2*iso.rho==0

def test_fcc():
    '''tests some propoerties of fcc crystals'''
    fcc = pydis.Dislocation(b=[1,1,0],n0=[-1,1,-1],sym='fcc')
    fcc.init_symbols()
    fcc.vcrit=fcc.computevcrit()
    fcc.sound =fcc.computesound([1,1,0])
    fcc.compute_Lame(scheme='voigt')
    fcc.init_all()
    assert fcc.bulk-(fcc.c11+2*fcc.c12)/3==0
    assert pydis.roundcoeff(sp.simplify(fcc.rho*fcc.vcrit['screw']**2 - 3*fcc.cp*fcc.c44/(fcc.c44+2*fcc.cp)),11)==0
    assert sp.simplify(sp.Matrix(fcc.sound)-fcc.vcrit['edge'])==sp.Matrix([0,0,0])

def test_hcp():
    '''tests some propoerties of fcc crystals'''
    # hcp (basal)
    hcp = pydis.Dislocation(b=[-2,1,1,0],n0=[0, 0, 0, 1],lat_a=1,lat_c=sp.symbols('c0',positive=True),Miller=True,sym='hcp')
    hcp.init_symbols()
    hcp.vcrit = hcp.computevcrit()
    hcp.cp = (hcp.c11-hcp.c12)/2
    assert pydis.roundcoeff(hcp.rho*hcp.vcrit['screw']**2-hcp.cp,10)==0

    # hcp (prismatic)
    hcp = pydis.Dislocation(b=[-2,1,1,0],n0=[-1, 0, 1, 0],lat_a=1,lat_c=sp.symbols('c0',positive=True),Miller=True,sym='hcp')
    hcp.init_symbols()
    hcp.vcrit = hcp.computevcrit()
    hcp.cp = (hcp.c11-hcp.c12)/2
    assert pydis.roundcoeff(sp.simplify(hcp.rho*(hcp.vcrit['edge'][0]**2+hcp.vcrit['screw']**2)-hcp.cp-hcp.c44),10)==0

    # hcp (pyrmidal)
    hcp = pydis.Dislocation(b=[-2,1,1,0],n0=[-1,0,1,1],lat_a=1,lat_c=sp.symbols('c0',positive=True),Miller=True,sym='hcp')
    hcp.init_symbols()
    hcp.vcrit = hcp.computevcrit()
    c11,c12,c44,c0 = (hcp.c11,hcp.c12,hcp.c44,hcp.cc)
    cp = (c11-c12)/2
    hcpresult = sp.simplify(sp.simplify(hcp.rho*hcp.vcrit['screw']**2) - sp.simplify(c44*cp*(3/4+c0**2)/(3/4*c44 + c0**2*cp)))
    assert abs(hcpresult.subs({c0:1.6,c44:1,c11:1.9,c12:0.9}))<1e-12

def test_disloc_props(metal_list=None,Ntheta=2):
    '''checks some propoerties of the steady-state dislocation displacement gradient field'''
    Y = initialize_dislocs(metal_list=metal_list,Ntheta=Ntheta)
    for X in Y:
        Y[X].alignC2()
        Y[X].computevcrit()
        if Y[X].sym=='iso':
            num_edge = num_screw = Y[X].ct ## Barnett routine is not called in the isotropic case
        else:
            num_edge = sorted(Y[X].vcrit_barnett[0,-1])[:2]
            num_screw = sorted(Y[X].vcrit_barnett[0,0])[:2]
        if not np.any(np.isclose(num_edge,Y[X].vcrit_edge,rtol=1e-02)):
            print(f"Warning: numerical accuracy of vcrit_edge for {X} may be less than 1%")
        if pydis.usefortran:
            ## check fortran implementation of stroh geometry also:
            t,m0,M,N,Cv = various_subroutines.strohgeometry(Y[X].b,Y[X].n0,Y[X].theta,Y[X].phi)
            assert np.allclose(m0,Y[X].m0.T)
            assert np.allclose(t,Y[X].t.T)
            assert np.allclose(Cv,Y[X].Cv)
            assert np.allclose(M,np.moveaxis(Y[X].M,-1,0))
            assert np.allclose(N,np.moveaxis(Y[X].N,-1,0))
        ## need high tolerance in assert statements since numerical barnett scheme is inaccurate in highly symmetric cases
        ## for screw disloc. with reflection symm. we have an analytic expression, so warn only for edge case
        ## where the reflection symm. routine also relies on numerical solutions (albeit a more accurate one we think)
        assert np.any(np.isclose(num_edge,Y[X].vcrit_edge,rtol=1.1e-01)), print('edge',X,num_edge,Y[X].vcrit_edge)
        assert np.any(np.isclose(num_screw,Y[X].vcrit_screw,rtol=1e-01)), print('screw',X,num_screw,Y[X].vcrit_screw)
        if pydis.CheckReflectionSymmetry(Y[X].C2_aligned[0]):
            Y[X].computeuij(0.5)
            trace_of_screw = np.trace(Y[X].uij[:,:,0]) # trace is zero for pure screw dislocations
            assert np.all(trace_of_screw<1e-15), print(X,trace_of_screw)
            if X in pydis.metal_data.fcc_metals: 
                vlim_edge = np.sqrt(min(Y[X].cp,Y[X].c44)/Y[X].rho)
                assert np.isclose(Y[X].vcrit_edge,vlim_edge)
    if pydis.usefortran:
        ## check that fortran implementation of averaging schemes matches the python implementations
        for X in Y:
            if Y[X].sym in ('cubic','fcc','bcc'):
                assert np.allclose(np.array(Y[X].compute_Lame(scheme='improved',roundto=0)[:2]),np.array(elastic_constants.kroeneraverage(Y[X].C2)))
            if X in ('Cu','Tibasal','Sn','Mo'): # check only one rep. for each symmetry (Mo is isotropic in contrast to Mo110 etc.)
                assert np.allclose(np.array(Y[X].compute_Lame(scheme='hill',roundto=0)[:2]),np.array(elastic_constants.hillaverage(Y[X].C2)))

def test_fortransubroutines():
    '''tests some of the fortran code, if it is available'''
    if not pydis.usefortran:
        print("\ntest_fortransubroutines(): cannot import fortran subroutines, therefore nothing to test")
        return
    # test inv()
    A = np.random.rand(9).reshape((3,3))
    Ainv = utilities.inv(A)
    assert np.all(np.abs(A@Ainv-pydis.utilities.delta)<1e-9) and np.all(np.abs(np.linalg.inv(A)-Ainv)<1e-9)
    # test linspace()
    assert sum(abs(np.linspace(0,1,11)-utilities.linspace(0,1,11)))<1e-15
    # test trapz() and cumtrapz()
    f = 21*(np.random.rand(10)-0.3)
    x = np.random.rand(10)
    assert np.allclose(trapezoid(f,x),utilities.trapz(f,x))
    assert np.allclose(cumulative_trapezoid(f,x,initial=0),utilities.cumtrapz(f,x))
    ## check that fortran implementations of elasticC2, elasticC3, and (un)voigt give the same results as the python version
    xtric = 0.01+20*np.random.rand(56)
    assert np.allclose(elastic_constants.elasticc2(xtric[:2],sym='iso'),pydis.elasticC2(c12=xtric[0],c44=xtric[1],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:3],sym='cubic'),pydis.elasticC2(c11=xtric[0],c12=xtric[1],c44=xtric[2],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:5],sym='hcp'),pydis.elasticC2(c11=xtric[0],c12=xtric[1],c13=xtric[2],c33=xtric[3],c44=xtric[4],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:6],sym='tetr'),pydis.elasticC2(c11=xtric[0],c12=xtric[1],c13=xtric[2],c33=xtric[3],c44=xtric[4],c66=xtric[5],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:6],sym='trig'),pydis.elasticC2(cij=xtric[:6],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:7],sym='tetr2'),pydis.elasticC2(cij=xtric[:7],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:9],sym='orth'),pydis.elasticC2(cij=xtric[:9],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:13],sym='mono'),pydis.elasticC2(cij=xtric[:13],voigt=True))
    assert np.allclose(elastic_constants.elasticc2(xtric[:21],sym='tric'),pydis.elasticC2(cij=xtric[:21],voigt=True))
    ##
    assert np.allclose(elastic_constants.elasticc3(xtric[:3],sym='iso'),pydis.elasticC3(c123=xtric[0],c144=xtric[1],c456=xtric[2],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:6],sym='cubic'),pydis.elasticC3(c111=xtric[0],c112=xtric[1],c123=xtric[2],c144=xtric[3],c166=xtric[4],c456=xtric[5],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:10],sym='hcp'),pydis.elasticC3(c111=xtric[0],c112=xtric[1],c113=xtric[2],c123=xtric[3],c133=xtric[4],c144=xtric[5],
                                                                                                       c155=xtric[6],c222=xtric[7],c333=xtric[8],c344=xtric[9],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:12],sym='tetr'),pydis.elasticC3(c111=xtric[0],c112=xtric[1],c113=xtric[2],c123=xtric[3],c133=xtric[4],c144=xtric[5],
                                                                                          c155=xtric[6],c166=xtric[7],c333=xtric[8],c344=xtric[9],c366=xtric[10],c456=xtric[11],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:14],sym='trig'),pydis.elasticC3(cijk=xtric[:14],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:16],sym='tetr2'),pydis.elasticC3(cijk=xtric[:16],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:20],sym='orth'),pydis.elasticC3(cijk=xtric[:20],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric[:32],sym='mono'),pydis.elasticC3(cijk=xtric[:32],voigt=True))
    assert np.allclose(elastic_constants.elasticc3(xtric,sym='tric'),pydis.elasticC3(cijk=xtric,voigt=True))
    ##
    A = np.random.rand(6)
    assert np.all(pydis.UnVoigt(A)==elastic_constants.unvgt_one(A))
    assert np.all(A==elastic_constants.vgt_two(elastic_constants.unvgt_one(A)))
    A = np.resize(np.random.rand(6**2),(6,6))
    assert np.all(pydis.UnVoigt(A)==elastic_constants.unvgt_two(A))
    assert np.all(A==elastic_constants.vgt_four(elastic_constants.unvgt_two(A)))
    A = np.resize(np.random.rand(6**3),(6,6,6))
    assert np.all(pydis.UnVoigt(A)==elastic_constants.unvgt_three(A))
    assert np.all(A==elastic_constants.vgt_six(elastic_constants.unvgt_three(A)))
