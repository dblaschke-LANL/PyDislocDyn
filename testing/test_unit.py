#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - Nov. 22, 2025
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
import numpy as np
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
    pydis.writeallinputfiles()
    Y = {}
    if metal_list is None:
        metal_list = sorted(tmppydislocdyn.glob("*"))
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
