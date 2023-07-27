#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - July 27, 2023
'''This script implements regression testing for PyDislocDyn. Required argument: 'folder' containing old results.
   (To freshly create a folder to compare to later, run from within an empty folder with argument 'folder' set to '.')
   For additional options, call this script with '--help'.'''
import os
import sys
import ast
import difflib
import lzma
import numpy as np
import sympy as sp
import pandas as pd

dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..")
sys.path.append(dir_path)

from metal_data import fcc_metals, bcc_metals, hcp_metals, tetr_metals, writeallinputfiles
from elasticconstants import Voigt, UnVoigt, strain_poly
from linetension_calcs import parse_options, str2bool, read_2dresults, Ncores, readinputfile

runtests = 'all' ## allowed values: all, LT, drag, dragiso, aver
skip_calcs = False
verbose = False
## drag and dragiso options:
Nbeta = 7
phononwind_opts="{'maxrec':4,'target_accuracy':1e-2}"
NT = 1
## dragiso only options:
metals_iso = 'Cu Fe'
# drag and LT options:
metals = "Al Mo Ti Sn"
Ntheta = 4
bccslip='all'
hcpslip='all'
## drag only options:
computevcrit_for_speed = 'auto'
use_exp_Lame=True
use_iso=False
## LT only options:
Nbeta_LT = 50
Ntheta_LT = 200
Nphi = 500
scale_by_mu = 'exp'
## misc only options:
P=0 ## pressure in strain_poly test
volpres=False ## set to True to compute volume preserving version of the strains

OPTIONS = {"runtests":str, "metals_iso":str, "metals":str, "verbose":str2bool, "Ncores":int, "phononwind_opts":ast.literal_eval, \
           "NT":int, "skip_calcs":str2bool, "use_exp_Lame":str2bool, "use_iso":str2bool, "bccslip":str, "hcpslip":str,\
           "Nbeta":int, "Ntheta":int, "Nbeta_LT":int, "Ntheta_LT":int, "Nphi":int, "scale_by_mu":str, "P":sp.Symbol, "volpres":str2bool}

def printtestresult(success):
    if success: print("----------\nPASSED\n----------\n")
    else: print("----------\nFAILED\n----------\n")

def readfile(fname):
    '''reads a text file (or xz compressed text file) and returns a list of its lines'''
    if fname[-3:] == '.xz':
        with lzma.open(fname,"rt") as f1:
            f1lines = f1.readlines()
    else:
        with open(fname,"r", encoding="utf8") as f1:
            f1lines = f1.readlines()
    return f1lines

def diff(f1,f2,verbose=True):
    '''Compares two text files'''
    f1lines = readfile(f1)
    f2lines = readfile(f2)
    thediff = difflib.unified_diff(f1lines, f2lines, fromfile=f1,tofile=f2,lineterm="",n=0)
    equal = (f1lines==f2lines)
    if verbose and not equal:
        for line in thediff:
            print(line.strip("\n"))
    return equal

def isclose(f1,f2):
    '''Returns True if all elements of arrays f1 and f2 are 'close' to one another and theirs shapes match, and False otherwise.'''
    out = False
    if f1.shape==f2.shape:
        out = np.allclose(f1,f2,equal_nan=True)
    return out

def round_list(lst,ndigits=2):
    '''rounds all floats in a nested list'''
    if isinstance(lst,float):
        return round(lst,ndigits)
    elif isinstance(lst,list):
        return [round_list(i,ndigits) for i in lst]
    else:
        return lst

if __name__ == '__main__':
    tests_avail=['all', 'aver', 'dragiso', 'drag', 'LT', 'acc', 'misc']
    cwd = os.getcwd()
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print(f"\nUsage: {sys.argv[0]} <options> <folder_to_compare_cwd_to>\n")
            print(f"use option --runtests to run only one of these available tests: {tests_avail[1:]} (default: 'all').\n")
            print("available options:")
            for key in OPTIONS:
                print(f'--{key}={OPTIONS[key]}')
            sys.exit()
        old = parse_options(sys.argv[1:],OPTIONS,globals())[0] ## allowed values: all, LT, drag, dragiso, aver
    else:
        raise ValueError("missing one argument: folder containing old results")
    if  os.path.exists(old):
        print(f"comparing to {old}\n")
    else:
        raise ValueError(f"folder {old} does not exist")
    if runtests not in tests_avail:
        raise ValueError(f"{runtests=} unknown; please select from {tests_avail}")
    if computevcrit_for_speed == 'auto' or computevcrit_for_speed > Ntheta: computevcrit_for_speed = Ntheta
    if bccslip == 'all':
        slipkw_bcc = ['110', '112', '123']
    else:
        slipkw_bcc = [bccslip]
    if hcpslip == 'all':
        slipkw_hcp = ['basal','prismatic','pyramidal']
    else:
        slipkw_hcp=[hcpslip]
    metal_list = []
    for X in metals.split():
        if X in bcc_metals:
            for kw in slipkw_bcc:
                metal_list.append(X+kw)
        elif X in hcp_metals:
            for kw in slipkw_hcp:
                metal_list.append(X+kw)
        else:
            metal_list.append(X)
    ############### TEST aver #############################################
    if runtests in ['all', 'aver']:
        success = True
        fname = 'averaged_elastic_constants.txt'
        if not skip_calcs:
            print("running test 'aver' ...")
            os.system(os.path.join(dir_path,'polycrystal_averaging.py'))
        else: print("skipping test 'aver' as requested")
        print(f"checking {fname}:")
        if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
            success = False
        printtestresult(success)
    ############### TEST dragiso ##########################################
    if runtests in ['all', 'dragiso']:
        success = True
        fname = "drag_iso_fit.txt"
        if not skip_calcs:
            print("running test 'dragiso' ...")
            os.system(os.path.join(dir_path,"dragcoeff_iso.py")+f" --{Nbeta=} --{Ncores=} --{phononwind_opts=} --{NT=} '{metals_iso}' | tee dragiso.log")
        else: print("skipping test 'dragiso' as requested")
        metals_iso = metals_iso.split()
        print(f"\ncomparing dragiso results for: {metals_iso}")
        for X in metals_iso:
            dragname = "drag"
            if NT>1 and os.access(os.path.join(old,f"{dragname}_T_{X}.dat"), os.R_OK):
                dragname = "drag_T"
            if not diff(os.path.join(old,f"{dragname}_{X}.dat"),os.path.join(cwd,f"{dragname}_{X}.dat"),verbose=verbose):
                if not verbose: print(f"{dragname}_{X} differs")
                success = False
        if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
            if not verbose: print(f"{fname} differs")
            success=False
        printtestresult(success)
    ############### TEST drag #############################################
    if runtests in ['all', 'drag']:
        success = True
        drag_folder = 'drag'
        if not skip_calcs:
            if not os.path.exists(drag_folder):
                os.mkdir(drag_folder)
            print("running test 'drag' ...")
            os.chdir(os.path.join(cwd,drag_folder))
            dragopts = f" --{Ncores=} --{computevcrit_for_speed=} --{use_exp_Lame=} --{use_iso=} --{hcpslip=} --{bccslip=} --{phononwind_opts=} '{metals}'"
            os.system(os.path.join(dir_path,"dragcoeff_semi_iso.py")+dragopts+f" --{Ntheta=} --{Nbeta=} --{NT=} | tee dragsemi.log")
            os.chdir(cwd)
        else: print("skipping test 'drag' as requested")
        print(f"\ncomparing drag results for: {metal_list}")
        for X in metal_list:
            dragname = "drag_anis"
            if NT>1 and os.access(os.path.join(old,drag_folder,f"{dragname}_T_{X}.dat.xz"), os.R_OK):
                dragname = "drag_anis_T"
            f1 = read_2dresults(os.path.join(old,drag_folder,f"{dragname}_{X}.dat.xz"))
            f2 = read_2dresults(os.path.join(cwd,drag_folder,f"{dragname}_{X}.dat.xz"))
            if not (result:=isclose(f1,f2)):
                print(f"{drag_folder}/{dragname}_{X}.dat.xz differs")
                success=False
                if verbose and f1.shape==f2.shape: print(f1.compare(f2))
        fname = "drag_semi_iso_fit.txt"
        if not diff(os.path.join(old,drag_folder,fname),os.path.join(cwd,drag_folder,fname),verbose=verbose):
            if not verbose: print(f"{drag_folder}/{fname} differs")
            success=False
        printtestresult(success)
    ############### TEST LT ###############################################
    if runtests in ['all', 'LT']:
        success = True
        LT_folders = [f'LT_{scale_by_mu}', os.path.join(f"LT_{scale_by_mu}","fromfiles")]
        fname = "vcrit.dat"
        if not skip_calcs:
            if not os.path.exists(LT_folders[0]):
                os.mkdir(LT_folders[0])
            if not os.path.exists(LT_folders[1]):
                os.mkdir(LT_folders[1])
            print("running test 'LT' ...")
            os.chdir(os.path.join(cwd,LT_folders[0]))
            LTopts = f" --Ntheta={Ntheta_LT} --Ntheta2={Ntheta} --Nbeta={Nbeta_LT} --{Nphi=} --{hcpslip=} --{bccslip=} --{scale_by_mu=} "
            os.system(os.path.join(dir_path,"linetension_calcs.py")+LTopts+f"'{metals}' | tee LT.log")
            os.chdir(os.path.join(cwd,LT_folders[1]))
            os.system(os.path.join(dir_path,"linetension_calcs.py")+LTopts+os.path.join("..","temp_pydislocdyn","")+"* | tee LT.log")
            os.chdir(cwd)
        else: print("skipping test 'LT' as requested")
        print(f"\ncomparing LT results for: {metal_list}")
        for folder in LT_folders:
            for X in metal_list:
                f1 = read_2dresults(os.path.join(old,folder,f"LT_{X}.dat.xz"))
                f2 = read_2dresults(os.path.join(cwd,folder,f"LT_{X}.dat.xz"))
                if not (result:=isclose(f1,f2)):
                    print(f"{folder}/{X} differs")
                    success=False
                    if verbose and f1.shape==f2.shape: print(f"{f1.compare(f2)}\n")
            if not diff(os.path.join(old,folder,fname),os.path.join(cwd,folder,fname),verbose=verbose):
                if not verbose: print(f"{folder}/{fname} differs")
                success=False
        printtestresult(success)
    ############### TEST acc  ###############################################
    if runtests in ['all', 'acc']:
        metal_acc_screw = []
        metal_acc_edge = []
        for X in metal_list:
            if X in fcc_metals or "basal" in X or "prismatic" in X or "pyramidal" in X:
                metal_acc_screw.append(X)
            if "112" in X or "basal" in X or "prismatic" in X:
                metal_acc_edge.append(X)
        success = True
        if not skip_calcs:
            print("running test 'acc' ...")
            if not os.path.exists("temp_pydislocdyn"):
                os.mkdir("temp_pydislocdyn")
            os.chdir("temp_pydislocdyn")
            writeallinputfiles()
            os.chdir("..")
            uij_acc_screw = {}
            acc_screw = {}
            uij_acc_edge = {}
            acc_edge = {}
            print(f"calculating accelerating screw dislocation fields for {metal_acc_screw}")
            for X in metal_acc_screw:
                acc_screw[X] = readinputfile(os.path.join("temp_pydislocdyn",X),Nphi=50)
                acc_screw[X].computevcrit()
                acc_screw[X].plotdisloc(0.9*acc_screw[X].vcrit_screw/acc_screw[X].ct,a=1e14,fastapprox=True)
                uij_acc_screw[X] = pd.DataFrame(acc_screw[X].uij_acc_screw_aligned[2,0],index=acc_screw[X].r,columns=acc_screw[X].phi/np.pi)
                uij_acc_screw[X].index.name="r[burgers]"
                uij_acc_screw[X].columns.name="phi[pi]"
                uij_acc_screw[X].to_csv(os.path.join(cwd,f"uij_acc_screw_{X}.csv"))
            print(f"calculating accelerating edge dislocation fields for {metal_acc_edge}")
            for X in metal_acc_edge:
                acc_edge[X] = readinputfile(os.path.join("temp_pydislocdyn",X),Nphi=25)
                acc_edge[X].computevcrit()
                acc_edge[X].plotdisloc(0.9*acc_edge[X].vcrit_edge/acc_edge[X].ct,a=1e14,character='edge',component=[1,1],Nr=25)
                uij_acc_edge[X] = pd.DataFrame(acc_edge[X].uij_acc_edge_aligned[1,1],index=acc_edge[X].r,columns=acc_edge[X].phi/np.pi)
                uij_acc_edge[X].index.name="r[burgers]"
                uij_acc_edge[X].columns.name="phi[pi]"
                uij_acc_edge[X].to_csv(os.path.join(cwd,f"uij_acc_edge_{X}.csv"))
        print("\ncomparing acc results")
        for X in metal_acc_screw:
            f1 = pd.read_csv(os.path.join(old,f"uij_acc_screw_{X}.csv"),index_col=0)
            f2 = pd.read_csv(os.path.join(cwd,f"uij_acc_screw_{X}.csv"),index_col=0)
            if not (result:=isclose(f1,f2)):
                print(f"uij_acc_screw_{X}.csv differs")
                success=False
                if verbose and f1.shape==f2.shape: print(f1.compare(f2))
        for X in metal_acc_edge:
            f1 = pd.read_csv(os.path.join(old,f"uij_acc_edge_{X}.csv"),index_col=0)
            f2 = pd.read_csv(os.path.join(cwd,f"uij_acc_edge_{X}.csv"),index_col=0)
            if not (result:=isclose(f1,f2)):
                print(f"uij_acc_edge_{X}.csv differs")
                success=False
                if verbose and f1.shape==f2.shape: print(f1.compare(f2))
        printtestresult(success)
    ############### TEST misc ###############################################
    if runtests in ['all', 'misc']:
        success = True
        crystalsyms = ['iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric']
        if not skip_calcs:
            print("running test 'misc' ...")
            if not os.path.exists("temp_pydislocdyn"):
                os.mkdir("temp_pydislocdyn")
            os.chdir("temp_pydislocdyn")
            writeallinputfiles()
            os.chdir("..")
            Y = {}
            print(f"calculating limiting velocities, Rayleigh speeds, and radiation-free velocities for {metal_list}")
            for X in metal_list:
                Y[X] = readinputfile(os.path.join("temp_pydislocdyn",X),Ntheta=5)
                Y[X].computevcrit()
                Y[X].findvcrit_smallest()
                Y[X].findRayleigh()
                Y[X].find_vRF()
                with open(X+"props.txt", "w", encoding="utf8") as logfile:
                    np.set_printoptions(precision=2)
                    logfile.write(Y[X].__repr__())
                    logfile.write("\n\ntheta:\n")
                    logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
                    logfile.write(f'\nvcrit(theta)={Y[X].vcrit_all[1]}')
                    logfile.write(f'\nvcrit_smallest={Y[X].vcrit_smallest:.2f}')
                    logfile.write(f'\nvRayleigh(theta)={Y[X].Rayleigh}')
                    logfile.write(f'\nvRF={round_list(Y[X].vRF)}')
            print("calculating various deformations within the 'strain_poly' class")
            y = sp.Symbol('y')
            eta1, eta2, eta3, eta4, eta5, eta6 = sp.symbols('eta1 eta2 eta3 eta4 eta5 eta6')
            etaS = np.array([eta1, eta2, eta3, eta4, eta5, eta6])
            for sym in crystalsyms:
                poly=strain_poly(y=y,sym=sym)
                phi = poly.generate_poly(etaS,make_eta=False,P=P)
                alpha = {}
                polynom = {}
                strain = [[y,0,0,0,0,0],[y,y,y,0,0,0],[y,0,0,y,0,0],[0,0,0,y,y,y],[0,y,-y,y,y,0],[y,0,0,y,y,y],[0,0,0,y,0,0],[y,-y,0,0,0,y]]
                for i in range(len(strain)):
                    polynom[i] = poly.generate_poly(strain[i],preserve_volume=volpres,P=P)
                    alpha[i] = poly.alpha
                    strain[i] = Voigt(poly.strain)
                with open(f"deformations_results_{sym}.txt","w", encoding="utf8") as deffile:
                    deffile.write(f"{phi=}\n")
                    deffile.write("\nbelow we list alphas and corresponding phi2+phi3 polynomials; alphas are in Voigt notation (only consider symmetric cases)\n")
                    for i in range(len(strain)):
                        deffile.write("\nalpha[{}]: {}\n".format(i,np.array2string(Voigt(alpha[i]), separator=', ')))
                        deffile.write(f"poly[{i}]: {polynom[i]}\n")
        print("\ncomparing misc results")
        for X in metal_list:
            fname = X+"props.txt"
            if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
        for sym in crystalsyms:
            fname = f"deformations_results_{sym}.txt"
            if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
        printtestresult(success)
