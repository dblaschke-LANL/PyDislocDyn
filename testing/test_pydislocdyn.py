#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - May 20, 2024
'''This script implements regression testing for PyDislocDyn. Required argument: 'folder' containing old results.
   (To freshly create a folder to compare to later, run from within an empty folder with argument 'folder' set to '.')
   For additional options, call this script with '--help'.'''
import os
import sys
import difflib
import lzma
import numpy as np
import sympy as sp
import pandas as pd

dir_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)
dir_path = os.path.join(dir_path,'pydislocdyn')

from pydislocdyn.metal_data import fcc_metals, bcc_metals, hcp_metals, tetr_metals, ISO_l, c111
from pydislocdyn.utilities import parse_options, str2bool, isclose, compare_df
from pydislocdyn import read_2dresults, Ncores, Voigt, strain_poly, writeallinputfiles, readinputfile
from pydislocdyn.linetension_calcs import OPTIONS as OPTIONS_LT
from pydislocdyn.dragcoeff_semi_iso import OPTIONS as OPTIONS_drag

runtests = 'all' ## allowed values: all, LT, drag, dragiso, aver
skip_calcs = False
verbose = False
## drag and dragiso options:
Nbeta = 7
phononwind_opts={'maxrec':4,'target_accuracy':1e-2}
NT = 1
use_exp_Lame=True
## dragiso only options:
metals_iso = 'Cu Fe'
# drag and LT options:
metals = "Al Mo Ti Sn"
Ntheta = 4
bccslip='all'
hcpslip='all'
## drag only options:
skiptransonic = True
use_iso=False
## LT only options:
Nbeta_LT = 50
Ntheta_LT = 200
Nphi = 500
scale_by_mu = 'exp'
## misc only options:
P=0 ## pressure in strain_poly test
volpres=False ## set to True to compute volume preserving version of the strains

OPTIONS = {"runtests":str, "metals_iso":str, "metals":str, "verbose":str2bool, "skip_calcs":str2bool,
           "Nbeta_LT":int, "Ntheta_LT":int, "P":sp.Symbol, "volpres":str2bool}
OPTIONS |= OPTIONS_LT | OPTIONS_drag
OPTIONS.pop('Ntheta2') ## using Ntheta in this script instead

def printtestresult(success):
    '''print passed/failed message depending on Boolean input'''
    if success:
        print("----------\nPASSED\n----------\n")
    else: print("----------\nFAILED\n----------\n")

def readfile(fname):
    '''reads a text file (or xz compressed text file) and returns a list of its lines'''
    if fname[-4:] == '.tex' and not os.path.isfile(fname):
        fname = fname[:-4]+'.txt' ## allow comparing to old versions of pydislocdyn
    if fname[-3:] == '.xz':
        with lzma.open(fname,"rt") as f1:
            f1lines = f1.readlines()
    else:
        with open(fname,"r", encoding="utf8") as f1:
            f1lines = f1.readlines()
    if fname[-4:] == '.tex' or fname[-4:] == '.txt': ## ignore expected changes in LaTeX code
        f1lines = [x for x in f1lines if 'tabular' not in x and 'table' not in x and 'averages' not in x]
    return f1lines

def removewhitespace(somestring):
    '''replaces excessive whitespace with ' ' in the middle of a string, and removes leading/trailing whitespace (space and tab characters only)'''
    string1 = somestring.strip()
    string2 = string1.replace("\t"," ").replace("  "," ")
    while string2!=string1:
        string1 = string2
        string2 = string1.replace("\t"," ").replace("  "," ")
    return string2

def diff(f1,f2,verbose=True):
    '''Compares two text files'''
    f1lines = readfile(f1)
    f1lines = [removewhitespace(i) for i in f1lines]
    f2lines = readfile(f2)
    f2lines = [removewhitespace(i) for i in f2lines]
    thediff = difflib.unified_diff(f1lines, f2lines, fromfile=f1,tofile=f2,lineterm="",n=0)
    equal = f1lines==f2lines
    if verbose and not equal:
        for line in thediff:
            print(line.strip("\n"))
    return equal

def round_list(lst,ndigits=2):
    '''rounds all floats in a nested list'''
    if isinstance(lst,float):
        return float(round(lst,ndigits))
    if isinstance(lst,list):
        return [round_list(i,ndigits) for i in lst]
    return lst

if __name__ == '__main__':
    tests_avail=['all', 'aver', 'dragiso', 'drag', 'LT', 'acc', 'misc']
    cwd = os.getcwd()
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print(f"\nUsage: {sys.argv[0]} <options> <folder_to_compare_cwd_to>\n")
            print(f"use option --runtests to run only one of these available tests: {tests_avail[1:]} (default: 'all').\n")
            print("use option --fromfile to read additional options from a file")
            print("available options:")
            for key, OPTk in OPTIONS.items():
                print(f'--{key}={OPTk}')
            sys.exit()
        oldglobals = globals().copy()
        old, kwargs = parse_options(sys.argv[1:],OPTIONS,globals())
        if len(old)==0:
            raise ValueError("missing one argument: folder containing old results")
        old = old[0]
        phononwind_opts.update(kwargs)
        NEWopts = globals().keys()-oldglobals.keys() ## pass options which we haven't previously defined but that the user has set
        LTopts = {i:globals()[i] for i in NEWopts if i in OPTIONS_LT.keys()}
        dragopts = {i:globals()[i] for i in NEWopts if i in OPTIONS_drag.keys()}
        LTopts = [f" --{i}={j}" for i,j in LTopts.items()]
        dragopts = [f" --{i}={j}" for i,j in dragopts.items()]
    else:
        raise ValueError("missing one argument: folder containing old results")
    if os.path.exists(old):
        print(f"comparing to {old}\n")
    else:
        raise ValueError(f"folder {old} does not exist")
    if runtests not in tests_avail:
        raise ValueError(f"{runtests=} unknown; please select from {tests_avail}")
    if bccslip == 'all':
        slipkw_bcc = ['110', '112', '123']
    else:
        slipkw_bcc = [bccslip]
    if hcpslip == 'all':
        slipkw_hcp = ['basal','prismatic','pyramidal']
    else:
        slipkw_hcp=[hcpslip]
    if metals_iso == 'all':
        if use_exp_Lame:
            metals_iso_temp = sorted(list(ISO_l.keys()))
        else:
            metals_iso_temp = sorted(list(c111.keys()))
        metals_iso=''
        for i in metals_iso_temp:
            metals_iso += i+' '
        metals_iso = metals_iso.strip()
    if metals == 'all':
        from dragcoeff_semi_iso import metal as metals_temp
        metals = ''
        for i in metals_temp:
            metals += i+' '
        metals = metals.strip()
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
        fname = 'averaged_elastic_constants.tex'
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
            os.system(os.path.join(dir_path,"dragcoeff_iso.py")+f'{"".join(dragopts)} --{Nbeta=} --{Ncores=} --{use_exp_Lame=} --phononwind_opts="{phononwind_opts}" --{NT=} "{metals_iso}" | tee dragiso.log')
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
            dragopts = f'{"".join(dragopts)} --{Ncores=} --{skiptransonic=} --{use_exp_Lame=} --{use_iso=} --{hcpslip=} --{bccslip=} --phononwind_opts="{phononwind_opts}" "{metals}"'
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
                if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
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
            LTopts = f"{''.join(LTopts)} --Ntheta={Ntheta_LT} --Ntheta2={Ntheta} --Nbeta={Nbeta_LT} --{Nphi=} --{hcpslip=} --{bccslip=} --{scale_by_mu=} "
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
                    if verbose and f1.shape==f2.shape: print(f"{compare_df(f1,f2)}\n")
            if not diff(os.path.join(old,folder,fname),os.path.join(cwd,folder,fname),verbose=verbose):
                if not verbose: print(f"{folder}/{fname} differs")
                success=False
        printtestresult(success)
    ############### TEST acc  ###############################################
    if runtests in ['all', 'acc']:
        metal_acc_screw = []
        metal_acc_edge = []
        for X in metal_list:
            if X in fcc_metals or "basal" in X or "prismatic" in X or "pyramidal" in X or X in tetr_metals:
                metal_acc_screw.append(X)
            if "112" in X or "basal" in X or "prismatic" in X or X in tetr_metals:
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
                if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
        for X in metal_acc_edge:
            f1 = pd.read_csv(os.path.join(old,f"uij_acc_edge_{X}.csv"),index_col=0)
            f2 = pd.read_csv(os.path.join(cwd,f"uij_acc_edge_{X}.csv"),index_col=0)
            if not (result:=isclose(f1,f2)):
                print(f"uij_acc_edge_{X}.csv differs")
                success=False
                if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
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
                    np.set_printoptions(precision=2,suppress=True)
                    logfile.write(repr(Y[X]))
                    logfile.write("\n\ntheta:\n")
                    logfile.write('\n'.join(map("{:.6f}".format,Y[X].theta)))
                    logfile.write(f'\nvcrit(theta)={Y[X].vcrit_all[1]}')
                    logfile.write(f'\nvcrit_smallest={Y[X].vcrit_smallest:.2f}')
                    logfile.write(f'\nvRayleigh(theta)={Y[X].Rayleigh}')
                    logfile.write(f'\nvRF={round_list(Y[X].vRF)}')
            print("calculating various dislocation fields")
            for X in metal_list:
                Y[X].plotdisloc(0.5,nogradient=True,component=2,savefig=False)
                np.save(f'uk_05small_{X}',Y[X].uk_aligned[:,:,::10,::10])
                Y[X].computeuij(0.5,r=Y[X].r)
                Y[X].alignuij()
                np.save(f'uij_05small_{X}',Y[X].uij_aligned[:,:,:,::10,::10])
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
                for i,strni in enumerate(strain):
                    polynom[i] = poly.generate_poly(strni,preserve_volume=volpres,P=P)
                    alpha[i] = poly.alpha
                    strain[i] = Voigt(poly.strain)
                with open(f"deformations_results_{sym}.txt","w", encoding="utf8") as deffile:
                    deffile.write(f"{phi=}\n")
                    deffile.write("\nbelow we list alphas and corresponding phi2+phi3 polynomials; alphas are in Voigt notation (only consider symmetric cases)\n")
                    for i in range(len(strain)):
                        deffile.write(f"\nalpha[{i}]: {np.array2string(Voigt(alpha[i]), separator=', ')}\n")
                        deffile.write(f"poly[{i}]: {polynom[i]}\n")
        print("\ncomparing misc results")
        for X in metal_list:
            fname = X+"props.txt"
            if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
            for fname in [f'uk_05small_{X}.npy',f'uij_05small_{X}.npy']:
                f1 = np.load(os.path.join(old,fname))
                f2 = np.load(os.path.join(cwd,fname))
                if not isclose(f1,f2):
                    print(f"{fname} differs")
                    success=False
        for sym in crystalsyms:
            fname = f"deformations_results_{sym}.txt"
            if not diff(os.path.join(old,fname),os.path.join(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
        printtestresult(success)
