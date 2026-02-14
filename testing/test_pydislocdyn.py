#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - Feb. 13, 2026
'''This script implements regression testing for PyDislocDyn. Required argument: 'folder' containing old results.
   (To freshly create a folder to compare to later, run from within an empty folder with argument 'folder' set to '.')
   For additional options, call this script with '--help'.'''
import os
import sys
import subprocess
import pathlib
import difflib
import lzma
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
dir_path = pathlib.Path(__file__).resolve().parents[1] / 'pydislocdyn'

from pydislocdyn.metal_data import fcc_metals, bcc_metals, hcp_metals, tetr_metals, \
    ISO_l, c111, all_metals
from pydislocdyn.utilities import parse_options, str2bool, isclose, compare_df, OPTIONS
from pydislocdyn import read_2dresults, Ncores, Voigt, strain_poly, writeallinputfiles, readinputfile
from pydislocdyn.phononwind import OPTIONS as OPTIONS_drag
import numpy as np ## import pydislocdyn first as it will set the openmp thread number
import sympy as sp
import pandas as pd
if Ncores>1:
    from joblib import Parallel, delayed

OPTIONS_LT = OPTIONS | {"Ntheta2":int, "scale_by_mu":str, "bccslip":str, "hcpslip":str} 
OPTIONS_drag = OPTIONS_drag | {"use_iso":str2bool, "bccslip":str, "hcpslip":str, "skiptransonic":str2bool,
                     "Nq":int, "NphiX":int, "rmin":float, "rmax":float}
runtests = 'all'
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
fastapprox=True ## set to False to include terms that are (close to) zero in acc_screw disloc. field
vRF_resolution=50
vRF_fast=True

## TODO: instead of porting to argparse, switch all tests here to pytest functions
## maybe keep parse_options and OPTIONS for this script only an djust port the package?
OPTIONS = {"runtests":str, "metals_iso":str, "metals":str, "verbose":str2bool, "skip_calcs":str2bool,
           "Nbeta_LT":int, "Ntheta_LT":int, "P":sp.Symbol, "volpres":str2bool}
OPTIONS |= OPTIONS_LT | OPTIONS_drag
OPTIONS.pop('Ntheta2') ## using Ntheta in this script instead
OPTIONS |= {"fastapprox":str2bool, "vRF_resolution":int, "vRF_fast":str2bool}

def printtestresult(success,countfails=0):
    '''print passed/failed message depending on Boolean input'''
    if success:
        print("----------\nPASSED\n----------\n")
    else:
        print("----------\nFAILED\n----------\n")
        countfails += 1
    return countfails

def readfile(fname):
    '''reads a text file (or xz compressed text file) and returns a list of its lines'''
    fname = str(fname)
    if fname[-4:] == '.tex' and not pathlib.Path(fname).is_file():
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

def runscript(scriptname,args,logfname):
    '''Run script "scriptname" as a subprocess passing a list of command line arguments "args" and saving its stdout to a file "logfname"'''
    out = -1
    with open(logfname, 'w', encoding="utf8") as logfile:
        command = [pathlib.Path(dir_path,scriptname)]
        if sys.platform=='win32':
            command = ["python",pathlib.Path(dir_path,scriptname)]
        with subprocess.Popen(command+args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
            for line in subproc.stdout:
                sys.stdout.write(line)
                logfile.write(line)
            subproc.wait()
            out = subproc.returncode
    return out

if __name__ == '__main__':
    tests_avail=['all', 'aver', 'dragiso', 'drag', 'LT', 'acc', 'misc']
    cwd =pathlib.Path.cwd()
    tmppydislocdyn = pathlib.Path("temp_pydislocdyn")
    no_failed_tests = 0
    if len(sys.argv) > 1:
        oldglobals = globals().copy()
        starthelpwith=(f"\nUsage: {sys.argv[0]} <options> <folder_to_compare_cwd_to>\n\n"
            f"use option --runtests to run only one of these available tests: {tests_avail[1:]} (default: 'all').\n")
        old, kwargs = parse_options(sys.argv[1:],OPTIONS,globals(),starthelpwith=starthelpwith)
        if len(old)==0:
            raise ValueError("missing one argument: folder containing old results")
        old = pathlib.Path(old[0])
        if not isinstance(phononwind_opts, dict):
            raise ValueError(f"{phononwind_opts=} is not a dictionary, please check the formatting of the according command-line option!")
        phononwind_opts.update(kwargs)
        NEWopts = globals().keys()-oldglobals.keys() ## pass options which we haven't previously defined but that the user has set
        LTopts = {i:globals()[i] for i in NEWopts if i in OPTIONS_LT.keys()}
        dragopts = {i:globals()[i] for i in NEWopts if i in OPTIONS_drag.keys()}
        LTopts = [f" --{i}={j}" for i,j in LTopts.items()]
        dragopts = [f" --{i}={j}" for i,j in dragopts.items()]
    else:
        raise ValueError("missing one argument: folder containing old results")
    if old.exists():
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
        metals = ''
        if runtests in ['all', 'drag']:
            from pydislocdyn.dragcoeff_semi_iso import metal as all_metals
        for i in all_metals:
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
            if runscript('polycrystal_averaging.py',[],'poly.log')!=0:
                success=False
        else: print("skipping test 'aver' as requested")
        print(f"checking {fname}:")
        if not diff(pathlib.Path(old,fname),pathlib.Path(cwd,fname),verbose=verbose):
            success = False
        no_failed_tests = printtestresult(success,no_failed_tests)
    ############### TEST dragiso ##########################################
    if runtests in ['all', 'dragiso']:
        success = True
        if not skip_calcs:
            print("running test 'dragiso' ...")
            commandargs = dragopts + [f'--{Nbeta=}',f'--{Ncores=}',f'--{use_exp_Lame=}',f'--phononwind_opts={phononwind_opts}',f'--{NT=}',f'{metals_iso}']
            if runscript("dragcoeff_iso.py",commandargs,'dragiso.log')!=0:
                success=False
        else: print("skipping test 'dragiso' as requested")
        metals_iso = metals_iso.split()
        print(f"\ncomparing dragiso results for: {metals_iso}")
        for X in metals_iso:
            dragname = "drag"
            if NT>1 and os.access(pathlib.Path(old,f"{dragname}_T_{X}.dat"), os.R_OK):
                dragname = "drag_T"
            if not diff(pathlib.Path(old,f"{dragname}_{X}.dat"),pathlib.Path(cwd,f"{dragname}_{X}.dat"),verbose=verbose):
                if not verbose: print(f"{dragname}_{X} differs")
                success = False
            for ch in ["screw","edge","aver"]:
                fname = pathlib.Path("BofSig_iso",f"B_of_sigma_{X}{ch}.csv.xz")
                f1 = pd.read_csv(pathlib.Path(old,fname))
                f2 = pd.read_csv(pathlib.Path(cwd,fname))
                if not (result:=isclose(f1,f2)):
                    print(f"{fname} differs")
                    success=False
                    if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
        fname = "drag_iso_fit.txt"
        if not diff(pathlib.Path(old,fname),pathlib.Path(cwd,fname),verbose=verbose):
            if not verbose: print(f"{fname} differs")
            success=False
        no_failed_tests = printtestresult(success,no_failed_tests)
    ############### TEST drag #############################################
    if runtests in ['all', 'drag']:
        success = True
        drag_folder = pathlib.Path('drag')
        if not skip_calcs:
            drag_folder.mkdir(exist_ok=True)
            print("running test 'drag' ...")
            os.chdir(pathlib.Path(cwd,drag_folder))
            commandargs = dragopts + [f'--{Ncores=}',f'--{skiptransonic=}',f'--{use_exp_Lame=}',f'--{use_iso}',f'--{hcpslip=!s}',f'--{bccslip=!s}',f'--phononwind_opts={phononwind_opts}',f'--{Ntheta=}',f'--{Nbeta=}',f'--{NT=}',f'{metals}']
            if runscript("dragcoeff_semi_iso.py",commandargs,'dragsemi.log')!=0:
                success=False
            os.chdir(cwd)
        else: print("skipping test 'drag' as requested")
        print(f"\ncomparing drag results for: {metal_list}")
        for X in metal_list:
            dragname = "drag_anis"
            if NT>1 and os.access(pathlib.Path(old,drag_folder,f"{dragname}_T_{X}.dat.xz"), os.R_OK):
                dragname = "drag_anis_T"
            f1 = read_2dresults(pathlib.Path(old,drag_folder,f"{dragname}_{X}.dat.xz"))
            f2 = read_2dresults(pathlib.Path(cwd,drag_folder,f"{dragname}_{X}.dat.xz"))
            if not (result:=isclose(f1,f2)):
                print(f"{drag_folder}/{dragname}_{X}.dat.xz differs")
                success=False
                if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
            for ch in ["screw","edge","aver"]:
                fname = pathlib.Path(drag_folder,"BofSig_anis",f"B_of_sigma_{X}{ch}.csv.xz")
                f1 = pd.read_csv(pathlib.Path(old,fname))
                f2 = pd.read_csv(pathlib.Path(cwd,fname))
                if not (result:=isclose(f1,f2)):
                    print(f"{fname} differs")
                    success=False
                    if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
        fname = "drag_semi_iso_fit.txt"
        if not diff(pathlib.Path(old,drag_folder,fname),pathlib.Path(cwd,drag_folder,fname),verbose=verbose):
            if not verbose: print(f"{drag_folder}/{fname} differs")
            success=False
        no_failed_tests = printtestresult(success,no_failed_tests)
    ############### TEST LT ###############################################
    if runtests in ['all', 'LT']:
        success = True
        LT_folders = [pathlib.Path(f'LT_{scale_by_mu}'), pathlib.Path(f"LT_{scale_by_mu}","fromfiles")]
        fname = "vcrit.dat"
        if not skip_calcs:
            LT_folders[1].mkdir(parents=True,exist_ok=True)
            print("running test 'LT' ...")
            os.chdir(pathlib.Path(cwd,LT_folders[0]))
            LTopts = LTopts + [f'--Ntheta={Ntheta_LT}',f'--Ntheta2={Ntheta}',f'--Nbeta={Nbeta_LT}',f'--{Ncores=}',f'--{Nphi=}',f'--{hcpslip=!s}',f'--{bccslip=!s}',f'--{scale_by_mu=!s}']
            if runscript("linetension_calcs.py",LTopts+[f'{metals}'],'LT.log')!=0:
                success=False
            os.chdir(pathlib.Path(cwd,LT_folders[1]))
            filelist = sorted(pathlib.Path(os.pardir,"temp_pydislocdyn").glob("*"))
            if runscript("linetension_calcs.py",LTopts+filelist,'LT.log')!=0:
                success=False
            os.chdir(cwd)
        else: print("skipping test 'LT' as requested")
        print(f"\ncomparing LT results for: {metal_list}")
        for folder in LT_folders:
            for X in metal_list:
                f1 = read_2dresults(pathlib.Path(old,folder,f"LT_{X}.dat.xz"))
                f2 = read_2dresults(pathlib.Path(cwd,folder,f"LT_{X}.dat.xz"))
                if not (result:=isclose(f1,f2)):
                    print(f"{folder}/{X} differs")
                    success=False
                    if verbose and f1.shape==f2.shape: print(f"{compare_df(f1,f2)}\n")
            if not diff(pathlib.Path(old,folder,fname),pathlib.Path(cwd,folder,fname),verbose=verbose):
                if not verbose: print(f"{folder}/{fname} differs")
                success=False
        no_failed_tests = printtestresult(success,no_failed_tests)
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
            tmppydislocdyn.mkdir(exist_ok=True)
            os.chdir(tmppydislocdyn)
            writeallinputfiles()
            os.chdir("..")
            uij_acc_screw = {}
            acc_screw = {}
            uij_acc_edge = {}
            acc_edge = {}
            print(f"calculating accelerating screw dislocation fields for {metal_acc_screw}")
            for X in metal_acc_screw:
                acc_screw[X] = readinputfile(pathlib.Path("temp_pydislocdyn",X),Nphi=50)
                acc_screw[X].computevcrit()
                vel=0.9*acc_screw[X].vcrit_screw
                if verbose:
                    print(f"{X}: testing for constant acceleration")
                acc_screw[X].plotdisloc(vel/acc_screw[X].ct,a=1e14,fastapprox=fastapprox)
                uij_acc_screw[X] = pd.DataFrame(acc_screw[X].uij_acc_screw_aligned[2,0],index=acc_screw[X].r,columns=acc_screw[X].phi/np.pi)
                uij_acc_screw[X].index.name="r[burgers]"
                uij_acc_screw[X].columns.name="phi[pi]"
                uij_acc_screw[X].to_csv(pathlib.Path(cwd,f"uij_acc_screw_{X}.csv.xz"),compression='xz')
                ## another dynamic solution:
                ## assume l(t) = adot*t**3/6 ## (i.e. acceleration starts at 0 and increases at rate adot from t>0)
                adot = 6.2e25 ## time-derivative of acceleration, acc is initially zero at time t=0
                def eta(x):
                    '''returns time as a function of position x for the special case of a constant acceleration rate adot.'''
                    return np.sign(x)*np.cbrt(6*abs(x)/adot)
                def etapr(x):
                    '''returns the derivative of eta(x) (units: one over velocity)'''
                    return eta(x)/(3*x)
                time = np.sqrt(2*vel/adot) ## vel=adot*t**2/2, time=t(vel)
                distance = adot*time**3/6 ## distance covered by the core at time 'time'
                acc = adot*time ## current acceleration at time 'time'
                if verbose:
                    print("testing fully dynamic solution: assume acceleration starts at 0 and increases at rate a-dot from t>0.")
                    print(f"time to reach {vel=:.2f}m/s with a-dot={adot:.2e}m/s^3: t(v)={time:.2e}s")
                    print(f"current acceleration at time t(v) is: a(t(v))={acc:.2e}m/s^2")
                    print(f"dislocation moved distance d={distance:.2e}m when it reached velocity v\n")
                acc_screw[X].plotdisloc(a=None,eta_kw=eta,etapr_kw=etapr,t=time,shift=distance,beta=vel/acc_screw[X].ct,fastapprox=fastapprox)
                uij_acc_screw[X] = pd.DataFrame(acc_screw[X].uij_acc_screw_aligned[2,0],index=acc_screw[X].r,columns=acc_screw[X].phi/np.pi)
                uij_acc_screw[X].index.name="r[burgers]"
                uij_acc_screw[X].columns.name="phi[pi]"
                uij_acc_screw[X].to_csv(pathlib.Path(cwd,f"uij_acc_screw_alt_{X}.csv.xz"),compression='xz')
            print(f"calculating accelerating edge dislocation fields for {metal_acc_edge}")
            for X in metal_acc_edge:
                acc_edge[X] = readinputfile(pathlib.Path("temp_pydislocdyn",X),Nphi=25)
                acc_edge[X].computevcrit()
                acc_edge[X].plotdisloc(0.9*acc_edge[X].vcrit_edge/acc_edge[X].ct,a=1e14,character='edge',component=[1,1],Nr=25)
                uij_acc_edge[X] = pd.DataFrame(acc_edge[X].uij_acc_edge_aligned[1,1],index=acc_edge[X].r,columns=acc_edge[X].phi/np.pi)
                uij_acc_edge[X].index.name="r[burgers]"
                uij_acc_edge[X].columns.name="phi[pi]"
                uij_acc_edge[X].to_csv(pathlib.Path(cwd,f"uij_acc_edge_{X}.csv.xz"),compression='xz')
        else: print("skipping test 'acc' as requested")
        print("\ncomparing acc results")
        for X in metal_acc_screw:
            for fname in ("uij_acc_screw_", "uij_acc_screw_alt_"):
                fending = ".csv.xz"
                if not pathlib.Path(old,f"{fname}{X}{fending}").is_file():
                    fending = ".csv" # support reading old uncompressed files
                f1 = pd.read_csv(pathlib.Path(old,f"{fname}{X}{fending}"),index_col=0)
                f2 = pd.read_csv(pathlib.Path(cwd,f"{fname}{X}.csv.xz"),index_col=0)
                if not (result:=isclose(f1,f2)):
                    print(f"{fname}{X}.csv.xz differs")
                    success=False
                    if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
        for X in metal_acc_edge:
            f1 = pd.read_csv(pathlib.Path(old,f"uij_acc_edge_{X}{fending}"),index_col=0)
            f2 = pd.read_csv(pathlib.Path(cwd,f"uij_acc_edge_{X}.csv.xz"),index_col=0)
            if not (result:=isclose(f1,f2)):
                print(f"uij_acc_edge_{X}.csv.xz differs")
                success=False
                if verbose and f1.shape==f2.shape: print(compare_df(f1,f2))
        no_failed_tests = printtestresult(success,no_failed_tests)
    ############### TEST misc ###############################################
    if runtests in ['all', 'misc']:
        success = True
        crystalsyms = ['iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric']
        if not skip_calcs:
            print("running test 'misc' ...")
            tmppydislocdyn.mkdir(exist_ok=True)
            os.chdir(tmppydislocdyn)
            writeallinputfiles()
            os.chdir("..")
            Y = {}
            print(f"calculating limiting velocities, Rayleigh speeds, and radiation-free velocities for {metal_list}")
            for X in metal_list:
                Y[X] = readinputfile(pathlib.Path("temp_pydislocdyn",X),Ntheta=5)
                Y[X].computevcrit()
                Y[X].findvcrit_smallest()
                Y[X].findRayleigh()
                Y[X].find_vRF(fast=vRF_fast,resolution=vRF_resolution)
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
                Y[X].computeuij(0.5,r=Y[X].r)
                Y[X].alignuij()
                np.savez_compressed(f"u_{X}.npz",uk_05small=Y[X].uk_aligned[:,:,::10,::10],uij_05small=Y[X].uij_aligned[:,:,:,::10,::10])
            print("calculating various deformations within the 'strain_poly' class")
            y = sp.Symbol('y')
            eta1, eta2, eta3, eta4, eta5, eta6 = sp.symbols('eta1 eta2 eta3 eta4 eta5 eta6')
            etaS = np.array([eta1, eta2, eta3, eta4, eta5, eta6])
            def maincomputations(sym):
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
            if Ncores>1:
                Parallel(n_jobs=Ncores)(delayed(maincomputations)(sym) for sym in crystalsyms)
            else:
                [maincomputations(sym) for sym in crystalsyms]
        else: print("skipping tests 'misc' as requested")
        print("\ncomparing misc results")
        for X in metal_list:
            fname = X+"props.txt"
            if not diff(pathlib.Path(old,fname),pathlib.Path(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
            fname = f"u_{X}.npz"
            if pathlib.Path(old,fname).is_file():
                u_results1 = np.load(pathlib.Path(old,fname))
            else:
                u_results1 = {'uk_05small':np.load(pathlib.Path(old,f'uk_05small_{X}.npy')),'uij_05small':np.load(pathlib.Path(old,f'uij_05small_{X}.npy'))}
            u_results2 = np.load(pathlib.Path(cwd,fname))
            for aname in ['uk_05small','uij_05small']:
                f1 = u_results1[aname]
                f2 = u_results2[aname]
                if not isclose(f1,f2):
                    print(f"{fname}/{aname} differs")
                    success=False
        for sym in crystalsyms:
            fname = f"deformations_results_{sym}.txt"
            if not diff(pathlib.Path(old,fname),pathlib.Path(cwd,fname),verbose=verbose):
                if not verbose: print(f"{fname} differs")
                success=False
        no_failed_tests = printtestresult(success,no_failed_tests)
        
    assert no_failed_tests==0, f"{no_failed_tests} tests failed"
