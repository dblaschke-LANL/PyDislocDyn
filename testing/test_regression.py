#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - Feb. 16, 2026
'''This script implements regression testing for PyDislocDyn and is meant to be run with pytest.'''
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
from pydislocdyn.utilities import isclose
from pydislocdyn import read_2dresults, Ncores, Voigt, strain_poly, writeallinputfiles, readinputfile
from pydislocdyn.dragcoeff_semi_iso import metal as all_drag_metals
import numpy as np ## import pydislocdyn first as it will set the openmp thread number
import sympy as sp
import pandas as pd
if Ncores>1:
    from joblib import Parallel, delayed
    
cwd =pathlib.Path.cwd()
# all available metals (to be used if user requests 'all' in one of the tests below):
all_metals = " ".join(all_metals)

########## define helper functions for the tests below

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
    thediff = difflib.unified_diff(f1lines, f2lines, fromfile=str(f1),tofile=str(f2),lineterm="",n=0)
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

def convert_options(optiondict):
    '''takes a dictionary and converts it to a list of strings where the original key:value pairs are written as "--key=value"'''
    out = []
    for k,v in optiondict.items():
        out.append(f"--{k}={v}")
    return out

def expand_slipsystems(metals,bccslip='all',hcpslip='all'):
    '''takes a list of keyword-strings for metals and appends slip system names'''
    if isinstance(metals, str):
        metals = metals.split(" ")
    elif not (isinstance(metals, list) and isinstance(metals[0], str)):
        raise ValueError(f"epxected a string or list of strings but got {metals=}")
    out = []
    if bccslip == 'all':
        slipkw_bcc = ['110', '112', '123']
    else:
        slipkw_bcc = [bccslip]
    if hcpslip == 'all':
        slipkw_hcp = ['basal','prismatic','pyramidal']
    else:
        slipkw_hcp=[hcpslip]
    for X in metals:
        if X in bcc_metals:
            for kw in slipkw_bcc:
                out.append(X+kw)
        elif X in hcp_metals:
            for kw in slipkw_hcp:
                out.append(X+kw)
        else:
            out.append(X)
    return out

def prepare_testfolder(old,new,verbose=False):
    '''creates the required folders and chdir into the test folder'''
    testfolder = pathlib.Path(new , "regressiontests")
    testfolder.mkdir(parents=True,exist_ok=True)
    if old is None:
        old = testfolder
    else:
        old = pathlib.Path(new , old)
    if verbose:
        print(f"{testfolder=}, baseline folder={old}")
    os.chdir(testfolder)
    print(testfolder)
    print(old)
    return testfolder, old
    
########## tests:

def test_aver(old=None,new=cwd,skip_calcs=False,verbose=False):
    '''implements regression tests for frontend script polycrystal_averaging.py'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    fname = 'averaged_elastic_constants.tex'
    if not skip_calcs:
        print("running test 'aver' ...")
        assert runscript('polycrystal_averaging.py',[],'poly.log')==0
    else: print("skipping test 'aver' as requested")
    print(f"checking {fname}")
    assert diff(pathlib.Path(old,fname),pathlib.Path(testfolder,fname),verbose=verbose)
    os.chdir(new)

def test_dragiso(old=None,new=cwd,skip_calcs=False,verbose=False,metals='Cu Fe',**kwargs):
    '''implements regression tests for isotropic phonon drag calculations via frontend script dragcoeff_iso.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    options = {'Nbeta': 7, 'use_exp_Lame': True, 'phononwind_opts': {'maxrec': 4, 'target_accuracy': 0.01}, 'NT': 1} # defaults for this test
    options['Ncores'] = Ncores
    options.update(kwargs)
    commandargs = convert_options(options)
    if metals == 'all':
        if options['use_exp_Lame']:
            metals_temp = sorted(list(ISO_l.keys()))
        else:
            metals_temp = sorted(list(c111.keys()))
        metals=''
        for i in metals_temp:
            metals += i+' '
        metals = metals.strip()
    commandargs.append(f'{metals}')
    if not skip_calcs:
        print(f"running test 'dragiso' with {commandargs=} ...")
        assert runscript("dragcoeff_iso.py",commandargs,'dragiso.log')==0
    else: print("skipping test 'dragiso' as requested")
    metals = metals.split()
    print(f"\ncomparing dragiso results for: {metals}")
    for X in metals:
        dragname = "drag"
        if options['NT']>1:# and os.access(pathlib.Path(old,f"{dragname}_T_{X}.dat"), os.R_OK):
            dragname = "drag_T"
        assert diff(pathlib.Path(old,f"{dragname}_{X}.dat"),pathlib.Path(testfolder,f"{dragname}_{X}.dat"),verbose=verbose)
        for ch in ["screw","edge","aver"]:
            fname = pathlib.Path("BofSig_iso",f"B_of_sigma_{X}{ch}.csv.xz")
            f1 = pd.read_csv(pathlib.Path(old,fname))
            f2 = pd.read_csv(pathlib.Path(testfolder,fname))
            assert isclose(f1,f2,verbose=verbose)
    fname = "drag_iso_fit.txt"
    assert diff(pathlib.Path(old,fname),pathlib.Path(testfolder,fname),verbose=verbose)
    os.chdir(new)

def test_drag(old=None,new=cwd,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements regression tests for anisotropic phonon drag calculations via frontend script dragcoeff_semi_iso.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    options = {'Nbeta': 7, 'use_exp_Lame': True, 'phononwind_opts': {'maxrec': 4, 'target_accuracy': 0.01}, 'NT': 1,
               'bccslip': 'all', 'hcpslip': 'all', 'Ntheta': 4} # defaults for this test
    options['Ncores'] = Ncores
    options.update(kwargs)
    commandargs = convert_options(options)
    if metals == 'all':
        metals = " ".join(all_drag_metals)
    commandargs.append(f'{metals}')
    metals = expand_slipsystems(metals,bccslip=options['bccslip'],hcpslip=options['hcpslip'])
    drag_folder = pathlib.Path('drag')
    if not skip_calcs:
        drag_folder.mkdir(exist_ok=True)
        print(f"running test 'drag' with {commandargs=} ...")
        os.chdir(pathlib.Path(testfolder,drag_folder))
        assert runscript("dragcoeff_semi_iso.py",commandargs,'dragsemi.log')==0
        os.chdir(testfolder)
    else: print("skipping test 'drag' as requested")
    print(f"\ncomparing drag results for: {metals}")
    for X in metals:
        dragname = "drag_anis"
        if options['NT']>1 and os.access(pathlib.Path(old,drag_folder,f"{dragname}_T_{X}.dat.xz"), os.R_OK):
            dragname = "drag_anis_T"
        f1 = read_2dresults(pathlib.Path(old,drag_folder,f"{dragname}_{X}.dat.xz"))
        f2 = read_2dresults(pathlib.Path(testfolder,drag_folder,f"{dragname}_{X}.dat.xz"))
        assert isclose(f1,f2,verbose=verbose)
        for ch in ["screw","edge","aver"]:
            fname = pathlib.Path(drag_folder,"BofSig_anis",f"B_of_sigma_{X}{ch}.csv.xz")
            f1 = pd.read_csv(pathlib.Path(old,fname))
            f2 = pd.read_csv(pathlib.Path(testfolder,fname))
            assert isclose(f1,f2,verbose=verbose)
    fname = "drag_semi_iso_fit.txt"
    assert diff(pathlib.Path(old,drag_folder,fname),pathlib.Path(testfolder,drag_folder,fname),verbose=verbose)
    os.chdir(new)
    

def test_LT(old=None,new=cwd,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements regression tests for line tension calculations via frontend script linetension_calcs.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    opts = {'Nbeta':50,'Ntheta':200,'Ntheta2':4,'Nphi':500,'scale_by_mu':'exp','bccslip':'all','hcpslip':'all'} # defaults for this test
    opts['Ncores'] = Ncores
    opts.update(kwargs)
    commandargs = convert_options(opts)
    if metals=='all':
        metals = all_metals
    commandargs.append(f'{metals}')
    LT_folders = [pathlib.Path(f"LT_{opts['scale_by_mu']}"), pathlib.Path(f"LT_{opts['scale_by_mu']}","fromfiles")]
    fname = "vcrit.dat"
    if not skip_calcs:
        LT_folders[1].mkdir(parents=True,exist_ok=True)
        print(f"running test 'LT' with {commandargs=} ...")
        os.chdir(pathlib.Path(testfolder,LT_folders[0]))
        assert runscript("linetension_calcs.py",commandargs+[f'{metals}'],'LT.log')==0
        os.chdir(pathlib.Path(testfolder,LT_folders[1]))
        filelist = sorted(pathlib.Path(os.pardir,"temp_pydislocdyn").glob("*"))
        assert runscript("linetension_calcs.py",commandargs+filelist,'LT.log')==0
        os.chdir(testfolder)
    else: print("skipping test 'LT' as requested")
    metals = expand_slipsystems(metals,bccslip=opts['bccslip'],hcpslip=opts['hcpslip'])
    print(f"\ncomparing LT results for: {metals}")
    for folder in LT_folders:
        for X in metals:
            f1 = read_2dresults(pathlib.Path(old,folder,f"LT_{X}.dat.xz"))
            f2 = read_2dresults(pathlib.Path(testfolder,folder,f"LT_{X}.dat.xz"))
            assert isclose(f1,f2,verbose=verbose)
        assert diff(pathlib.Path(old,folder,fname),pathlib.Path(testfolder,folder,fname),verbose=verbose) is True
    os.chdir(new)

def test_acc(old=None,new=cwd,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements regression tests for accelerating dislocation solutions,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    opts = {'bccslip':'all','hcpslip':'all','fastapprox':True} # defaults for this test
    opts['Ncores'] = Ncores
    opts.update(kwargs)
    if metals=='all':
        metals = all_metals
    metal_acc_screw = []
    metal_acc_edge = []
    metal_list = expand_slipsystems(metals,bccslip=opts['bccslip'],hcpslip=opts['hcpslip'])
    for X in metal_list:
        if X in fcc_metals or "basal" in X or "prismatic" in X or "pyramidal" in X or X in tetr_metals:
            metal_acc_screw.append(X)
        if "112" in X or "basal" in X or "prismatic" in X or X in tetr_metals:
            metal_acc_edge.append(X)
    if not skip_calcs:
        print("running test 'acc' ...")
        tmppydislocdyn = pathlib.Path("temp_pydislocdyn")
        tmppydislocdyn.mkdir(exist_ok=True)
        os.chdir(tmppydislocdyn)
        writeallinputfiles()
        os.chdir("..")
        uij_acc_screw = {}
        acc_screw = {}
        uij_acc_edge = {}
        acc_edge = {}
        ## another dynamic solution:
        ## assume l(t) = adot*t**3/6 ## (i.e. acceleration starts at 0 and increases at rate adot from t>0)
        adot = 6.2e25 ## time-derivative of acceleration, acc is initially zero at time t=0
        def eta(x):
            '''returns time as a function of position x for the special case of a constant acceleration rate adot.'''
            return np.sign(x)*np.cbrt(6*abs(x)/adot)
        def etapr(x):
            '''returns the derivative of eta(x) (units: one over velocity)'''
            return eta(x)/(3*x)
        print(f"calculating accelerating screw dislocation fields for {metal_acc_screw}")
        for X in metal_acc_screw:
            acc_screw[X] = readinputfile(pathlib.Path("temp_pydislocdyn",X),Nphi=50)
            acc_screw[X].computevcrit()
            vel=0.9*acc_screw[X].vcrit_screw
            if verbose:
                print(f"{X}: testing for constant acceleration")
            acc_screw[X].plotdisloc(vel/acc_screw[X].ct,a=1e14,fastapprox=opts['fastapprox'])
            uij_acc_screw[X] = pd.DataFrame(acc_screw[X].uij_acc_screw_aligned[2,0],index=acc_screw[X].r,columns=acc_screw[X].phi/np.pi)
            uij_acc_screw[X].index.name="r[burgers]"
            uij_acc_screw[X].columns.name="phi[pi]"
            uij_acc_screw[X].to_csv(pathlib.Path(testfolder,f"uij_acc_screw_{X}.csv.xz"),compression='xz')
            time = np.sqrt(2*vel/adot) ## vel=adot*t**2/2, time=t(vel)
            distance = adot*time**3/6 ## distance covered by the core at time 'time'
            acc = adot*time ## current acceleration at time 'time'
            if verbose:
                print("testing fully dynamic solution: assume acceleration starts at 0 and increases at rate a-dot from t>0.")
                print(f"time to reach {vel=:.2f}m/s with a-dot={adot:.2e}m/s^3: t(v)={time:.2e}s")
                print(f"current acceleration at time t(v) is: a(t(v))={acc:.2e}m/s^2")
                print(f"dislocation moved distance d={distance:.2e}m when it reached velocity v\n")
            acc_screw[X].plotdisloc(a=None,eta_kw=eta,etapr_kw=etapr,t=time,shift=distance,beta=vel/acc_screw[X].ct,fastapprox=opts['fastapprox'])
            uij_acc_screw[X] = pd.DataFrame(acc_screw[X].uij_acc_screw_aligned[2,0],index=acc_screw[X].r,columns=acc_screw[X].phi/np.pi)
            uij_acc_screw[X].index.name="r[burgers]"
            uij_acc_screw[X].columns.name="phi[pi]"
            uij_acc_screw[X].to_csv(pathlib.Path(testfolder,f"uij_acc_screw_alt_{X}.csv.xz"),compression='xz')
        print(f"calculating accelerating edge dislocation fields for {metal_acc_edge}")
        for X in metal_acc_edge:
            acc_edge[X] = readinputfile(pathlib.Path("temp_pydislocdyn",X),Nphi=25)
            acc_edge[X].computevcrit()
            acc_edge[X].plotdisloc(0.9*acc_edge[X].vcrit_edge/acc_edge[X].ct,a=1e14,character='edge',component=[1,1],Nr=25)
            uij_acc_edge[X] = pd.DataFrame(acc_edge[X].uij_acc_edge_aligned[1,1],index=acc_edge[X].r,columns=acc_edge[X].phi/np.pi)
            uij_acc_edge[X].index.name="r[burgers]"
            uij_acc_edge[X].columns.name="phi[pi]"
            uij_acc_edge[X].to_csv(pathlib.Path(testfolder,f"uij_acc_edge_{X}.csv.xz"),compression='xz')
    else: print("skipping test 'acc' as requested")
    print("\ncomparing acc results")
    for X in metal_acc_screw:
        for fname in ("uij_acc_screw_", "uij_acc_screw_alt_"):
            fending = ".csv.xz"
            if not pathlib.Path(old,f"{fname}{X}{fending}").is_file():
                fending = ".csv" # support reading old uncompressed files
            f1 = pd.read_csv(pathlib.Path(old,f"{fname}{X}{fending}"),index_col=0)
            f2 = pd.read_csv(pathlib.Path(testfolder,f"{fname}{X}.csv.xz"),index_col=0)
            assert isclose(f1,f2,verbose=verbose)
    for X in metal_acc_edge:
        f1 = pd.read_csv(pathlib.Path(old,f"uij_acc_edge_{X}{fending}"),index_col=0)
        f2 = pd.read_csv(pathlib.Path(testfolder,f"uij_acc_edge_{X}.csv.xz"),index_col=0)
        assert isclose(f1,f2,verbose=verbose)
    os.chdir(new)

def test_misc(old=None,new=cwd,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements various regression tests for the dislocation class,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    opts = {'bccslip':'all','hcpslip':'all','fastapprox':True,'vRF_resolution':50,'vRF_fast':True} # defaults for this test
    opts['Ncores'] = Ncores
    opts.update(kwargs)
    if metals=='all':
        metals = all_metals
    metal_list = expand_slipsystems(metals,bccslip=opts['bccslip'],hcpslip=opts['hcpslip'])
    if not skip_calcs:
        print("running test 'misc' ...")
        tmppydislocdyn = pathlib.Path("temp_pydislocdyn")
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
            Y[X].find_vRF(fast=opts['vRF_fast'],resolution=opts['vRF_resolution'])
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
    else: print("skipping tests 'misc' as requested")
    print("\ncomparing misc results")
    for X in metal_list:
        fname = X+"props.txt"
        assert diff(pathlib.Path(old,fname),pathlib.Path(testfolder,fname),verbose=verbose)
        fname = f"u_{X}.npz"
        if pathlib.Path(old,fname).is_file():
            u_results1 = np.load(pathlib.Path(old,fname))
        else:
            u_results1 = {'uk_05small':np.load(pathlib.Path(old,f'uk_05small_{X}.npy')),'uij_05small':np.load(pathlib.Path(old,f'uij_05small_{X}.npy'))}
        u_results2 = np.load(pathlib.Path(testfolder,fname))
        for aname in ['uk_05small','uij_05small']:
            f1 = u_results1[aname]
            f2 = u_results2[aname]
            assert isclose(f1,f2)

def test_strainpoly(old=None,new=cwd,skip_calcs=False,verbose=False,**kwargs):
    '''implements regression tests for the strain_ppoly class,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder, old = prepare_testfolder(old,new,verbose)
    opts = {'P':0,'volpres':False} # defaults for this test
    opts['Ncores'] = Ncores
    opts.update(kwargs)
    crystalsyms = ['iso', 'cubic', 'hcp', 'tetr', 'trig', 'tetr2', 'orth', 'mono', 'tric']
    if not skip_calcs:
        print("running test 'strainpoly' ...")
        print("calculating various deformations within the 'strain_poly' class")
        y = sp.Symbol('y')
        eta1, eta2, eta3, eta4, eta5, eta6 = sp.symbols('eta1 eta2 eta3 eta4 eta5 eta6')
        etaS = np.array([eta1, eta2, eta3, eta4, eta5, eta6])
        def maincomputations(sym):
            poly=strain_poly(y=y,sym=sym)
            phi = poly.generate_poly(etaS,make_eta=False,P=opts['P'])
            alpha = {}
            polynom = {}
            strain = [[y,0,0,0,0,0],[y,y,y,0,0,0],[y,0,0,y,0,0],[0,0,0,y,y,y],[0,y,-y,y,y,0],[y,0,0,y,y,y],[0,0,0,y,0,0],[y,-y,0,0,0,y]]
            for i,strni in enumerate(strain):
                polynom[i] = poly.generate_poly(strni,preserve_volume=opts['volpres'],P=opts['P'])
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
    print("\ncomparing strainpoly results")
    for sym in crystalsyms:
        fname = f"deformations_results_{sym}.txt"
        assert diff(pathlib.Path(old,fname),pathlib.Path(testfolder,fname),verbose=verbose)
