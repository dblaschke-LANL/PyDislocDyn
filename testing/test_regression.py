#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Mar. 6, 2023 - Feb. 15, 2026
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
from pydislocdyn.utilities import str2bool, isclose, compare_df
from pydislocdyn import read_2dresults, Ncores, Voigt, strain_poly, writeallinputfiles, readinputfile
import numpy as np ## import pydislocdyn first as it will set the openmp thread number
import sympy as sp
import pandas as pd
if Ncores>1:
    from joblib import Parallel, delayed
    
cwd =pathlib.Path.cwd()

defaults_for_testing = {'P':0, ## pressure in strain_poly test
                        'volpres':False, ## set to True to compute volume preserving version of the strains
                        'fastapprox':True, ## set to False to include terms that are (close to) zero in acc_screw disloc. field
                        'vRF_resolution':50,
                        'vRF_fast':True}

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
    elif isinstance(metals, list) and isinstance(metals[0], str):
        metals = metals
    else:
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

def test_aver(old=None,skip_calcs=False,verbose=False):
    '''implements regression tests for frontend script polycrystal_averaging.py'''
    testfolder = cwd/"regressiontests"
    testfolder.mkdir(exist_ok=True)
    if old is None:
        old = testfolder
    else:
        old = cwd / old
    if verbose:
        print(f"{testfolder=}, baseline folder={old}")
    os.chdir(testfolder)
    fname = 'averaged_elastic_constants.tex'
    if not skip_calcs:
        print("running test 'aver' ...")
        assert runscript('polycrystal_averaging.py',[],'poly.log')==0
    else: print("skipping test 'aver' as requested")
    print(f"checking {fname}")
    assert diff(pathlib.Path(old,fname),pathlib.Path(testfolder,fname),verbose=verbose)
    os.chdir(cwd)

def test_dragiso(old=None,skip_calcs=False,verbose=False,metals='Cu Fe',**kwargs):
    '''implements regression tests for isotropic phonon drag calculations via frontend script dragcoeff_iso.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder = cwd/"regressiontests"
    testfolder.mkdir(exist_ok=True)
    if old is None:
        old = testfolder
    else:
        old = cwd / old
    if verbose:
        print(f"{testfolder=}, baseline folder={old}")
    os.chdir(testfolder)
    options = {'Nbeta': 7, 'use_exp_Lame': True, 'phononwind_opts': {'maxrec': 4, 'target_accuracy': 0.01}, 'NT': 1} # defaults for this test
    options['Ncores'] = Ncores
    options.update(kwargs)
    commandargs = convert_options(options)
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
    os.chdir(cwd)

def test_drag(old=None,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements regression tests for anisotropic phonon drag calculations via frontend script dragcoeff_semi_iso.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder = cwd/"regressiontests"
    testfolder.mkdir(exist_ok=True)
    if old is None:
        old = testfolder
    else:
        old = cwd / old
    if verbose:
        print(f"{testfolder=}, baseline folder={old}")
    os.chdir(testfolder)
    options = {'Nbeta': 7, 'use_exp_Lame': True, 'phononwind_opts': {'maxrec': 4, 'target_accuracy': 0.01}, 'NT': 1,
               'bccslip': 'all', 'hcpslip': 'all', 'Ntheta': 4} # defaults for this test
    options['Ncores'] = Ncores
    options.update(kwargs)
    commandargs = convert_options(options)
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
    os.chdir(cwd)
    

def test_LT(old=None,skip_calcs=False,verbose=False,metals='Al Mo Ti Sn',**kwargs):
    '''implements regression tests for line tension calculations via frontend script linetension_calcs.py,
       where folder "old" contains the baseline results; set to "None" to initialize a new baseline.'''
    testfolder = cwd/"regressiontests"
    testfolder.mkdir(exist_ok=True)
    if old is None:
        old = testfolder
    else:
        old = cwd / old
    if verbose:
        print(f"{testfolder=}, baseline folder={old}")
    os.chdir(testfolder)
    opts = {'Nbeta':50,'Ntheta':200,'Ntheta2':4,'Nphi':500,'scale_by_mu':'exp','bccslip':'all','hcpslip':'all'} # defaults for this test
    opts['Ncores'] = Ncores
    opts.update(kwargs)
    commandargs = convert_options(opts)
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
    os.chdir(cwd)
