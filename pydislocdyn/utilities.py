#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Dec. 18, 2024
'''This module contains various utility functions used by other submodules.'''
#################################
import sys
import os
import shutil
import time
import multiprocessing
from fractions import Fraction
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import sympy as sp
##################
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
import matplotlib.pyplot as plt
##### use pdflatex and specify font through preamble:
if shutil.which('latex'):
    mpl.use("pgf")
    texpreamble = "\n".join([
          r"\usepackage[utf8x]{inputenc}",
          r"\usepackage[T1]{fontenc}",
          r"\DeclareUnicodeCharacter{2212}{-}",
          r"\IfFileExists{fouriernc.sty}",
          r"{\usepackage{fouriernc}}{}",
          r"\usepackage{amsmath}",
    ])
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": texpreamble,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": texpreamble,
    })
##################
plt.rc('font',**{'family':'serif','size':'11'})
plt.rcParams['font.serif'].insert(0,'Liberation Serif')
plt.rcParams['font.sans-serif'].insert(0,'Liberation Sans')
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
fntsettings = {'family':'serif', 'fontsize':11}
from matplotlib.ticker import AutoMinorLocator
##################
import pandas as pd
Ncpus = multiprocessing.cpu_count()
nonumba=False
usefortran = False
try:
    from numba import jit
except ImportError:
    nonumba=True
    from functools import partial
    def jit(func=None,forceobj=True,nopython=False):
        '''define a dummy decorator if numba is unavailable at runtime'''
        if func is None:
            return partial(jit, forceobj=forceobj,nopython=nopython)
        return func
try:
    ompthreads = None
    if "OMP_NUM_THREADS" not in os.environ: ## allow user-override by setting this var. before running the python code
        ompthreads = int(np.sqrt(Ncpus))
        while Ncpus/ompthreads != round(Ncpus/ompthreads):
            ompthreads -= 1 ## choose an optimal value (assuming joblib is installed), such that ompthreads*Ncores = Ncpus and ompthreads ~ Ncores
        os.environ["OMP_NUM_THREADS"] = str(ompthreads)
    import pydislocdyn.subroutines as fsub
    if fsub.version()>=20231205:
        usefortran = True
        if ompthreads is None: ompthreads = fsub.ompinfo() ## don't rely on ompinfo() after os.environ (does not work on every system)
    else:
        print("Error: the subroutines module is outdated, please re-compile with f2py or by calling pydislocdy.utilities.compilefortranmodule() and reloading pydislocdyn")
        ompthreads = 0
except ImportError:
    ompthreads = 0

try:
    from joblib import Parallel, delayed
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/max(2,ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = Ncpus = 1 ## must be 1 without joblib

dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)

def compilefortranmodule(buildopts=''):
    '''Compiles the Fortran subroutines if a Fortran compiler is available.
       Keyword 'buildopts' may be used to pass additional options to f2py.'''
    cwd = os.getcwd()
    if sys.version_info[:2]<=(3,11) and shutil.which('gfortran'):
        compilerflags = '--f90flags=-fopenmp -lgomp' # flags specific to gfortran, require f2py with (old) distutils backend
    elif sys.version_info[:2]>=(3,12):
        compilerflags = '--dep=openmp' # requires f2py with new meson backend (default in numpy>=2.0 and python>=3.12)
    else:
        compilerflags = '' ## when in doubt, build without OpenMP support
    if buildopts != '':
        compilerflags += f" {buildopts}"
    os.chdir(os.path.dirname(__file__))
    os.system(f'python -m numpy.f2py {compilerflags} -c subroutines.f90 -m subroutines')
    os.chdir(cwd)

def printthreadinfo(Ncores,ompthreads=ompthreads):
    '''print a message to screen informing whether joblib parallelization (Ncores) or OpenMP parallelization (ompthreads)
       or both are currently employed; also warn if imports of numba and/or subroutines failed.'''
    if Ncores > 1 and ompthreads == 0: # check if subroutines were compiled with OpenMP support
        print(f"using joblib parallelization with {Ncores} cores")
    elif Ncores > 1:
        print(f"Parallelization: joblib with {Ncores} cores and OpenMP with {ompthreads} threads")
    elif ompthreads > 0:
        print(f"using OpenMP parallelization with {ompthreads} threads")
    if nonumba: print("\nWARNING: cannot find just-in-time compiler 'numba', execution will be slower\n")
    if not usefortran:
        print("\nWARNING: module 'subroutines' not found, execution will be slower")
        print("call pydislocdyn.utilities.compilefortranmodule() or run 'python -m numpy.f2py -c subroutines.f90 -m subroutines'")
        print("to compile this module (OpenMP is also supported with appropriate compiler flags), then reload pydislocdyn")

def str2bool(arg):
    '''converts a string to bool'''
    if arg in ['True', 'true', '1', 't', 'yes', 'y']:
        out=True
    elif arg in ['False', 'false', '0', 'f', 'no', 'n']:
        out=False
    else:
        raise ValueError(f"cannot convert {arg} to bool")
    return out

def guesstype(arg):
    '''takes a string and tries to convert to int, float, bool, falling back to a string'''
    try:
        out = int(arg)
    except ValueError:
        try:
            out = float(arg)
        except ValueError:
            try:
                out = str2bool(arg)
            except ValueError:
                out = arg ## fall back to string
    return out

def loadinputfile(fname,optionmode=False):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py and returns its data as a dictionary.'''
    inputparams={}
    with open(fname,"r", encoding="utf8") as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            if line[0] != "#":
                if optionmode:
                    currentline = line.lstrip().rstrip().split('=')
                    currentline = [i.strip() for i in currentline]
                    currentline.insert(1,'=')
                else:
                    currentline = line.lstrip().rstrip().split()
                if len(currentline) > 2:
                    key = currentline[0]
                    if len(currentline)==3 or currentline[3]=='#':
                        value = currentline[2]
                        if value[-1] == '#':
                            value = value[:-1]
                    else:
                        value = currentline[2]
                        for i in range(len(currentline)-3):
                            addval = currentline[i+3]
                            if addval[0] == '#':
                                break
                            if value[-1] == '#':
                                value = value[:-1]
                                break
                            value += addval
                    inputparams[key] = value
    return inputparams

OPTIONS = {"Ncores":int, "Ntheta":int, "Nbeta":int, "skip_plots":str2bool, "Nphi":int} ## options used by 3 frontend scripts
def parse_options(arglist,optionlist,globaldict=globals(),starthelpwith=f"\nUsage: {sys.argv[0]} <options> <inputfile(s)>\n"):
    '''Search commandline arguments for known options to set by comparing to a list of keyword strings "optionlist".
    These will then override default variables. This function also returns a copy of 'arglist' stripped of all
    option calls for further processing (e.g. opening input files that were passed etc.). If options are to be read
    from a file "fname" (with format key = value), use option --fromfile=fname; all options will be processed in
    order of appearance even if an option appears more than once.'''
    out = arglist
    if '--help' in out:
        print(starthelpwith)
        print("use option --fromfile to read additional options from a file")
        print("available options (see code manual for details):")
        for key, OPTk in optionlist.items():
            print(f'--{key}={OPTk.__name__}')
        sys.exit()
    setoptions = [i for i in out if "--" in i and i[:2]=="--"]
    kwargs = {}
    for i in setoptions:
        out.remove(i)
        if "=" not in i: continue ## ignore options without assigned values
        key,val = i[2:].split("=")
        if key=="fromfile":
            print(f"getting options from file {val}:")
            optiondict = loadinputfile(val,optionmode=True)
            for fkey,fval in optiondict.items():
                if fkey in optionlist:
                    globaldict[fkey] = optionlist[fkey](fval)
                    print(f"setting {fkey}={globaldict[fkey]}")
                else:
                    kwargs[fkey] = guesstype(fval)
            print(f"done with options file {val}")
        elif key in optionlist:
            globaldict[key] = optionlist[key](val)
            print(f"setting {key}={globaldict[key]}")
        else:
            kwargs[key] = guesstype(val)
    time.sleep(1) ## avoid race conditions after changing global variables
    return (out,kwargs)

def showoptions(optionlist,globaldict=globals()):
    '''returns a python dictionary of all options that can be set by the user (showing default values for those that have not been set by the user)'''
    optiondict = {}
    for key in optionlist:
        optiondict[key] = globaldict.get(key)
    return optiondict

def str_to_array(arg,dtype=float):
    '''converts a string containing comma separated numbers to a numpy array of specified data type (floats by default).'''
    try:
        out = np.asarray(arg.split(','),dtype=dtype)
    except ValueError:
        out = arg.split(',')
        out = np.asarray([Fraction(x) for x in out],dtype=object)
    return out
    
def read_2dresults(fname):
    '''Read results (such as line tension or drag coefficient) from file fname and return a Pandas DataFrame where index=beta (or [temperature,beta]) and columns=theta.'''
    if os.access((newfn:=fname+'.xz'), os.R_OK):
        fname = newfn # new default
    elif os.access((newfn:=fname), os.R_OK):
        pass # old default
    elif os.access((newfn:=fname+'.gz'), os.R_OK):
        fname = newfn
    else: raise FileNotFoundError(f'tried {fname}.xz, {fname}, and {fname}.gz')
    out = pd.read_csv(fname,skiprows=1,index_col=0,sep='\t')
    try:
        out.columns = pd.to_numeric(out.columns)*np.pi
    except ValueError:
        out = pd.read_csv(fname,skiprows=1,index_col=[0,1],sep='\t')
        out.columns = pd.to_numeric(out.columns)*np.pi
    if len(out.index.names)==1:
        out.index.name='beta'
    out.columns.name='theta'
    return out
    
def isclose(f1,f2):
    '''Returns True if all elements of arrays f1 and f2 are 'close' to one another and their shapes match, and False otherwise.'''
    out = False
    if f1.shape==f2.shape:
        out = np.allclose(f1,f2,equal_nan=True)
    return out

def compare_df(f1,f2):
    '''Compares two pandas.DataFrames using the pandas.compare method, but ignoring rounding errors (i.e. everything numpy.isclose decides is close enough)'''
    if f1.shape != f2.shape:
        return f"Error: Cannot compare arrays with different shapes: {f1.shape=}, {f2.shape=}"
    if isinstance(f1, np.ndarray) or isinstance(f2, np.ndarray):
        f1 = pd.DataFrame(f1)
        f2 = pd.DataFrame(f2)
    if isinstance(f1, pd.Series) or isinstance(f2, pd.Series):
        themask = pd.Series(np.invert(np.isclose(f1,f2,equal_nan=True)),index=f1.index)
    else:
        themask = pd.DataFrame(np.invert(np.isclose(f1,f2,equal_nan=True)),index=f1.index,columns=f1.columns)
    f1masked = f1[themask]
    f2masked = f2[themask]
    return f1masked.compare(f2masked)

################################################
hbar = 1.0545718e-34
kB = 1.38064852e-23

delta = np.diag((1,1,1))

@jit(nopython=True)
def heaviside(x):
    '''step function with convention heaviside(0)=1/2'''
    return (np.sign(x)+1)/2

@jit(nopython=True)
def deltadistri(x,epsilon=2e-16):
    '''approximates the delta function as exp(-(x/epsilon)^2)/epsilon*sqrt(pi)'''
    return np.exp(-(x/epsilon)**2)/(epsilon*np.sqrt(np.pi))

def artan(x,y):
    '''returns a variation of np.arctan2(x,y): since numpys implementation jumps to negative values in 3rd and 4th quadrant, shift those by 2pi so that atan(tan(phi))=phi for phi=[0,2pi]'''
    out = np.arctan2(x,y)
    out += 2*np.pi*np.heaviside(-out,0)
    return out

def roundcoeff(x,acc=12):
    '''This function traverses a sympy expression x and rounds all floats to 'acc' digits within it.'''
    x = sp.S(x)
    for a in sp.preorder_traversal(x):
        if isinstance(a, sp.Float):
            x = x.subs(a, round(a, acc))
    return x

def rotaround(v,s,c):
    '''Computes the rotation matrix with unit vector 'v' as the rotation axis and s,c are the sin/cos of the angle.'''
    if isinstance(v,np.ndarray) and v.dtype == np.dtype('O'):
        vx = np.zeros((3,3),dtype=object)
    else:
        vx = np.zeros((3,3))
    vx[0,1] = v[2]
    vx[1,0] = -v[2]
    vx[0,2] = -v[1]
    vx[2,0] = v[1]
    vx[1,2] = v[0]
    vx[2,1] = -v[0]
    out = delta +s*vx + np.dot(vx,vx)*(1-c)
    return out

if nonumba:
    trapz = trapezoid
    cumtrapz = cumulative_trapezoid
else:
    @jit(nopython=True)
    def trapz(y,x):
        '''integrate over the last axis using the trapezoidal rule (i.e. equivalent to numpy.trapz(y,x,axis=-1))'''
        theshape = y.shape
        n = theshape[-1]
        f = y.T
        outar = np.zeros(theshape).T
        for i in range(n-1):
            outar[i] = (0.5*(f[i+1]+f[i])*(x[i+1]-x[i]))
        return np.sum(outar.T,axis=-1)
    
    @jit(nopython=True)
    def cumtrapz(y,x,initial=0):
        '''Cumulatively integrate over the last axis using the trapezoidal rule (i.e. equivalent to scipy.integrate.cumtrapz(y,x,axis=-1,initial=0),
           but faster due to the use of the numba.jit compiler).'''
        theshape = y.shape
        n = theshape[-1]
        f = y.T
        outar = np.zeros(theshape).T
        tmp = np.zeros(theshape[:-1]).T
        for i in range(n-1):
            tmp += (0.5*(f[i+1]+f[i])*(x[i+1]-x[i]))
            outar[i+1] = tmp
        return outar.T

#############################################################################

if usefortran:
    ## gives faster results even for jit-compiled computeuij while forceobj=True there (see below)
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           All arguments are arrays, i.e. A and B have shape (3,Ntheta) where Ntheta is e.g. the number of character angles.'''
        return np.moveaxis(fsub.elbrak(np.moveaxis(A,-1,0),np.moveaxis(B,-1,0),elC),0,-1)
    
    def elbrak1d(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           This function is similar to elbrak(), but its arguments do not depend on the character angle, i.e. A, B have shape (3).'''
        return fsub.elbrak1d(A,B,elC)
else:
    @jit(nopython=True)
    def elbrak(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           All arguments are arrays, i.e. A and B have shape (3,Ntheta) where Ntheta is e.g. the number of character angles.'''
        Ntheta = len(A[0,:,0])
        Nphi = len(A[0,0])
        tmp = np.zeros((Nphi))
        AB = np.zeros((3,3,Ntheta,Nphi))
        for th in range(Ntheta):
            for l in range(3):
                for o in range(3):
                    for k in range(3):
                        for p in range(3):
                            # AB[l,o,th] += A[k,th]*elC[k,l,o,p,th]*B[p,th]
                            #### faster numba-jit code is generated if we write the above like this (equivalent in pure python):
                            np.add(AB[l,o,th], np.multiply(np.multiply(A[k,th], elC[k,l,o,p,th],tmp), B[p,th],tmp), AB[l,o,th])
        
        return AB
    
    @jit(nopython=True)
    def elbrak1d(A,B,elC):
        '''Compute the bracket (A,B) := A.elC.B, where elC is a tensor of 2nd order elastic constants (potentially shifted by a velocity term or similar) and A,B are vectors.
           This function is similar to elbrak(), but its arguments do not depend on the character angle, i.e. A, B have shape (3).'''
        Nphi = len(A)
        AB = np.zeros((Nphi,3,3))
        for ph in range(Nphi):
            for l in range(3):
                for o in range(3):
                    for k in range(3):
                        for p in range(3):
                            AB[ph,l,o] += A[ph,k]*elC[k,l,o,p]*B[ph,p]
        return AB
