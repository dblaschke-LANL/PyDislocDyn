#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 5, 2017 - Feb. 23, 2026
'''This module contains various utility functions used by other submodules.'''
#################################
import sys
import os
import shutil
import pathlib
import glob
import argparse
import math
from fractions import Fraction
import multiprocessing
Ncpus = multiprocessing.cpu_count()
def _ompthreads_auto():
    '''finds a optimal value for openmp parallelization of the fortran subroutines, i.e. OMP_NUM_THREADS'''
    if Ncpus<=3:
        return Ncpus
    ompthrds = int(math.sqrt(Ncpus))
    while Ncpus/ompthrds != round(Ncpus/ompthrds):
        ompthrds -= 1 ## choose an optimal value (assuming joblib is installed), such that ompthrds*Ncores = Ncpus and ompthrds ~ Ncores
    return ompthrds
if "OMP_NUM_THREADS" not in os.environ: ## allow user-override by setting this var. before running the python code
    _ompthreads = _ompthreads_auto()
    os.environ["OMP_NUM_THREADS"] = str(_ompthreads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(_ompthreads)
    except ImportError:
        pass
    del _ompthreads

import numpy as np
import sympy as sp
##################
import matplotlib as mpl
if 'ipykernel' not in sys.modules:
    mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
import matplotlib.pyplot as plt
##### use pdflatex and specify font through preamble:
if shutil.which('latex') and 'ipykernel' not in sys.modules:
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
def ompthreads():
    '''dummy fct that returns 0; only used if fortran subroutines are missing so that nothing else breaks'''
    return 0
try:
    import pydislocdyn.subroutines as fsub
    if fsub.version()>=20260223:
        usefortran = True
        ompthreads = fsub.ompinfo
    else:
        print("Error: the subroutines module is outdated, please re-compile by calling pydislocdy.utilities.compilefortranmodule() and reloading pydislocdyn")
except ImportError:
    pass

try:
    from joblib import Parallel, delayed
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/max(2,ompthreads()))) ## don't overcommit, ompthreads()=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used)
    ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = Ncpus = 1 ## must be 1 without joblib

dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)

def compilefortranmodule(buildopts='',clean=False):
    '''Compiles the Fortran subroutines if a Fortran compiler is available.
       Keyword 'buildopts' may be used to pass additional options to f2py.
       To delete files created by this function, set "clean"=True.'''
    cwd =pathlib.Path.cwd()
    compilerflags = '' ## when in doubt, build without OpenMP support
    if sys.version_info[:2]<=(3,11):
        import setuptools
        if setuptools.__version__ < '70' and np.__version__<'2.0':
            if shutil.which('gfortran'):
                compilerflags = '--f90flags=-fopenmp -lgomp' # flags specific to gfortran, require f2py with (old) distutils backend
        elif np.__version__>='1.26':
            compilerflags = '--dep=openmp --backend=meson' # requires meson to be installed
        else:
            raise Exception("I need either (setuptools <= 69) or (numpy>=1.26 and meson) to compile.")
    elif sys.version_info[:2]>=(3,12):
        compilerflags = '--dep=openmp' # requires f2py with new meson backend (default in numpy>=2.0 and python>=3.12)
    if buildopts != '':
        compilerflags += f" {buildopts}"
    os.chdir(pathlib.Path(__file__).parent)
    if clean:
        to_delete = ["subroutines.cpython*","fmoderror_py*.txt"]
        user_input = input(f'Deleting {to_delete[0]} and {to_delete[1]}; proceed? [y/N] ')
        if user_input.lower() in ('y', 'yes'):
            for files in to_delete:
                for f in glob.glob(files):
                    os.remove(f)
        os.chdir(cwd)
        return 0
    error = os.system(f'python -m numpy.f2py {compilerflags} -c subroutines.f90 -m subroutines')
    fname  = f"fmoderror_py{sys.version_info[0]}.{sys.version_info[1]}.txt"
    if error != 0:
        with open(fname,"w", encoding="utf8") as f1:
            f1.write(f"{error}")
        print(f"\nERROR: compilefortranmodule() failed using {compilerflags=}")
        print("make sure a Fortran compiler that is supported by numpy.f2py is installed")
        if '--dep' in compilerflags:
            print("as well as meson;")
            print("additional options (if necessary), such as e.g. '--build-dir', may be passed via my 'buildopts' keyword.")
    elif os.path.isfile(fname):
        os.remove(fname)
    os.chdir(cwd)
    return error

def printthreadinfo(Ncores):
    '''print a message to screen informing whether joblib parallelization (Ncores) or OpenMP parallelization (ompthreads())
       or both are currently employed; also warn if import of subroutines failed.'''
    _ompthreads = ompthreads()
    if Ncores > 1 and _ompthreads == 0: # check if subroutines were compiled with OpenMP support
        print(f"using joblib parallelization with {Ncores} cores")
    elif Ncores > 1:
        print(f"Parallelization: joblib with {Ncores} cores and OpenMP with {_ompthreads} threads")
    elif _ompthreads > 0:
        print(f"using OpenMP parallelization with {_ompthreads} threads")
    if not usefortran:
        print("\nWARNING: module 'subroutines' not found, execution will be slower")
        print("call pydislocdyn.utilities.compilefortranmodule() to compile this module, then reload pydislocdyn")
        if nonumba: print("\nWARNING: cannot find just-in-time compiler 'numba' either, execution will be very slow\n")

def str2bool(arg):
    '''converts a string to bool'''
    if arg in ['True', 'true', '1', 't', 'yes', 'y']:
        out=True
    elif arg in ['False', 'false', '0', 'f', 'no', 'n']:
        out=False
    else:
        raise ValueError(f"cannot convert {arg} to bool")
    return out

def convertfloat(arg):
    '''Return float(arg) if arg is a number or a length-1 numpy array, return an error otherwise.'''
    if isinstance(arg, np.ndarray) and len(arg)==1:
        return float(arg[0])
    return float(arg)

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

def init_parser(usage=f"\n{sys.argv[0]} <options> <inputfile(s)>\n\n",**kwargs):
    '''initializes an instance of an argparse.ArgumentParser() class with some default options, allowing additional arguments to be passed via **kwargs.'''
    parser = argparse.ArgumentParser(usage=usage,formatter_class=argparse.ArgumentDefaultsHelpFormatter,fromfile_prefix_chars='@',**kwargs)
    parser.add_argument('-Ncores','--Ncores', type=int, help='set the number of cores to use for joblib parallelization; will be auto-adjusted for optimal performance unless set by the user')
    parser.add_argument('-skip_plots','--skip_plots', action='store_true', help='as its name suggests, plots are skipped if this is set')
    return parser

def _separate_options(arglist):
    '''Separates options (format --key=value) from positional arguments (no dash), assuming arglist is a list of commandline arguments unknown to the argparse parser.
       Note that this function will silently ignore any options deviating from the expected format.'''
    out = [i.strip() for i in arglist]
    options = [i for i in out if "-" in i and i[:1]=="-"]
    kwargs = {}
    for i in options:
        out.remove(i)
        if "=" not in i: continue ## ignore options without assigned values
        key,val = i[1:].split("=")
        if key[0]=='-':
            key = key[1:]
        if key=="fromfile":
            raise ValueError("Option '--fromfile' is no longer supported, please use prefix '@' to read options from a file.")
        kwargs[key] = guesstype(val)
    return (out,kwargs)

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
    fname = str(fname)
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
    
def isclose(f1,f2,verbose=False):
    '''Returns True if all elements of arrays f1 and f2 are 'close' to one another and their shapes match, and False otherwise.'''
    out = False
    if f1.shape==f2.shape:
        out = np.allclose(f1,f2,equal_nan=True)
        if verbose and not out:
            print(compare_df(f1,f2))
    elif verbose:
        print("shapes differ")
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

def plotuij(uij,r,phi,lim=(-1,1),showplt=True,title=None,savefig=False,fntsize=11,axis=(-0.5,0.5,-0.5,0.5),figsize=(3.5,4.0),cmap=plt.cm.rainbow,showcontour=False,**kwargs):
    '''Generates a heat map plot of a 2-dim. dislocation field, where the x and y axes are in units of Burgers vectors and
    the color-encoded values are dimensionless displacement gradients.
    Required parameters are the 2-dim. array for the displacement gradient field, uij, as well as arrays r and phi for
    radius (in units of Burgers vector) and polar angle; note that the plot will be converted to Cartesian coordinates.
    Options include, the colorbar limits "lim", whether or not to call plt.show(), an optional title for the plot,
    which filename (if any) to save it as, the fontsize to be used, the plot range to be passed to pyplot.axis(), the size of
    the figure, which colormap to use, and whether or not show contours (showcontour may also include a list of levels).
    Additional options may be passed on to pyplot.contour via **kwargs (ignored if showcontour=False).'''
    phi_msh, r_msh = np.meshgrid(phi,r)
    x_msh = r_msh*np.cos(phi_msh)
    y_msh = r_msh*np.sin(phi_msh)
    if showplt and mpl.rcParams['text.usetex']:
        # print("Warning: turning off matplotlib LaTeX backend in order to show the plot")
        plt.rcParams.update({"text.usetex": False})
    plt.figure(figsize=figsize)
    plt.axis(axis)
    plt.xticks(np.linspace(*axis[:2],5),fontsize=fntsize,family=fntsettings['family'])
    plt.yticks(np.linspace(*axis[2:],5),fontsize=fntsize,family=fntsettings['family'])
    plt.xlabel(r'$x[b]$',fontsize=fntsize,family=fntsettings['family'])
    plt.ylabel(r'$y[b]$',fontsize=fntsize,family=fntsettings['family'])
    if title is not None: plt.title(title,fontsize=fntsize,family=fntsettings['family'],loc='left')
    if np.all(uij==0): raise ValueError('Dislocation field contains only zeros, forgot to calculate?')
    if uij.shape != (len(r),len(phi)):
        uij = np.outer(1/r,uij)
    colmsh = plt.pcolormesh(x_msh, y_msh, uij, vmin=lim[0], vmax=lim[-1], cmap=cmap, shading='gouraud')
    colmsh.set_rasterized(True)
    cbar = plt.colorbar()
    if not isinstance(showcontour,bool):
        kwargs['levels'] = showcontour
        showcontour = True
    if showcontour:
        if 'levels' not in kwargs: kwargs['levels'] = np.linspace(-1,1,6)
        if 'colors' not in kwargs: kwargs['colors'] = 'white'
        if 'linewidths' not in kwargs: kwargs['linewidths'] = 0.7
        plt.contour(x_msh,y_msh,uij,**kwargs)
    cbar.ax.tick_params(labelsize=fntsize)
    if savefig is not False: plt.savefig(savefig,format='pdf',bbox_inches='tight',dpi=150)
    if showplt:
        plt.show()
    plt.close()
