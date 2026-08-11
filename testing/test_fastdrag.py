#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Aug. 6, 2026 - Aug. 11, 2026
'''This script verifies that both the Python code and the Fortran code give the same results
   for the dislocation limiting velocities up to the defined precision; it is meant to be run with pytest.'''
import os
import sys
import pathlib
import subprocess
import shutil
import pytest
try:
    from threadpoolctl import threadpool_limits
    use_ctl = True
except ImportError:
    use_ctl = False
import numpy as np
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
dir_path = pathlib.Path(__file__).resolve().parents[1]
example_path = dir_path / "examples"
import pydislocdyn
from pydislocdyn import read_dislocdyn_output, Ncpus, ompthreads
cwd =pathlib.Path.cwd()
from test_regression import prepare_inputfiles

frnd = 6 ## number of digits the fortran code rounds to when outputtting drag
tmpfolder="temp_pydislocdyn"
skiptests = False
usefpm = False
reason  = ""
executable = dir_path / "dislocdyn.x"

aver_lame = pydislocdyn.metal_data.all_metals.difference(pydislocdyn.metal_data.ISO_c44)

fpm = shutil.which('fpm')
os.chdir(dir_path)
if fpm is None:
    fpmoutput = "    "
else:
    with subprocess.Popen([fpm,"run","--profile","release","--","-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
        fpmoutput = []
        for line in subproc.stdout:
            fpmoutput.append(line)
        subproc.wait()
os.chdir(cwd)
if fpmoutput[-1].strip()[:4]=='2026':
    usefpm = True

## check if fortran executable exists, skip these tests if not:
if not executable.exists() and not usefpm:
    reason = "Fortran executable not found - please compile and re-run this script!"
    skiptests = True

os.chdir(example_path)
prepare_inputfiles(tmpfolder)
tmpfolder = example_path / tmpfolder
os.chdir(cwd)
basecommand = [executable]
if usefpm:
    basecommand = [fpm,"run","--profile","release","--"]

_ompthreads = ompthreads()
def _reset_ompthreads():
    '''revert OMP_NUM_THREADS to initial value'''
    if use_ctl:
        threadpool_limits(_ompthreads)
def _maximize_ompthreads():
    '''set OMP_NUM_THREADS to Ncpus'''
    if use_ctl:
        threadpool_limits(Ncpus)

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_drag_fcc(rnd=6,beta=0.25,Ntheta=3):
    '''tests if the fortran and python codes agree on drag coefficients of various fcc crystals'''
    testfolder = pathlib.Path(example_path / "fcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    Y = {}
    drag_py = {}
    drag_f = {}
    _maximize_ompthreads()
    command = basecommand.copy()
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        command.append(tmpfolder / X)
    fname = "drag_fcc.in"
    with open(fname,"w",encoding="utf8") as infile:
        infile.write(f"sim_type = drag\nntheta = {Ntheta}\nlogfile = {fname[:-2]}log\nechoinput = true\n")
        infile.write(f"betamin = {beta}\nnbeta = 1\n")
        for key, value in pydislocdyn.metal_data.example_slip_planes['fcc'].items():
            infile.write(f"{key} = ")
            value = np.array(value,dtype=float)
            infile.write(", ".join(map("{}".format,value))+"\n")
    command.append(testfolder / fname)
    with open("drag_fcc.log", 'w', encoding="utf8") as logfile:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
            for line in subproc.stdout:
                logfile.write(line)
            subproc.wait()
    drag_f = read_dislocdyn_output("drag_fcc.log",postprocess=True)
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=Ntheta)
        drag_py[X] = pydislocdyn.phonondrag(Y[X],beta,maxrec=-1,Debye_series=True).round(frnd)
        # compare results
        assert drag_py[X].round(rnd).eq(drag_f[X]['drag'].round(rnd)).all().all(), \
            f"{X} differs:\n{drag_py[X].round(rnd).compare(drag_f[X]['drag'].round(rnd))}"
    _reset_ompthreads()

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_drag_bcc(rnd=6,beta=0.25,Ntheta=3):
    '''tests if the fortran and python codes agree on drag coefficients of various bcc crystals'''
    testfolder = pathlib.Path(example_path / "bcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    Y = {}
    drag_py = {}
    drag_f = {}
    _maximize_ompthreads()
    for slip in ["110", "112", "123"]:
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.bcc_metals)):
            command.append(tmpfolder / (X+slip))
        fname = f"drag_bcc{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = drag\nntheta = {Ntheta}\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            infile.write(f"betamin = {beta}\nnbeta = 1\n")
            if slip in ["110", "123"] and Ntheta>2:
                infile.write("include_negative_theta = true\n")
            for key, value in pydislocdyn.metal_data.example_slip_planes['bcc'+slip].items():
                infile.write(f"{key} = ")
                value = np.array(value,dtype=float)
                infile.write(", ".join(map("{}".format,value))+"\n")
        command.append(testfolder / fname)
        with open(f"drag_bcc{slip}.log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        drag_f[slip] = read_dislocdyn_output(f"drag_bcc{slip}.log",postprocess=True)
    for Xm in sorted(list(pydislocdyn.metal_data.bcc_metals)):
        for slip in ["110", "112", "123"]:
            X = Xm+slip
            symmetric = True
            if slip in ['110','123']:
                symmetric = False
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=Ntheta,symmetric=symmetric)
            if Xm in aver_lame:
                Y[X].compute_Lame(roundto=0) ## default is -8 in python, but no rounding in fortran
                Y[X].init_sound()
            drag_py[X] = pydislocdyn.phonondrag(Y[X],beta,maxrec=-1,Debye_series=True).round(frnd)
            # compare results
            assert drag_py[X].round(rnd).eq(drag_f[slip][X]['drag'].round(rnd)).all().all(), \
                f"{X} differs:\n{drag_py[X].round(rnd).compare(drag_f[slip][X]['drag'].round(rnd))}"
    _reset_ompthreads()

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_drag_hcp(rnd=6,beta=0.25,Ntheta=3):
    '''tests if the fortran and python codes agree on drag coefficients of various hcp crystals'''
    testfolder = pathlib.Path(example_path / "hcp_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    hcpslip = {"bas":"basal","pris":"prismatic","pyr":"pyramidal"}
    Y = {}
    drag_py = {}
    drag_f = {}
    _maximize_ompthreads()
    for slip, slipL in hcpslip.items():
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.hcp_metals)):
            command.append(tmpfolder / f"{X}basal")
        fname = f"drag_hcp{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = drag\nntheta = {Ntheta}\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            infile.write(f"betamin = {beta}\nnbeta = 1\n")
            for key, value in pydislocdyn.metal_data.example_slip_planes['hcp'+slipL].items():
                infile.write(f"{key} = ")
                value = np.array(value,dtype=float)
                infile.write(", ".join(map("{}".format,value))+"\n")
        command.append(testfolder / fname)
        with open(f"drag_hcp{slip}.log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        drag_f[slip] = read_dislocdyn_output(f"drag_hcp{slip}.log",postprocess=True)
    for Xm in sorted(list(pydislocdyn.metal_data.hcp_metals)):
        for slip, slipL in hcpslip.items():
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / str(Xm+slipL),Ntheta=Ntheta)
            drag_py[X] = pydislocdyn.phonondrag(Y[X],beta,maxrec=-1,Debye_series=True).round(frnd)
            # compare results
            assert drag_py[X].round(rnd).eq(drag_f[slip][f'{Xm}basal']['drag'].round(rnd)).all().all(), \
                f"{X} differs:\n{drag_py[X].round(rnd).compare(drag_f[slip][f'{Xm}basal']['drag'].round(rnd))}"
    _reset_ompthreads()

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_drag_tetr(rnd=6,beta=0.25,Ntheta=3):
    '''tests if the fortran and python codes agree on drag coefficients of various tetragonal crystals'''
    testfolder = pathlib.Path(example_path / "tetr_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    Y = {}
    drag_py = {}
    drag_f = {}
    _maximize_ompthreads()
    # bct metals:
    for islip in range(10):
        slip = str(islip+1)
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.bct_metals)):
            command.append(tmpfolder / (X+slip))
        fname = f"drag_tetr_bct{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = drag\nntheta = {Ntheta}\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            infile.write(f"betamin = {beta}\nnbeta = 1\n")
            if islip in [3,5,9] and Ntheta>2: # islip starts at 0, i.e. non-symmetric bct slip planes are 4,6,10
                infile.write("include_negative_theta = true\n")
            for key, value in pydislocdyn.metal_data.example_slip_planes['bct'+slip].items():
                infile.write(f"{key} = ")
                value = np.array(value,dtype=float)
                infile.write(", ".join(map("{}".format,value))+"\n")
        command.append(testfolder / fname)
        with open(f"{fname[:-2]}log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        drag_f['bct'+slip] = read_dislocdyn_output(f"{fname[:-2]}log",postprocess=True)
    # fct metals:
    for islip in range(3):
        slip = str(islip+1)
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.fct_metals)):
            command.append(tmpfolder / (X+slip))
        fname = f"drag_tetr_fct{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = drag\nntheta = {Ntheta}\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            infile.write(f"betamin = {beta}\nnbeta = 1\n")
            if islip==1 and Ntheta>2: # islip starts at 0, i.e. non-symmetric fct slip plane is 2
                infile.write("include_negative_theta = true\n")
            for key, value in pydislocdyn.metal_data.example_slip_planes['fct'+slip].items():
                infile.write(f"{key} = ")
                value = np.array(value,dtype=float)
                infile.write(", ".join(map("{}".format,value))+"\n")
        command.append(testfolder / fname)
        with open(f"{fname[:-2]}log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        drag_f['fct'+slip] = read_dislocdyn_output(f"{fname[:-2]}log",postprocess=True)
    # the same with python, then compare:
    for Xm in sorted(list(pydislocdyn.metal_data.tetr_metals)):
        nslip = 3
        slip = 'fct'
        if Xm in pydislocdyn.metal_data.bct_metals:
            nslip=10
            slip = 'bct'
        for islip in range(nslip):
            X = Xm+str(islip+1)
            symmetric = True
            if (slip=='bct' and islip in [3,5,9]) or (slip=='fct' and islip==1):
                symmetric = False
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=Ntheta,symmetric=symmetric)
            if Xm in aver_lame:
                Y[X].compute_Lame(roundto=0) ## default is -8 in python, but no rounding in fortran
                Y[X].init_sound()
            drag_py[X] = pydislocdyn.phonondrag(Y[X],beta,maxrec=-1,Debye_series=True).round(frnd)
            # compare results
            assert drag_py[X].round(rnd).eq(drag_f[slip+str(islip+1)][X]['drag'].round(rnd)).all().all(), \
                f"{X} differs:\n{drag_py[X].round(rnd).compare(drag_f[slip+str(islip+1)][X]['drag'].round(rnd))}"
    _reset_ompthreads()
