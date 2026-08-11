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
import numpy as np
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
dir_path = pathlib.Path(__file__).resolve().parents[1]
example_path = dir_path / "examples"
import pydislocdyn
from pydislocdyn import read_dislocdyn_output
cwd =pathlib.Path.cwd()
# os.chdir(dir_path / "testing")
from test_regression import prepare_inputfiles

Y = {}
vlim_py = {}
vlim_f = {}
frnd = 2 ## number of digits the fortran code rounds to when outputtting vlim
tmpfolder="temp_pydislocdyn"
skiptests = False
usefpm = False
reason  = ""
executable = dir_path / "dislocdyn.x"

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

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_fcc(rnd=2):
    '''tests if the fortran and python codes agree on limiting velocities of various fcc crystals'''
    testfolder = pathlib.Path(example_path / "fcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    command = basecommand.copy()
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        command.append(tmpfolder / X)
    command.append(example_path / "vlim_fcc.in")
    with open("vlim_fcc.log", 'w', encoding="utf8") as logfile:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
            for line in subproc.stdout:
                logfile.write(line)
            subproc.wait()
    vlim_f_raw = read_dislocdyn_output("vlim_fcc.log",postprocess=True)
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99)
        vlim_py[X] = Y[X].computevcrit(return_all=True)
        vlim_py[X].columns = vlim_py[X].columns/np.pi
        vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
        # compare results
        vlim_f[X] = vlim_f_raw[X]['vlim'].reset_index()
        assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
            f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_bcc(rnd=2):
    '''tests if the fortran and python codes agree on limiting velocities of various bcc crystals'''
    testfolder = pathlib.Path(example_path / "bcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    vlim_f_raw = {}
    for slip in ["110", "112", "123"]:
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.bcc_metals)):
            command.append(tmpfolder / (X+slip))
        command.append(example_path / f"vlim_bcc{slip}.in")
        with open(f"vlim_bcc{slip}.log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        vlim_f_raw[slip] = read_dislocdyn_output(f"vlim_bcc{slip}.log",postprocess=True)
    for Xm in sorted(list(pydislocdyn.metal_data.bcc_metals)):
        for slip in ["110", "112", "123"]:
            X = Xm+slip
            symmetric = True
            if slip in ['110','123']:
                symmetric = False
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99,symmetric=symmetric)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # compare results
            vlim_f[X] = vlim_f_raw[slip][X]['vlim'].reset_index()
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_hcp(rnd=2):
    '''tests if the fortran and python codes agree on limiting velocities of various hcp crystals'''
    testfolder = pathlib.Path(example_path / "hcp_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    hcpslip = {"bas":"basal","pris":"prismatic","pyr":"pyramidal"}
    vlim_f_raw = {}
    for slip in hcpslip:
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.hcp_metals)):
            command.append(tmpfolder / f"{X}basal")
        command.append(example_path / f"vlim_hcp{slip}.in")
        with open(f"vlim_hcp{slip}.log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    logfile.write(line)
                subproc.wait()
        vlim_f_raw[slip] = read_dislocdyn_output(f"vlim_hcp{slip}.log",postprocess=True)
    for Xm in sorted(list(pydislocdyn.metal_data.hcp_metals)):
        for slip, slipL in hcpslip.items():
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / str(Xm+slipL),Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # compare results
            vlim_f[X] = vlim_f_raw[slip][f"{Xm}basal"]['vlim'].reset_index()
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_tetr(rnd=2):
    '''tests if the fortran and python codes agree on limiting velocities of various tetragonal crystals'''
    testfolder = pathlib.Path(example_path / "tetr_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    vlim_f_raw = {}
    # bct metals:
    for islip in range(10):
        slip = str(islip+1)
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.bct_metals)):
            command.append(tmpfolder / (X+slip))
        fname = f"vlim_tetr_bct{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = vlimit\nntheta = 99\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            if islip in [3,5,9]: # islip starts at 0, i.e. non-symmetric bct slip planes are 4,6,10
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
        vlim_f_raw['bct'+slip] = read_dislocdyn_output(f"{fname[:-2]}log",postprocess=True)
    # fct metals:
    for islip in range(3):
        slip = str(islip+1)
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.fct_metals)):
            command.append(tmpfolder / (X+slip))
        fname = f"vlim_tetr_fct{slip}.in"
        with open(fname,"w",encoding="utf8") as infile:
            infile.write(f"sim_type = vlimit\nntheta = 99\nlogfile = {fname[:-2]}log\nechoinput = true\n")
            if islip==1: # islip starts at 0, i.e. non-symmetric fct slip plane is 2
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
        vlim_f_raw['fct'+slip] = read_dislocdyn_output(f"{fname[:-2]}log",postprocess=True)
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
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99,symmetric=symmetric)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # compare results
            vlim_f[X] = vlim_f_raw[slip+str(islip+1)][X]['vlim'].reset_index()
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
