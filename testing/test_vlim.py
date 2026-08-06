#!/usr/bin/env python3
# test suite for PyDislocDyn
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Aug. 5, 2026
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
                # sys.stdout.write(line)
                logfile.write(line)
            subproc.wait()
    vlim_f_raw = read_dislocdyn_output("vlim_fcc.log")
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99)
        vlim_py[X] = Y[X].computevcrit(return_all=True)
        vlim_py[X].columns = vlim_py[X].columns/np.pi
        vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
        # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
        # compare results
        vlim_f[X] = vlim_f_raw[X]['vlim'].reset_index()
        vlim_f[X].columns = vlim_py[X].columns
        vlim_f[X].index.name = vlim_py[X].index.name
        assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
            f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_bcc(rnd=1):
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
                    # sys.stdout.write(line)
                    logfile.write(line)
                subproc.wait()
        vlim_f_raw[slip] = read_dislocdyn_output(f"vlim_bcc{slip}.log")
    for Xm in sorted(list(pydislocdyn.metal_data.bcc_metals)):
        for slip in ["110", "112", "123"]:
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
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
                    # sys.stdout.write(line)
                    logfile.write(line)
                subproc.wait()
        vlim_f_raw[slip] = read_dislocdyn_output(f"vlim_hcp{slip}.log")
    for Xm in sorted(list(pydislocdyn.metal_data.hcp_metals)):
        for slip, slipL in hcpslip.items():
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / str(Xm+slipL),Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
            # compare results
            vlim_f[X] = vlim_f_raw[slip][f"{Xm}basal"]['vlim'].reset_index()
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_tetr(rnd=2):
    '''tests if the fortran and python codes agree on limiting velocities of various tetragonal crystals'''
    testfolder = pathlib.Path(example_path / "tetr_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    vlim_f_raw = {}
    for slip in ["1","9"]:
        command = basecommand.copy()
        for X in sorted(list(pydislocdyn.metal_data.tetr_metals)):
            command.append(tmpfolder / X)
        command.append(example_path / f"vlim_tetr{slip}.in")
        with open(f"vlim_tetr{slip}.log", 'w', encoding="utf8") as logfile:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    # sys.stdout.write(line)
                    logfile.write(line)
                subproc.wait()
        vlim_f_raw[slip] = read_dislocdyn_output(f"vlim_tetr{slip}.log")
    for Xm in sorted(list(pydislocdyn.metal_data.tetr_metals)):
        for slip in ["1"]:
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / Xm,Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
            # compare results
            vlim_f[X] = vlim_f_raw[slip][Xm]['vlim'].reset_index()
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
