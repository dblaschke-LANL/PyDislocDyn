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
import pytest
import numpy as np
import pandas as pd
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
dir_path = pathlib.Path(__file__).resolve().parents[1]
example_path = dir_path / "examples"
import pydislocdyn
cwd =pathlib.Path.cwd()
# os.chdir(dir_path / "testing")
from test_regression import prepare_inputfiles

Y = {}
vlim_py = {}
vlim_f = {}
frnd = 2 ## number of digits the fortran code rounds to when outputtting vlim
tmpfolder="temp_pydislocdyn"
skiptests = False
reason  = ""
executable = dir_path / "dislocdyn.x"

## check if fortran executable exists, skip these tests if not:
if not executable.exists():
    reason = "Fortran executable not found - please compile and re-run this script!"
    skiptests = True

os.chdir(example_path)
prepare_inputfiles(tmpfolder)
tmpfolder = example_path / tmpfolder
os.chdir(cwd)

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_fcc(rnd=2):
    testfolder = pathlib.Path(example_path / "fcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    for X in sorted(list(pydislocdyn.metal_data.fcc_metals)):
        Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99)
        vlim_py[X] = Y[X].computevcrit(return_all=True)
        vlim_py[X].columns = vlim_py[X].columns/np.pi
        vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
        # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
        with open("vlim_"+X+".txt", 'w', encoding="utf8") as logfile:
            with subprocess.Popen([dir_path / "dislocdyn.x",tmpfolder / X,example_path / "vlim_fcc.in"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                for line in subproc.stdout:
                    # sys.stdout.write(line)
                    logfile.write(line)
                subproc.wait()
        # compare results
        vlim_f[X] = pd.read_csv(f"vlim_{X}.txt",skiprows=19,header=None,nrows=99,sep=r"\s+")
        vlim_f[X].columns = vlim_py[X].columns
        vlim_f[X].index.name = vlim_py[X].index.name
        assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
            f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
    os.remove("dislocdyn.log")
    os.chdir(cwd)

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_bcc(rnd=1):
    testfolder = pathlib.Path(example_path / "bcc_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    for Xm in sorted(list(pydislocdyn.metal_data.bcc_metals)):
        for slip in ["110", "112", "123"]:
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / X,Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
            with open("vlim_"+X+".txt", 'w', encoding="utf8") as logfile:
                with subprocess.Popen([dir_path / "dislocdyn.x",tmpfolder / X,example_path / f"vlim_bcc{slip}.in"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                    for line in subproc.stdout:
                        # sys.stdout.write(line)
                        logfile.write(line)
                    subproc.wait()
            # compare results
            vlim_f[X] = pd.read_csv(f"vlim_{X}.txt",skiprows=19,header=None,nrows=99,sep=r"\s+")
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
    os.remove("dislocdyn.log")
    os.chdir(cwd)

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_hcp(rnd=2):
    testfolder = pathlib.Path(example_path / "hcp_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    hcpslip = {"bas":"basal","pris":"prismatic","pyr":"pyramidal"}
    for Xm in sorted(list(pydislocdyn.metal_data.hcp_metals)):
        for slip in hcpslip:
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / str(Xm+hcpslip[slip]),Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
            with open("vlim_"+X+".txt", 'w', encoding="utf8") as logfile:
                with subprocess.Popen([dir_path / "dislocdyn.x",tmpfolder / f"{Xm}basal",example_path / f"vlim_hcp{slip}.in"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                    for line in subproc.stdout:
                        # sys.stdout.write(line)
                        logfile.write(line)
                    subproc.wait()
            # compare results
            vlim_f[X] = pd.read_csv(f"vlim_{X}.txt",skiprows=19,header=None,nrows=99,sep=r"\s+")
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
    os.remove("dislocdyn.log")
    os.chdir(cwd)

@pytest.mark.skipif(skiptests, reason=reason)
def test_fortran_vlim_tetr(rnd=2):
    testfolder = pathlib.Path(example_path / "tetr_metals")
    testfolder.mkdir(parents=True,exist_ok=True)
    os.chdir(testfolder)
    for Xm in sorted(list(pydislocdyn.metal_data.tetr_metals)):
        # for slip in ["1","9"]:
        for slip in ["1"]:
            X = Xm+slip
            Y[X] = pydislocdyn.readinputfile(tmpfolder / Xm,Ntheta=99)
            vlim_py[X] = Y[X].computevcrit(return_all=True)
            vlim_py[X].columns = vlim_py[X].columns/np.pi
            vlim_py[X] = vlim_py[X].T.round(frnd).reset_index()
            # vlim_py[X].to_csv("vlim_"+X+"_py.txt",sep=" ",float_format='%.2f')
            with open("vlim_"+X+".txt", 'w', encoding="utf8") as logfile:
                with subprocess.Popen([dir_path / "dislocdyn.x",tmpfolder / Xm,example_path / f"vlim_tetr{slip}.in"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as subproc:
                    for line in subproc.stdout:
                        # sys.stdout.write(line)
                        logfile.write(line)
                    subproc.wait()
            # compare results
            vlim_f[X] = pd.read_csv(f"vlim_{X}.txt",skiprows=19,header=None,nrows=99,sep=r"\s+")
            vlim_f[X].columns = vlim_py[X].columns
            vlim_f[X].index.name = vlim_py[X].index.name
            assert vlim_py[X].round(rnd).eq(vlim_f[X].round(rnd)).all().all(), \
                f"{X} differs:\n{vlim_py[X].round(rnd).compare(vlim_f[X].round(rnd))}"
    os.remove("dislocdyn.log")
    os.chdir(cwd)
