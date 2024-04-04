#!/usr/bin/env python3
# Compute averages of elastic constants for polycrystals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 7, 2017 - Apr. 4, 2024
'''If run as a script, this file will compute polycrystal averages of second and third order elastic constants, either for
   all metals predefined in metal_data.py, or for those input files passed as arguments to the script; results are written
   to a text file 'averaged_elastic_constants.tex'.'''
#################################
import sys
import os
import pandas as pd
## workaround for spyder's runfile() command when cwd is somewhere else:
dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__),os.pardir))
if dir_path not in sys.path:
    sys.path.append(dir_path)
##
from pydislocdyn.elasticconstants import elasticS2, elasticS3, UnVoigt
import pydislocdyn.metal_data as data
from pydislocdyn.crystals import readinputfile, IsoAverages, lam, mu, Murl, Murm, Murn

metal = sorted(list(data.all_metals.intersection(data.CRC_c11.keys()))) ## generate a list of those metals for which we have sufficient data
    
def dict_to_pandas(dictionary):
    '''converts a dictionary containing polycrystalline averages to a pandas DataFrame'''
    if len(dictionary)==1:
        dictionary['dummy'] = dictionary[list(dictionary.keys())[0]]
    out = pd.DataFrame(dictionary,dtype=float).T
    if len(dictionary)==1: out.drop('dummy')
    out.columns = pd.Index([r'$\lambda$',r'$\mu$','$l$','$m$','$n$'])
    return out

def writelatex(data,caption="",soec_only=False,dropna=False):
    '''converts a pandas DataFrame containing polycrystalline averages to a LaTeX table'''
    out = data
    if soec_only:
        out = out.iloc[:,:2]
    if dropna:
        out.dropna(inplace=True)
    if pd.__version__ < '1.3':
        print(f"Warning: using workaround for pandas {pd.__version__}; version 1.3 or higher is recommended")## work around for pandas 1.1 and 1.2 (results in slightly different formatting though)
        return out.to_latex(caption=caption,float_format="%.1f",escape=False).replace(r'{}','').replace('\n\\midrule',r' \hline').replace('\n\\toprule','').replace('\n\\bottomrule','').replace('\n\\centering','')
    out = out.style.format(precision=1)
    if not soec_only:
        out = out.format('{:.0f}',subset=['$l$','$m$','$n$'])
    return out.to_latex(caption=caption).replace(r'{}','').replace(r'{$','$').replace(r'$}','$').replace(r'$ \\',r'$ \\ \hline') ## pandas 1.3 puts curly brackets on column names, pandas >=1.4 does not

if __name__ == '__main__':
    Y={}
    use_metaldata=True
    metal_kws = metal.copy()
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        try:
            inputdata = [readinputfile(i) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError as fnameerror:
            ## only compute the metals the user has asked us to (or otherwise all those for which we have sufficient data)
            metal = sys.argv[1].split()
            for X in metal:
                if X not in metal_kws:
                    raise ValueError(f"One or more input files not found and {X} is not a valid keyword") from fnameerror
    
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        for X in metal:
            ## change alt_soec=True to use SOEC numbers from Thomas:1968, Hiki:1966, Leese:1968, Powell:1984, and Graham:1968
            ## for some of the cubic metals (these were used for arXiv:1706.07132)
            data.writeinputfile(X,X,alt_soec=False) # write temporary input files for requested X of metal_data
            Y[X] = readinputfile(X)
        os.chdir("..")
    
    metal_cubic = []
    metal_toec = []
    metal_toec_cubic = []
    for X in metal:
        ### will compute improved averages for cubic metals only:
        if Y[X].sym in ('fcc', 'bcc', 'cubic'):
            metal_cubic.append(X)
        if abs(Y[X].c123)>0 or Y[X].cijk is not None: ### subset for which we have TOEC
            metal_toec.append(X)
            if Y[X].sym in ('fcc', 'bcc', 'cubic'):
                metal_toec_cubic.append(X)
    print(f"Computing for: {metal} (={len(metal)} metals)")

    # results to be stored in the following dictionaries (for various metals)
    VoigtAverage = {}
    ReussAverage = {}
    HillAverage = {}
    ImprovedAv = {}
    
    aver = IsoAverages(lam,mu,Murl,Murm,Murn) ### initialize isotropic quantities first
    print(f"Computing Voigt and Reuss averages for SOEC of {len(metal)} metals and for TOEC of {len(metal_toec)} metals ...")
    # do the calculations for various metals:
    C2 = {}
    C3 = {}
    S2 = {}
    S3 = {}
    for X in metal:
        #### divide by 1e9 to get the results in units of GPa
        C2[X] = UnVoigt(Y[X].C2/1e9)
        S2[X] = elasticS2(C2[X])
        C3[X] = None
        S3[X] = None
    
    for X in metal_toec:
        C3[X] = UnVoigt(Y[X].C3/1e9)
        S3[X] = elasticS3(S2[X],C3[X])

    for X in metal:
        VoigtAverage[X] = aver.voigt_average(C2[X],C3[X])
        ReussAverage[X] = aver.reuss_average(S2[X],S3[X])
        HillAverage[X] = aver.hill_average()
    
    print(f"Computing improved averages for SOEC of {len(metal_cubic)} cubic metals and for TOEC of {len(metal_toec_cubic)} cubic metals ...")
    
    for X in metal_cubic:
        ImprovedAv[X] = aver.improved_average(C2[X],C3[X])
    
    ##### write results to files (as LaTeX tables):
    Averages = {}
    with open("averaged_elastic_constants.tex","w", encoding="utf8") as averfile:
        for title, dictionary in [('Voigt',VoigtAverage),('Reuss',ReussAverage),('Hill',HillAverage),('improved',ImprovedAv)]:
            if title not in ['improved'] or len(metal_cubic)>0:
                Averages[title] = dict_to_pandas(dictionary)
                averfile.write(writelatex(Averages[title],caption=f"{title} averages [GPa]:",soec_only=True))
                averfile.write("\n\n")
                if len(metal_toec)>0:
                    averfile.write(writelatex(Averages[title],dropna=True)+"\n\n")
    print("done.")
