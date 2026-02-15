#!/usr/bin/env python3
# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Feb. 15, 2026
'''If run as a script, this file will compute the dislocation line tension and generate various plots.
The script takes as (optional) arguments either the names of PyDislocDyn input files or keywords for
metals that are predefined in metal_data.py, falling back to all available if no argument is passed.
'''
#################################
import sys
import os
import pathlib
import shutil
import lzma
import numpy as np
dir_path = str(pathlib.Path(__file__).resolve().parents[1])
if dir_path not in sys.path:
    sys.path.append(dir_path)
##
from pydislocdyn import metal_data as data
from pydislocdyn.utilities import printthreadinfo, Ncores, read_2dresults, init_parser, \
    plt, fntsettings, AutoMinorLocator ## matplotlib stuff
from pydislocdyn.elasticconstants import UnVoigt
from pydislocdyn.dislocations import Dislocation, readinputfile
if Ncores>1:
    from joblib import Parallel, delayed

parser = init_parser(usage=f"\n{sys.argv[0]} <options> <inputfile(s)>\n\n",description=f"{__doc__}\n")
parser.add_argument('-Ntheta','--Ntheta', type=int, default=600, help='set the resolution of the character angles (angles between disloc. line and Burgers vector) used in line tension calculations')
parser.add_argument('-Ntheta2','--Ntheta2', type=int, default=21, help='set the resolution of the character angles considered for calculating limiting velocities (set to None or 0 to bypass entirely)')
parser.add_argument('-Nbeta','--Nbeta', type=int, default=500, help='set the resolution of the dislocation velocities to consider; set to 0 to bypass line tension calculations')
parser.add_argument('-Nphi', '--Nphi', type=int, default=1000, help='set the resolution of the polar angles (integration angles used in the integral method for computing dislocations)')
parser.add_argument('-scale_by_mu', '--scale_by_mu', type=str, default='exp', help="""choose which shear modulus to use for rescaling to dimensionless quantities;
                allowed values are: 'crude', 'aver', and 'exp'.
                Choose 'crude' for mu = (c11-c12+2c44)/4, 'aver' for mu=Hill average  (resp. improved average for cubic), 
                and 'exp' for experimental mu supplemented by 'aver' where data are missing
                (note: when using input files, 'aver' and 'exp' are equivalent in that mu provided 
                 in that file will be used and an average is only computed if mu is missing)""")
parser.add_argument('-bccslip', '--bccslip', type=str, default='all', help='''Choose among predefined bcc-slip systems when using metal_data.py (see that file for details);
allowed values: '110', '112', '123', 'all' (for all three)''')
parser.add_argument('-hcpslip', '--hcpslip', type=str, default='all', help='''Choose among predefined bcc-slip systems when using metal_data.py (see that file for details);
allowed values: 'basal', 'prismatic', 'pyramidal', 'all'  (for all three)''')

metal = sorted(list(data.all_metals | {'ISO'})) ### input data; also test isotropic limit

### start the calculations
if __name__ == '__main__':
    opts, args = parser.parse_known_args()
    if opts.Ncores is not None:
        Ncores = opts.Ncores
    Y={}
    metal_list = []
    use_metaldata=True
    printthreadinfo(Ncores)
    ### set range & step sizes after parsing the commandline for options
    dtheta = np.pi/(opts.Ntheta-2)
    theta = np.linspace(-np.pi/2-dtheta,np.pi/2+dtheta,opts.Ntheta+1)
    beta = np.linspace(0,1,opts.Nbeta)
    metal_kws = metal.copy()
    if len(sys.argv) > 1 and len(args)>0:
        try:
            inputdata = [readinputfile(i,init=False,theta=theta,Nphi=opts.Nphi) for i in args]
            Y = {i.name:i for i in inputdata}
            metal = metal_list = list(Y.keys())
            use_metaldata=False
            print(f"success reading input files {args}")
        except FileNotFoundError as fnameerror:
            ## only compute the metals the user has asked us to
            metal = args[0].split()
            for X in metal:
                if X not in metal_kws:
                    raise ValueError(f"One or more input files not found and {X} is not a valid keyword") from fnameerror
        
    bcc_metals = data.bcc_metals.copy()
    hcp_metals = data.hcp_metals.copy()
    if use_metaldata:
        pathlib.Path("temp_pydislocdyn").mkdir(exist_ok=True)
        os.chdir("temp_pydislocdyn")
        if opts.scale_by_mu not in ('exp','crude','aver'):
            raise ValueError("option 'scale_by_mu' must be one of 'exp','crude', or 'aver'.")
        if opts.scale_by_mu=='exp':
            isokw=False ## use single crystal elastic constants and additionally write average shear modulus of polycrystal to temp. input file
        else:
            isokw='omit' ## writeinputfile(..., iso='omit') will bypass writing ISO_c44 values to the temp. input files and missing Lame constants will always be auto-generated by averaging
        for X in metal:
            if X in bcc_metals:
                if opts.bccslip == 'all':
                    slipkw = ['110', '112', '123']
                else:
                    slipkw=[opts.bccslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokw,bccslip=kw)
                    metal_list.append(X+kw)
                    bcc_metals.add(X+kw)
            elif X in hcp_metals:
                if opts.hcpslip == 'all':
                    slipkw = ['basal','prismatic','pyramidal']
                else:
                    slipkw=[opts.hcpslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokw,hcpslip=kw)
                    metal_list.append(X+kw)
                    hcp_metals.add(X+kw)
            elif X=='ISO':
                metal_list.append(X)
            else:
                data.writeinputfile(X,X,iso=isokw) # write temporary input files for requested X of metal_data
                metal_list.append(X)
        for X in metal_list:
            if X=='ISO': ## define some isotropic elastic constants to check isotropic limit:
                Y[X] = Dislocation(sym='iso', name='ISO', b=[1,0,0], n0=[0,1,0], theta=theta, Nphi=opts.Nphi, lat_a=1e-10)
                Y[X].burgers = 1e-10
                Y[X].c44 = 1e9
                Y[X].poisson = 1/3
                Y[X].rho = 1e3
            else:
                Y[X] = readinputfile(X,init=False,theta=theta,Nphi=opts.Nphi)
        os.chdir("..")
        metal = metal_list
        ## list of metals symmetric in +/-theta (for the predefined slip systems):
        metal_symm = sorted(list({'ISO'}.union(data.fcc_metals).union(hcp_metals).union(data.tetr_metals).intersection(metal)))
    else:
        metal_symm = set([]) ## fall back to computing for character angles of both signs if we don't know for sure that the present slip system is symmetric

    for X in metal:
        Y[X].init_C2()
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        if opts.scale_by_mu == 'crude':
            Y[X].mu = (Y[X].C2[0,0]-Y[X].C2[0,1]+2*Y[X].C2[3,3])/4
        #### will generate missing mu/lam by averaging over single crystal constants (improved Hershey/Kroener scheme for cubic, Hill otherwise)
        ### for hexagonal/tetragonal metals, this corresponds to the average shear modulus in the basal plane (which is the most convenient for the present calculations)
        if Y[X].mu is None:
            Y[X].compute_Lame()

    if opts.Nbeta > 0:
        with open("linetension_calcs_options.log","w", encoding="utf8") as logfile:
            optiondict = vars(opts)
            for key, item in optiondict.items():
                if key not in ['Ncores', 'skip_plots']:
                    logfile.write(f"{key} = {item}\n")
                           
    C2 = {}
    scaling = {}
    beta_scaled = {}
    ## compute smallest critical velocity in ratio to the scaling velocity computed from the average shear modulus mu (see above):
    ## i.e. this corresponds to the smallest velocity leading to a divergence in the dislocation field at some character angle theta
    for X in metal:
        Y[X].findvcrit_smallest()
        if Y[X].ct==0:
            Y[X].ct = np.sqrt(Y[X].mu/Y[X].rho)
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        Y[X].C2norm = UnVoigt(Y[X].C2/Y[X].mu)
        ## ct was determined from mu above and thus may not be the actual transverse sound speed (if scale_by_mu='crude')
        scaling[X] = min(1,round(Y[X].vcrit_smallest/Y[X].ct+5e-3,2))
        beta_scaled[X] = scaling[X]*beta
    
    def maincomputations(i):
        '''wrap all main computations into a single function definition to be run in a parallelized loop'''
        X = metal[i]
        dislocation = Y[X]
        
        ### compute dislocation displacement gradient uij and line tension LT
        def compute_lt(j):
            dislocation.computeuij(beta=beta_scaled[X][j])
            dislocation.computeEtot()
            dislocation.computeLT()
            return 4*np.pi*dislocation.LT
            
        LT = np.array([compute_lt(j) for j in range(len(beta))])
        
        # write the results to disk (and backup previous results if they exist):
        if os.access(fname:=f"LT_{X}.dat.xz", os.R_OK):
            shutil.move(fname,fname[:-3]+".bak.xz")
        with lzma.open(f"LT_{X}.dat.xz","wt") as LTfile:
            LTfile.write(f"### dimensionless line tension prefactor LT(beta,theta) for {X}, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.\n")
            LTfile.write('beta/theta[pi]\t' + '\t'.join("{:.4f}".format(thi) for thi in theta[1:-1]/np.pi) + '\n')
            for j in range(len(beta)):
                LTfile.write(f"{beta_scaled[X][j]:.4f}\t" + '\t'.join("{:.6f}".format(thi) for thi in LT[j]) + '\n')

        return 0
        
    # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
    print(f"Computing the line tension for: {metal}")
    if Ncores == 1 and opts.Nbeta >=1:
        [maincomputations(i) for i in range(len(metal))]
    elif opts.Nbeta<1:
        print("skipping line tension calculations, Nbeta>0 required")
    else:
        Parallel(n_jobs=Ncores)(delayed(maincomputations)(i) for i in range(len(metal)))

################## create plots ################
    if opts.skip_plots:
        print("skipping plots as requested")
        plt_metal = []
    else:
        plt_metal = metal
    skip_plt = []

## load data from LT calculation
    LT = {}
    for X in plt_metal:
        try:
            LT[X] = read_2dresults(f"LT_{X}.dat") ## for every X, LT has Nbeta rows and Ntheta columns
        except FileNotFoundError:
            skip_plt.append(X)
            
    def mkLTplots(X):
        '''generates nice plots showing the dislocation line tension of metal X'''
        namestring = f"{X}"
        beta_trunc = [j for j in LT[X].index if j <=Y[X].vcrit_smallest/Y[X].ct]
        if X in metal_symm:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.2))
            LT_trunc = LT[X].iloc[:len(beta_trunc),int((LT[X].shape[1]-1)/2):].to_numpy()
            y_msh, x_msh = np.meshgrid(LT[X].columns[int((LT[X].shape[1]-1)/2):],beta_trunc)
            plt.yticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),**fntsettings)
        else:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,4.5))
            LT_trunc = LT[X].iloc[:len(beta_trunc)].to_numpy()
            y_msh, x_msh = np.meshgrid(LT[X].columns,beta_trunc)
            plt.yticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$-\pi/2$", r"$-3\pi/8$", r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),**fntsettings)
        plt.xticks(**fntsettings)
        plt.yticks(**fntsettings)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if X=='ISO':
            plt.xlabel(r'$\beta_\mathrm{t}=v/c_\mathrm{t}$',**fntsettings)
            plt.title(r'Isotropic',**fntsettings)
        else:
            plt.xlabel(r'$\beta_{\bar\mu}$',**fntsettings)
            plt.title(namestring,**fntsettings)
        plt.ylabel(r'$\vartheta$',rotation=0,**fntsettings)
        colmsh = plt.pcolormesh(x_msh,y_msh, LT_trunc, vmin=-0.5, vmax=2, cmap=plt.cm.rainbow, shading='gouraud')
        plt.colorbar()
        plt.contour(x_msh,y_msh,LT_trunc, colors=('black','red','black','black','black','black'), levels=[-0.5,0,0.5,1,1.5,2], linewidths=[0.7,1.0,0.7,0.7,0.7,0.7], linestyles=['solid','solid','dashed','dashdot','dotted','solid'])
        colmsh.set_rasterized(True)
        plt.axhline(0, color='grey', linewidth=0.5, linestyle='dotted')
        plt.savefig(f"LT_{X}.pdf",format='pdf',bbox_inches='tight',dpi=450)
        plt.close()

    for X in set(plt_metal).difference(set(skip_plt)):
        mkLTplots(X)
    
################################################
    if opts.Ntheta2==0 or opts.Ntheta2 is None:
        sys.exit()
    
    print(f"Computing critical velocities for: {metal}")
    for X in metal:
        if X in metal_symm:
            current_symm=True
            Y[X].computevcrit(theta=np.linspace(0,np.pi/2,opts.Ntheta2))
        else:
            current_symm=False
            Y[X].computevcrit(theta=np.linspace(-np.pi/2,np.pi/2,2*opts.Ntheta2-1))
    
    ## write vcrit results to disk, then plot
    with open("vcrit.dat","w", encoding="utf8") as vcritfile:
        vcritfile.write("theta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*opts.Ntheta2-1)) + '\n')
        vcritfile.write("metal / vcrit[m/s] (3 solutions per angle)\n")
        for X in sorted(list(set(metal))):
            for i in range(3):
                vcritfile.write(f"{X}\t" + '\t'.join("{:.0f}".format(thi) for thi in np.flipud(Y[X].vcrit_all[i+1,:])) + '\n')
                
    def mkvcritplot(X,Ntheta):
        '''Generates a plot showing the limiting (or critical) dislocation glide velocities as a function of character angle.'''
        fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(4.5,3.5))
        plt.tight_layout(h_pad=0.0)
        plt.xticks(**fntsettings)
        plt.yticks(**fntsettings)
        vcrit0 = Y[X].vcrit_all[1:].T
        if len(vcrit0)==Ntheta:
            plt.xticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),**fntsettings)
            thetapoints = np.linspace(0,np.pi/2,Ntheta)
        else:
            plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),**fntsettings)
            thetapoints = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
        ax1.axis((min(thetapoints),max(thetapoints),np.nanmin(vcrit0)*0.97,np.nanmax(vcrit0)*1.02)) ## define plot range
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.set_ylabel(r'$v_\mathrm{c}$[m/s]',**fntsettings)
        ax1.set_title(f"3 vcrit solutions for {X}",**fntsettings)
        for i in range(3):
            ax1.plot(thetapoints,vcrit0[:,i])
        ax1.set_xlabel(r'$\vartheta$',**fntsettings)
        plt.savefig(f"vcrit_{X}.pdf",format='pdf',bbox_inches='tight')
        plt.close()
    
    for X in sorted(list(set(metal))):
        mkvcritplot(X,opts.Ntheta2)
