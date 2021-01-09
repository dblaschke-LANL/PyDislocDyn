# Compute the line tension of a moving dislocation for various metals
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Jan. 9, 2021
#################################
import sys
import os
import numpy as np
import sympy as sp
import scipy
from scipy.optimize import fmin
##################
import matplotlib as mpl
mpl.use('Agg', force=False) # don't need X-window, allow running in a remote terminal session
##### use pdflatex and specify font through preamble:
# mpl.use("pgf")
# pgf_with_pdflatex = {
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": [
#           r"\usepackage[utf8x]{inputenc}",
#           r"\usepackage[T1]{fontenc}",
#           r"\usepackage{fouriernc}",
#           r"\usepackage{amsmath}",
#           ]
# }
# mpl.rcParams.update(pgf_with_pdflatex)
##################
import matplotlib.pyplot as plt
plt.rc('font',**{'family':'Liberation Serif','size':'11'})
fntsize=11
from matplotlib.ticker import AutoMinorLocator
##################
import metal_data as data
from elasticconstants import elasticC2, Voigt, UnVoigt
from polycrystal_averaging import metal_props, loadinputfile
import dislocations as dlc
try:
    from joblib import Parallel, delayed, cpu_count
    ## detect number of cpus present:
    Ncpus = cpu_count()
    ## choose how many cpu-cores are used for the parallelized calculations (also allowed: -1 = all available, -2 = all but one, etc.):
    Ncores = max(1,int(Ncpus/max(2,dlc.ompthreads))) ## don't overcommit, ompthreads=# of threads used by OpenMP subroutines (or 0 if no OpenMP is used) ## use half of the available cpus (on systems with hyperthreading this corresponds to the number of physical cpu cores)
    # Ncores = -2
    if dlc.ompthreads == 0: # check if subroutines were compiled with OpenMP support
        print("using joblib parallelization with ",Ncores," cores")
    else:
        print("Parallelization: joblib with ",Ncores," cores and OpenMP with ",dlc.ompthreads," threads")
except ImportError:
    print("WARNING: module 'joblib' not found, will run on only one core\n")
    Ncores = Ncpus = 1 ## must be 1 without joblib
    if dlc.ompthreads > 0:
        print("using OpenMP parallelization with ",dlc.ompthreads," threads")

## choose which shear modulus to use for rescaling to dimensionless quantities
## allowed values are: 'crude', 'aver', and 'exp'
## choose 'crude' for mu = (c11-c12+2c44)/4, 'aver' for mu=Hill average  (resp. improved average for cubic), and 'exp' for experimental mu supplemented by 'aver' where data are missing
## (note: when using input files, 'aver' and 'exp' are equivalent in that mu provided in that file will be used and an average is only computed if mu is missing)
scale_by_mu = 'exp'
skip_plots=False ## set to True to skip generating line tension plots from the results
### choose resolution of discretized parameters: theta is the angle between disloc. line and Burgers vector, beta is the dislocation velocity,
### and phi is an integration angle used in the integral method for computing dislocations
Ntheta = 600
Ntheta2 = 21 ## number of character angles for which to calculate critical velocities (set to None or 0 to bypass entirely)
Nbeta = 500 ## set to 0 to bypass line tension calculations
Nphi = 1000
## choose among predefined slip systems when using metal_data.py (see that file for details)
bccslip = '110' ## allowed values: '110' (default), '112', '123', 'all' (for all three)
hcpslip = 'basal' ## allowed values: 'basal', 'prismatic', 'pyramidal', 'all' (for all three)
### and range & step sizes
dtheta = np.pi/(Ntheta-2)
theta = np.linspace(-np.pi/2-dtheta,np.pi/2+dtheta,Ntheta+1)
beta = np.linspace(0,1,Nbeta)
#####

#### input data:
metal = sorted(list(data.fcc_metals.union(data.bcc_metals).union(data.hcp_metals).union(data.tetr_metals)))
metal = sorted(metal + ['ISO']) ### test isotropic limit

## define a new method to compute critical velocities within the imported pca.metal_props class:
def computevcrit(self,Ntheta,Ncores=Ncores,symmetric=False,cache=False,theta_list=None):
    '''Computes the 'critical velocities' of a dislocation for the number Ntheta (resp. 2*Ntheta-1) of character angles in the interval [0,pi/2] (resp. [-pi/2, pi/2] if symmetric=False),
       i.e. the velocities that will lead to det=0 within the StrohGeometry.
       Optionally, an explicit list of angles in units of pi/2 may be passed via theta_list (Ntheta is a required argument, but is ignored in this case).
       Additionally, the crystal symmetry must also be specified via sym= one of 'iso', 'fcc', 'bcc', 'hcp', 'tetr', 'trig', 'orth', 'mono', 'tric'.'''
    cc11, cc12, cc13, cc14, cc22, cc23, cc33, cc44, cc55, cc66 = sp.symbols('cc11, cc12, cc13, cc14, cc22, cc23, cc33, cc44, cc55, cc66', real=True)
    bt2, phi = sp.symbols('bt2, phi', real=True)
    substitutions = {cc11:self.C2[0,0]/1e9, cc12:self.C2[0,1]/1e9, cc44:self.C2[3,3]/1e9}
    if self.sym in ['hcp', 'tetr', 'trig', 'orth', 'mono', 'tric']:
        substitutions.update({cc13:self.C2[0,2]/1e9, cc33:self.C2[2,2]/1e9})
    if self.sym in ['tetr', 'orth', 'mono', 'tric']:
        substitutions.update({cc66:self.C2[5,5]/1e9})
    if self.sym=='trig' or self.sym=='tric':
        substitutions.update({cc14:self.C2[0,3]/1e9})
    if self.sym in ['orth', 'mono', 'tric']:
        substitutions.update({cc22:self.C2[1,1]/1e9, cc23:self.C2[1,2]/1e9, cc55:self.C2[4,4]/1e9})
    if self.sym=='mono' or self.sym=='tric':
        cc15, cc25, cc35, cc46 = sp.symbols('cc15, cc25, cc35, cc46', real=True)
        substitutions.update({cc15:self.C2[0,4]/1e9, cc25:self.C2[1,4]/1e9, cc35:self.C2[2,4]/1e9, cc46:self.C2[3,5]/1e9})
    if self.sym=='tric':
        cc16, cc24, cc26, cc34, cc36, cc45, cc56 = sp.symbols('cc16, cc24, cc26, cc34, cc36, cc45, cc56', real=True)
        substitutions.update({cc16:self.C2[0,5]/1e9, cc24:self.C2[1,3]/1e9, cc26:self.C2[1,5]/1e9, cc34:self.C2[2,3]/1e9, cc36:self.C2[2,5]/1e9, cc45:self.C2[3,4]/1e9, cc56:self.C2[4,5]/1e9})
    if self.sym=='iso' or self.sym=='fcc' or self.sym=='bcc':
        C2 = elasticC2(c11=cc11,c12=cc12,c44=cc44)
    elif self.sym=='hcp':
        C2 = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33)
    elif self.sym=='tetr':
        C2 = elasticC2(c11=cc11,c12=cc12,c13=cc13,c44=cc44,c33=cc33,c66=cc66)
    elif self.sym=='trig':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc14,cc33,cc44))
    elif self.sym=='orth':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc22,cc23,cc33,cc44,cc55,cc66))
    elif self.sym=='mono':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc15,cc22,cc23,cc25,cc33,cc35,cc44,cc46,cc55,cc66))
    elif self.sym=='tric':
        C2 = elasticC2(cij=(cc11,cc12,cc13,cc14,cc15,cc16,cc22,cc23,cc24,cc25,cc26,cc33,cc34,cc35,cc36,cc44,cc45,cc46,cc55,cc56,cc66))
    else:
        raise ValueError("sym={} not implemented".format(self.sym))
    if symmetric or Ntheta==1:
        Theta = (sp.pi/2)*np.linspace(0,1,Ntheta)
    else:
        Theta = (sp.pi/2)*np.linspace(-1,1,2*Ntheta-1)
        Ntheta=len(Theta)
    if Ntheta==1:
        Theta = np.array([Theta])
    if theta_list is not None:
        Theta = (sp.pi/2)*np.asarray(theta_list)
        Ntheta = len(Theta)
    ## workaround for sympy<1.3:
    if sp.__version__ < "1.3":
        ldfymod=["math", "mpmath"]
    else:
        ldfymod=["scipy"]
    ###
    def compute_bt2(N,m0,C2,bt2,cc44=cc44):
        NC2N = np.dot(N,np.dot(N,C2))
        thedot = np.dot(N,m0)
        for a in sp.preorder_traversal(thedot):
            if isinstance(a, sp.Float):
                thedot = thedot.subs(a, round(a, 12))
        thematrix = NC2N - bt2*cc44*(thedot**2)*np.diag((1,1,1))
        thedet = sp.det(sp.Matrix(thematrix))
        return sp.solve(thedet,bt2)
    def computevcrit(b,n0,C2,Ntheta,bt2=bt2,Ncores=Ncores):
        Ncores = min(Ncores,Ncpus) # don't over-commit and don't fail if joblib not loaded
        t = np.zeros((Ntheta,3))
        m0 = np.zeros((Ntheta,3))
        N = np.zeros((Ntheta,3),dtype=object)
        for thi in range(Ntheta):
            t[thi] = sp.cos(Theta[thi])*b + sp.sin(Theta[thi])*np.cross(b,n0)
            m0[thi] = np.cross(n0,t[thi])
            N[thi] = n0*sp.cos(phi) - m0[thi]*sp.sin(phi)
        bt2_curr = np.zeros((Ntheta,3),dtype=object)
        if cache is False or cache is True:
            if Ncores == 1:
                for thi in range(Ntheta):
                    bt2_curr[thi] = compute_bt2(N[thi],m0[thi],C2,bt2)
            else:
                bt2_curr = np.array(Parallel(n_jobs=Ncores)(delayed(compute_bt2)(N[thi],m0[thi],C2,bt2) for thi in range(Ntheta)))
        else:
            bt2_curr=cache ## load user-provided from previous calculation
        if cache is True:
            self.cache_bt2 = bt2_curr.copy()
        vcrit = np.zeros((2,Ntheta,3))
        def findmin(bt2_curr,substitutions=substitutions,phi=phi,norm=(self.C2[3,3]/self.rho)):
            bt2_res = np.zeros((3,2),dtype=complex)
            for i in range(len(bt2_curr)):
                bt2_curr[i] = (bt2_curr[i].subs(substitutions))
                fphi = sp.lambdify((phi),bt2_curr[i],modules=ldfymod)
                def f(x):
                    out = fphi(x)
                    return np.real(out)
                bt2_res[i,0] = fmin(f,0.01,disp=False)
                bt2_res[i,1] = bt2_curr[i].subs({phi:bt2_res[i,0]})
            mask = np.round(np.imag(bt2_res[:,1]),12)==0 ## only keep real solutions
            return np.array([np.sqrt(norm*np.real(bt2_res[:,1][mask])),np.real(bt2_res[:,0][mask])])
        if Ncores == 1:
            for thi in range(Ntheta):
                vcrit[:,thi] = findmin(bt2_curr[thi],substitutions,phi,(self.C2[3,3]/self.rho))
        else:
            vcrit = np.moveaxis(np.array(Parallel(n_jobs=Ncores)(delayed(findmin)(bt2_curr[thi],substitutions,phi,self.C2[3,3]/self.rho) for thi in range(Ntheta))),1,0)
        return vcrit
    
    self.vcrit = computevcrit(self.b,self.n0,C2,Ntheta,Ncores=Ncores)
    return self.vcrit[0]
metal_props.computevcrit=computevcrit

class Dislocation(dlc.StrohGeometry,metal_props):
    '''This class has all properties and methods of classes StrohGeometry and metal_props, plus an additional method: computevcrit.
       If optional keyword Miller is set to True, b and n0 are interpreted as Miller indices (and Cartesian otherwise); note since n0 defines a plane its Miller indices are in reziprocal space.'''
    def __init__(self,b, n0, theta, Nphi,sym='iso', name='some_crystal',Miller=False):  
        metal_props.__init__(self, sym, name)
        if Miller==True:
            self.Millerb = b
            b = self.Miller_to_Cart(self.Millerb)
            self.Millern0 = n0
            n0 = self.Miller_to_Cart(self.Millern0,reziprocal=True)
        dlc.StrohGeometry.__init__(self, b, n0, theta, Nphi)
        self.sym = sym
    def __repr__(self):
        return  metal_props.__repr__(self) + "\n" + dlc.StrohGeometry.__repr__(self)

def readinputfile(fname,init=True,theta=[0,np.pi/2],Nphi=500):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the Dislocation class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.
       Array theta contains all dislocation characters to be considered, and integer Nphi denotes the resolution to be used for polar angle phi.'''
    inputparams = loadinputfile(fname)
    sym = inputparams['sym']
    if 'name' in inputparams.keys():
        name = inputparams['name']
    else:
        name = str(fname)
    if 'Millerb' in inputparams.keys() or 'Millern0' in inputparams.keys():
        temp = metal_props(sym,name) ## need a metal_props method to convert to Cartesian b, n0
        temp.populate_from_dict(inputparams)
        b = temp.b
        n0 = temp.n0
    else:
        b = np.asarray(inputparams['b'].split(','),dtype=float)
        n0 = np.asarray(inputparams['n0'].split(','),dtype=float)
    out = Dislocation(sym=sym, name=name, b=b, n0=n0, theta=theta, Nphi=Nphi)
    out.populate_from_dict(inputparams)
    if init:
        out.init_all()
    return out

### start the calculations
if __name__ == '__main__':
    Y={}
    inputdata = {}
    metal_list = []
    use_metaldata=True
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        try:
            for i in range(len(args)):
                inputdata[i]=readinputfile(args[i],init=False,theta=theta,Nphi=Nphi)
                X = inputdata[i].name
                metal_list.append(X)
                Y[X] = inputdata[i]
            use_metaldata=False
            metal = metal_list
            print("success reading input files ",args)
        except FileNotFoundError:
            ## only compute the metals the user has asked us to
            metal = sys.argv[1].split()
    bcc_metals = data.bcc_metals.copy()
    hcp_metals = data.hcp_metals.copy()
    if use_metaldata:
        if not os.path.exists("temp_pydislocdyn"):
            os.mkdir("temp_pydislocdyn")
        os.chdir("temp_pydislocdyn")
        if scale_by_mu=='exp':
            isokw=False ## use single crystal elastic constants and additionally write average shear modulus of polycrystal to temp. input file
        else:
            isokw='omit' ## writeinputfile(..., iso='omit') will bypass writing ISO_c44 values to the temp. input files and missing Lame constants will always be auto-generated by averaging
        for X in metal:
            if X in bcc_metals:
                if bccslip == 'all':
                    slipkw = ['110', '112', '123']
                else:
                    slipkw=[bccslip]
                for kw in slipkw:
                    data.writeinputfile(X,X+kw,iso=isokw,bccslip=kw)
                    metal_list.append(X+kw)
                    bcc_metals.add(X+kw)
            elif X in hcp_metals:
                if hcpslip == 'all':
                    slipkw = ['basal','prismatic','pyramidal']
                else:
                    slipkw=[hcpslip]
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
            if X=='ISO':
                Y[X] = Dislocation(sym='iso', name='ISO', b=[1,0,0], n0=[0,1,0], theta=theta, Nphi=Nphi)
                ### define some isotropic constants to check isotropic limit:
                Y[X].a = Y[X].burgers = 1e-10
                Y[X].c44 = 1e9
                Y[X].rho = 1e3
                nu = 1/3 ## define Poisson's ratio
                Y[X].c12 = round(Y[X].c44*2*nu/(1-2*nu),2)
                Y[X].c11 = Y[X].c12+2*Y[X].c44
            else:
                Y[X] = readinputfile(X,init=False,theta=theta,Nphi=Nphi)
        os.chdir("..")
        metal = metal_list
        ## list of metals symmetric in +/-theta (for the predefined slip systems):
        metal_symm = sorted(list({'ISO'}.union(data.fcc_metals).union(hcp_metals).union(data.tetr_metals).intersection(metal)))
    else:
        metal_symm = set([]) ## fall back to computing for character angles of both signs if we don't know for sure that the present slip system is symmetric

    for X in metal:
        Y[X].init_C2()
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        if scale_by_mu == 'crude':
            Y[X].mu = (Y[X].c11-Y[X].c12+2*Y[X].c44)/4
        #### will generate missing mu/lam by averaging over single crystal constants (improved Hershey/Kroener scheme for cubic, Hill otherwise)
        ### for hexagonal/tetragonal metals, this corresponds to the average shear modulus in the basal plane (which is the most convenient for the present calculations)
        if Y[X].mu==None:
            Y[X].compute_Lame()

    print("Computing the line tension for: ",metal)

    if Nbeta > 0:
        with open("theta.dat","w") as thetafile:
            thetafile.write('\n'.join("{:.6f}".format(thi) for thi in theta[1:-1]))
                           
    C2 = {}
    scaling = {}
    beta_scaled = {}
    ## compute smallest critical velocity in ratio to the scaling velocity computed from the average shear modulus mu (see above):
    ## i.e. this corresponds to the smallest velocity leading to a divergence in the dislocation field at some character angle theta
    vcrit_smallest = {}
    for X in metal:
        if Y[X].sym=='iso':
            vcrit_smallest[X] = 1
        else:
            ## happens to be the same formula for both fcc and some bcc (but corresponding to pure edge for fcc and mixed with theta=arctan(sqrt(2)) for bccslip='110', pure screw for bccslip='112')
            vcrit_smallest[X] = min(1,np.sqrt((Y[X].c11-Y[X].c12)/(2*Y[X].c44))) ## scaled by c44, will rescale to user's choice below
            ### also the correct value for some (but not all) hexagonal metals, i.e. depends on values of SOEC and which slip plane is considered
            ### ... but not for tetragonal and bccslip='123', where vcrit needs to be determined numerically:
    ## numerically determined values at T=300K for metals in metal_data.py:
    if use_metaldata:
        vcrit_smallest['In'] = 0.549
        vcrit_smallest['Sn'] = 0.749
        vcrit_smallest['Znbasal'] = 0.998 ## for basal slip
        if hcpslip=='prismatic' or hcpslip=='pyramidal' or hcpslip=='all':
            vcrit_smallest['Cdprismatic'] = vcrit_smallest['Cdpyramidal'] = 0.932
            vcrit_smallest['Znprismatic'] = vcrit_smallest['Znpyramidal'] = 0.766
        if bccslip=='123' or bccslip=='all':
            vcrit_smallest['Fe123'] = 0.616
            vcrit_smallest['K123'] = 0.393
            vcrit_smallest['Ta123'] = 0.807
    for X in metal:
        ## want to scale everything by the average shear modulus and thereby calculate in dimensionless quantities
        C2[X] = UnVoigt(Y[X].C2/Y[X].mu)
        # we limit calculations to a velocity range close to what we actually need,
        vcrit_smallest[X] = vcrit_smallest[X]*np.sqrt(Y[X].c44/Y[X].mu) ## rescale to mu instead of c44
        if Y[X].vcrit_smallest != None: ## overwrite with data from input file, if available
            Y[X].ct = np.sqrt(Y[X].mu/Y[X].rho)
            vcrit_smallest[X] = Y[X].vcrit_smallest/Y[X].ct ## ct was determined from mu above and thus may not be the actual transverse sound speed (if scale_by_mu='crude')
        scaling[X] = min(1,round(vcrit_smallest[X]+5e-3,2))
        beta_scaled[X] = scaling[X]*beta

        
    # wrap all main computations into a single function definition to be run in a parallelized loop below
    def maincomputations(i):
        X = metal[i]
        with open("beta_{}.dat".format(X),"w") as betafile:
            betafile.write('\n'.join("{:.5f}".format(bti) for bti in beta_scaled[X]))
    
        dislocation = Y[X]
        
        ### compute dislocation displacement gradient uij and line tension LT
        def compute_lt(j):
            dislocation.computeuij(beta=beta_scaled[X][j], C2=C2[X])
            dislocation.computeEtot()
            dislocation.computeLT()
            return 4*np.pi*dislocation.LT
            
        LT = np.array([compute_lt(j) for j in range(len(beta))])
        
        # write the results to disk:
        with open("LT_{}.dat".format(X),"w") as LTfile:
            LTfile.write("### dimensionless line tension prefactor LT(beta,theta) for {}, one row per beta, one column per theta; theta=0 is pure screw, theta=pi/2 is pure edge.".format(X) + '\n')
            LTfile.write('beta/theta[pi]\t' + '\t'.join("{:.4f}".format(thi) for thi in theta[1:-1]/np.pi) + '\n')
            for j in range(len(beta)):
                LTfile.write("{:.4f}".format(beta_scaled[X][j]) + '\t' + '\t'.join("{:.6f}".format(thi) for thi in LT[j]) + '\n')

        return 0
        
    # run these calculations in a parallelized loop (bypass Parallel() if only one core is requested, in which case joblib-import could be dropped above)
    if Ncores == 1 and Nbeta >=1:
        [maincomputations(i) for i in range(len(metal))]
    elif Nbeta<1:
        print("skipping line tension calculations, Nbeta>0 required")
    else:
        Parallel(n_jobs=Ncores)(delayed(maincomputations)(i) for i in range(len(metal)))

################## create plots ################
    if skip_plots:
        print("skipping plots as requested")
        plt_metal = []
    else:
        plt_metal = metal
    skip_plt = []

## load data from LT calculation
    LT = {}
    theta_plt = {}
    Ntheta_plt = {}
    beta_plt = {}
    for X in plt_metal:
        ## for every X, LT has shape (len(theta+1),Nbeta), first column is beta all others are LT for various dislocation types theta in the range -pi/2 to  pi/2
        try:
            with open("LT_{}.dat".format(X),"r") as LTfile:
                lines = list(line.rstrip() for line in LTfile)
                ### first read theta from file (already known, but make this code independent from above)
                theta_plt[X] = np.pi*np.asarray(lines[1].split()[1:],dtype='float')
                Ntheta_plt[X] = len(theta_plt[X])
                ### determine length of beta from file
                Nbeta_plt = len(lines)-2
                ### read beta vs drag coeff from file:
                LT[X] = np.zeros((Nbeta_plt,Ntheta_plt[X]+1))
                for j in range(Nbeta_plt):
                    LT[X][j] = np.asarray(lines[j+2].split(),dtype='float')
                beta_plt[X] = LT[X][:,0]
        except FileNotFoundError:
            skip_plt.append(X)
            
    def mkLTplots(X):
        namestring = "{}".format(X)
        beta_trunc = [j for j in beta_plt[X] if j <=vcrit_smallest[X]]
        if X in metal_symm:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,3.2))
            LT_trunc = LT[X][:len(beta_trunc),int((Ntheta_plt[X]+1)/2):]
            y_msh , x_msh = np.meshgrid(theta_plt[X][int((Ntheta_plt[X]-1)/2):],beta_trunc)
            plt.yticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        else:
            fig, ax = plt.subplots(1, 1, sharey=False, figsize=(4.5,4.5))
            LT_trunc = LT[X][:len(beta_trunc),1:]
            y_msh , x_msh = np.meshgrid(theta_plt[X],beta_trunc)
            plt.yticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$-\pi/2$", r"$-3\pi/8$", r"$-\pi/4$", r"$-\pi/8$", r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"),fontsize=fntsize)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if X=='ISO':
            plt.xlabel(r'$\beta_\mathrm{t}=v/c_\mathrm{t}$',fontsize=fntsize)
            plt.title(r'Isotropic',fontsize=fntsize)
        else:
            plt.xlabel(r'$\beta_{\bar\mu}$',fontsize=fntsize)
            plt.title(namestring,fontsize=fntsize)
        plt.ylabel(r'$\vartheta$',rotation=0,fontsize=fntsize)
        colmsh = plt.pcolormesh(x_msh,y_msh,LT_trunc,vmin=-0.5, vmax=2,cmap = plt.cm.rainbow)
        plt.colorbar()
        plt.contour(x_msh,y_msh,LT_trunc, colors=('black','red','black','black','black','black'), levels=[-0.5,0,0.5,1,1.5,2], linewidths=[0.7,1.0,0.7,0.7,0.7,0.7], linestyles=['solid','solid','dashed','dashdot','dotted','solid'])
        colmsh.set_rasterized(True)
        plt.axhline(0, color='grey', linewidth=0.5, linestyle='dotted')
        plt.savefig("LT_{}.pdf".format(X),format='pdf',bbox_inches='tight',dpi=450)
        plt.close()

    for X in set(plt_metal).difference(set(skip_plt)):
        mkLTplots(X)
    
################################################    
    if Ntheta2==0 or Ntheta2==None:
        sys.exit()
    
    print("Computing critical velocities for: ",metal)
    
    ## if Ncores was set automatically considering the number of OpenMP threads, increase it here (no OpenMP support in computevcrit):
    if dlc.ompthreads > 1 and Ncores == max(1,int(Ncpus/max(2,dlc.ompthreads))):
        Ncores = max(1,int(Ncpus/2))
    ### setup predefined slip-geometries for symbolic calculations:
    b_pre = [np.array([1,1,0])/sp.sqrt(2), np.array([1,-1,1])/sp.sqrt(3), np.array([1,-1,1])/sp.sqrt(3), np.array([1,-1,1])/sp.sqrt(3)]
    n0_pre = [-np.array([1,-1,1])/sp.sqrt(3), np.array([1,1,0])/sp.sqrt(2), np.array([1,-1,-2])/sp.sqrt(6), np.array([1,-2,-3])/sp.sqrt(14)]
    for X in metal:
        for i in range(len(b_pre)):
            if abs(np.dot(b_pre[i],Y[X].b)-1)<1e-15 and abs(np.dot(n0_pre[i],Y[X].n0)-1)<1e-15:
                Y[X].b=b_pre[i]
                Y[X].n0=n0_pre[i] ## improve speed and accuracy below by using sympy sqrt for the normalization of these vectors
        
    vcrit = {} # values contain critical velocities and associated polar angles
    bt2_cache = []
    for X in metal:
        if X in metal_symm:
            current_symm=True
        else:
            current_symm=False
        foundcache=False
        for i in range(len(bt2_cache)):
            if abs(np.dot(bt2_cache[i][0],Y[X].b)-1)<1e-15 and abs(np.dot(bt2_cache[i][1],Y[X].n0)-1)<1e-15:
                foundcache=i
                break
        if Y[X].sym=='iso': print("skipping isotropic {}, vcrit=ct".format(X))
        elif foundcache is False:
            Y[X].computevcrit(Ntheta2,symmetric=current_symm,cache=True)
            bt2_cache.append((Y[X].b,Y[X].n0,Y[X].cache_bt2)) ## avoid expensive repeated calculations of the same bt2: store in bt2_cache for reuse
            vcrit[X]=Y[X].vcrit
        else:
            Y[X].computevcrit(Ntheta2,symmetric=current_symm,cache=bt2_cache[foundcache][2])
            vcrit[X]=Y[X].vcrit

    ## write vcrit results to disk, then plot
    with open("vcrit.dat","w") as vcritfile:
        vcritfile.write("theta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*Ntheta2-1)) + '\n')
        vcritfile.write("metal / vcrit[m/s] (3 solutions per angle)\n")
        for X in sorted(list(set(metal).intersection(vcrit.keys()))):
            for i in range(3):
                vcritfile.write("{}\t".format(X) + '\t'.join("{:.0f}".format(thi) for thi in np.flipud(vcrit[X][0,:,i])) + '\n')
        vcritfile.write("\ntheta/pi\t" + '\t'.join("{:.4f}".format(thi) for thi in np.linspace(1/2,-1/2,2*Ntheta2-1)) + '\n')
        vcritfile.write("metal / phi(vcrit)[pi] - polar angle at which vcrit occurs (3 solutions per theta)\n")
        for X in sorted(list(set(metal).intersection(vcrit.keys()))):
            for i in range(3):
                vcritfile.write("{}\t".format(X) + '\t'.join("{:.4f}".format(thi) for thi in np.flipud(vcrit[X][1,:,i]/np.pi)) + '\n')
                
    def mkvcritplot(X,vcrit,Ntheta):
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]}, figsize=(5.5,6.0))
        plt.tight_layout(h_pad=0.0)
        plt.xticks(fontsize=fntsize)
        plt.yticks(fontsize=fntsize)
        vcrit0 = vcrit[0]
        vcrit1 = vcrit[1]/np.pi
        if len(vcrit0)==Ntheta:
            plt.xticks([0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(0,np.pi/2,Ntheta)
        else:
            plt.xticks([-np.pi/2,-3*np.pi/8,-np.pi/4,-np.pi/8,0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2],(r"$\frac{-\pi}{2}$", r"$\frac{-3\pi}{8}$", r"$\frac{-\pi}{4}$", r"$\frac{-\pi}{8}$", r"$0$", r"$\frac{\pi}{8}$", r"$\frac{\pi}{4}$", r"$\frac{3\pi}{8}$", r"$\frac{\pi}{2}$"),fontsize=fntsize)
            thetapoints = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
        ax1.axis((min(thetapoints),max(thetapoints),np.min(vcrit0)*0.97,np.max(vcrit0)*1.02)) ## define plot range
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.set_ylabel(r'$v_\mathrm{c}$[m/s]',fontsize=fntsize)
        ax1.set_title("3 vcrit solutions for {}".format(X),fontsize=fntsize)
        plt.setp(ax1.get_xticklabels(), visible=False)
        if np.all(np.round(np.max(vcrit0,axis=1),6)==round(np.max(vcrit0,axis=1)[0],6)) and np.all(np.round(np.min(vcrit0,axis=1),6)==round(np.min(vcrit0,axis=1)[0],6)):
            vcrit0 = np.sort(vcrit0)
        for i in range(3):
            ax1.plot(thetapoints,vcrit0[:,i])
        ##
        ax2.axis((min(thetapoints),max(thetapoints),np.min(vcrit1)*0.97,np.max(vcrit1)*1.02)) ## define plot range
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.set_xlabel(r'$\vartheta$',fontsize=fntsize)
        ax2.set_ylabel(r'$\phi(v_\mathrm{c})/\pi$',fontsize=fntsize)
        if np.all(np.round(np.max(vcrit1,axis=1),6)==round(np.max(vcrit1,axis=1)[0],6)) and np.all(np.round(np.min(vcrit1,axis=1),6)==round(np.min(vcrit1,axis=1)[0],6)):
            vcrit1 = np.sort(vcrit1)
        for i in range(3):
            ax2.plot(thetapoints,vcrit1[:,i])
        plt.savefig("vcrit_{}.pdf".format(X),format='pdf',bbox_inches='tight')
        plt.close()
    
    vcrit_smallest_new = {}
    for X in sorted(list(set(metal).intersection(vcrit.keys()))):
        mkvcritplot(X,vcrit[X],Ntheta2)
        ## (re-)compute smallest critical velocities: needs high resolution in Ntheta2 to be accurate
        vcrit_smallest_new[X] = np.min(vcrit[X][0]) ## in m/s suitable for input files
        # vcrit_smallest_new[X] = np.min(vcrit[X][0])*np.sqrt(Y[X].rho/Y[X].c44) ## scaled by c44 (for input from metal_data, this is what was hard coded above for speed)
        
