# Compute various properties of a moving dislocation
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Apr. 23, 2025
'''This submodule contains the Dislocation class which inherits from the StrohGeometry class and the metal_props class.
   As such, it is the most complete class to compute properties of dislocations, both steady state and accelerating.
   Additionally, the Dislocation class can calculate properties like limiting velocities of dislocations. We also define
   a function, readinputfile, which reads a PyDislocDyn input file and returns an instance of the Dislocation class.'''
#################################
import numpy as np
import sympy as sp
from mpmath import findroot
from scipy import optimize, integrate
from ..utilities import jit, rotaround, heaviside, deltadistri, elbrak1d, roundcoeff, \
    fntsettings, mpl, plt ## =matplotlib.pyplot
from ..elasticconstants import Voigt, UnVoigt, CheckReflectionSymmetry
from ..crystals import metal_props, loadinputfile
from .steadystate import StrohGeometry

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

class Dislocation(StrohGeometry,metal_props):
    '''This class has all properties and methods of classes StrohGeometry and metal_props, as well as some additional methods: computevcrit, findvcrit_smallest, findRayleigh.
       If optional keyword Miller is set to True, b and n0 are interpreted as Miller indices (and Cartesian otherwise); note since n0 defines a plane its Miller indices are in reziprocal space.'''
    def __init__(self,b, n0, theta=[0,np.pi/2], Nphi=500,sym='iso', name='some_crystal',Miller=False,lat_a=None,lat_b=None,lat_c=None,lat_alpha=None,lat_beta=None,lat_gamma=None):
        metal_props.__init__(self, sym, name)
        if lat_a is not None: self.ac=lat_a
        if lat_b is not None: self.bc=lat_b
        if lat_c is not None: self.cc=lat_c
        if lat_alpha is not None: self.alphac=lat_alpha
        if lat_beta is not None: self.betac=lat_beta
        if lat_gamma is not None: self.gammac=lat_gamma
        if Miller:
            self.Millerb = b
            b = self.Miller_to_Cart(self.Millerb)
            self.Millern0 = n0
            n0 = self.Miller_to_Cart(self.Millern0,reziprocal=True)
        StrohGeometry.__init__(self, b, n0, theta, Nphi)
        self.vcrit_smallest=self.vcrit_screw=self.vcrit_edge=None
        self.C2_aligned_screw = self.C2_aligned_edge = None
        self.sym = sym
        self.vcrit_barnett = None
        self.vcrit_all = None
        self.Rayleigh = None
        self.vRF = None
    
    def alignC2(self):
        '''Calls self.computerot() and then computes the rotated SOEC tensor C2_aligned in coordinates aligned with the slip plane for each character angle.'''
        self.computerot()
        if self.C2.dtype == np.dtype('O'):
            self.C2_aligned = np.zeros((self.Ntheta,6,6),dtype=object)
        else:
            self.C2_aligned = np.zeros((self.Ntheta,6,6)) ## compute C2 rotated into dislocation coordinates
        for th in range(self.Ntheta):
            if self.sym=='iso':
                self.C2_aligned[th] = self.C2 ## avoids rounding errors in the isotropic case where C2 shouldn't change
            else:
                self.C2_aligned[th] = Voigt(np.dot(self.rot[th],np.dot(self.rot[th],np.dot(self.rot[th],np.dot(UnVoigt(self.C2),self.rot[th].T)))))
    
    def computevcrit_barnett(self, theta_list=None, setvcrit=True, verbose=False):
        '''Computes the limiting velocities following Barnett et al., J. Phys. F, 3 (1973) 1083, sec. 5.
           All parameters are optional: unless a list of character angles is passed explicitly via theta_list,
           we calculate limiting velocities for all character angles in self.theta.
           Option setvcrit determines whether or not to overwrite attribute self.vcrit_barnett.
           Note that this method does not check for subtle cancellations that may occur in the dislocation displacement gradient at those velocities;
           use the frontend method .computevcrit(theta) for fully automated determination of the lowest critical velocity at each character angle.'''
        norm=(self.C2[3,3]/self.rho)
        C2 = UnVoigt(self.C2/self.C2[3,3])
        if theta_list is None:
            Ntheta = self.Ntheta
            m0 = self.m0
            theta = self.theta
        else:
            Ntheta = len(theta_list)
            theta = np.asarray(theta_list)
            t = np.outer(np.cos(theta),self.b) + np.outer(np.sin(theta),np.cross(self.b,self.n0))
            m0 = np.cross(self.n0,t)
        out = np.zeros((2,Ntheta,3))
        for th in range(Ntheta):
            def findvlim(phi,i):
                M = m0[th]*np.cos(phi) + self.n0*np.sin(phi)
                MM = np.dot(M,np.dot(C2,M))
                P = -np.trace(MM)
                Q = 0.5*(P**2-np.trace((MM @ MM)))
                # R = -np.linalg.det(MM)
                R = -(MM[0,0]*MM[1,1]*MM[2,2] + MM[0,2]*MM[1,0]*MM[2,1] + MM[0,1]*MM[1,2]*MM[2,0] \
                      - MM[0,2]*MM[1,1]*MM[2,0] - MM[0,0]*MM[1,2]*MM[2,1] - MM[0,1]*MM[1,0]*MM[2,2])
                a = Q - P**2/3
                d = (2*P**3-9*Q*P+27*R)/27
                gamma = np.arccos(-0.5*d/np.sqrt(-a**3/27))
                tmpout = -P/3 + 2*np.sqrt(-a/3)*np.cos((gamma+2*i*np.pi)/3)
                return np.abs(np.sqrt(tmpout*norm)/np.cos(phi))
            for i in range(3):
                ## default minimizer sometimes yields nan, but bounded method doesn't always find the smallest value, so run both:
                with np.errstate(invalid='ignore'): ## don't need to know about arccos producing nan while optimizing
                    minresult1 = optimize.minimize_scalar(findvlim,bounds=(0,2.04*np.pi),args=i) # slightly enlarge interval for better results despite rounding errors in some cases
                    minresult2 = optimize.minimize_scalar(findvlim,method='bounded',bounds=(0,2.04*np.pi),args=i)
                if verbose and not (minresult1.success and minresult2.success):
                    print(f"Warning ({self.name}, theta={theta[th]}):\n{minresult1}\n{minresult2}\n\n")
                ## always take the smaller result, ignore nan:
                choose = np.nanargmin(np.array([minresult1.fun,minresult2.fun]))
                if choose == 0:
                    minresult = minresult1
                else: minresult = minresult2
                out[0,th,i] = minresult.fun
                out[1,th,i] = minresult.x
        if setvcrit: self.vcrit_barnett = out
        return out[0]
        
    def computevcrit_screw(self):
        '''Compute the limiting velocity of a pure screw dislocation analytically, provided the slip plane is a reflection plane, use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        self.C2_aligned_screw = self.C2_aligned[self.findedgescrewindices()[0]]
        A = self.C2_aligned_screw[4,4]
        B = 2*self.C2_aligned_screw[3,4]
        C = self.C2_aligned_screw[3,3]
        if CheckReflectionSymmetry(self.C2_aligned_screw):
            self.vcrit_screw = np.sqrt((A-B**2/(4*C))/self.rho)
        return self.vcrit_screw
    
    def computevcrit_edge(self):
        '''Compute the limiting velocity of a pure edge dislocation analytically, provided the slip plane is a reflection plane (cf. L. J. Teutonico 1961, Phys. Rev. 124:1039), use computevcrit() otherwise.'''
        if self.C2_aligned is None:
            self.alignC2()
        self.C2_aligned_edge = self.C2_aligned[self.findedgescrewindices()[1]]
        if CheckReflectionSymmetry(self.C2_aligned_edge):
            c11=self.C2_aligned_edge[0,0]
            c22=self.C2_aligned_edge[1,1]
            c66=self.C2_aligned_edge[5,5]
            c12=self.C2_aligned_edge[0,1]
            test = np.abs(self.C2_aligned_edge/self.C2[3,3]) ## check for additional symmetry requirements
            if test[0,5] + test[1,5] < 1e-12:
                self.vcrit_edge = np.sqrt(min(c66,c11)/self.rho)
                ## cover case of q<0 (cf. Teutonico 1961 paper, eq (39); line above was only for q>0):
                if ((c11*c22-c12**2-2*c12*c66) - (c22+c66)*min(c66,c11))/(c22*c66)<0:
                    ## analytic solution to Re(lambda=0) in eq. (39) (with sp.solve); sqrt below is real because of if statement above:
                    minval = (2*np.sqrt(c22*c66*(-c11*c22 + c11*c66 + c12**2 + 2*c12*c66 + c22*c66))*(c12 + c66) - (-c11*c22**2 + c11*c22*c66 + c12**2*c22 + c12**2*c66 + 2*c12*c22*c66 + 2*c12*c66**2 + 2*c22*c66**2))/((c22 - c66)**2)
                    self.vcrit_edge = min(self.vcrit_edge,np.sqrt(minval/self.rho))
            else:
                c16 = self.C2_aligned_edge[0,5]
                c26 = self.C2_aligned_edge[1,5]
                def theroot(y,rv2):
                    return accedge_theroot(y,rv2,c11,c12,c16,c22,c26,c66)
                y,rv2 = sp.symbols('y,rv2')
                ysol = sp.solve(theroot(y,rv2),y) ## 4 complex roots as fcts of rv2=rho*v**2
                yfct=sp.lambdify(rv2,ysol,modules=[np.emath,'scipy'])
                @np.vectorize
                def f(x):
                    return np.abs(np.asarray(yfct(x)).imag.prod()) ## lambda=i*y, and any Re(lambda)=0 implies a divergence/limiting velocity
                with np.errstate(invalid='ignore'):
                    rv2limit_sol = optimize.root(f,x0=1e5)
                    rv2limit = rv2limit_sol.x
                    if rv2limit_sol.success and len(rv2limit_sol.x)==1:
                        self.vcrit_edge = np.sqrt(rv2limit[0]/self.rho)
                    else:
                        print(f'Warning: {self.name}.computevcrit_edge() (resp. scipy.optimize.root()) failed, debug info: {rv2limit_sol=}, {np.sqrt(rv2limit[0]/self.rho)=}, {f(rv2limit)=}')
        return self.vcrit_edge

    def computevcrit(self,theta=None,set_screwedge=True,setvcrit=True):
        '''Compute the lowest critical (or limiting) velocities for all dislocation character angles within list 'theta'. If theta is omitted, we fall back to attribute .theta (default).
        The list of results will be stored in method .vcrit_all, i.e. .vcrit_all[0]=theta and .vcrit_all[1] contains the corresponding lowest limiting velocities.
        Additionally, .vcrit_all[3] contains the highest critical velocities and .vcrit_all[2] contains the intermediate critical velocities.
        Option set_screwedge=True guarantees that attributes .vcrit_screw and .vcrit_edge will be set, and 'setvrit=True' will overwrite self.vcrit_barnett.'''
        if theta is None:
            theta=self.theta
        indices = self.findedgescrewindices(theta)
        self.vcrit_all = np.empty((4,len(theta)))
        self.vcrit_all[0] = theta
        if self.sym=='iso':
            self.computevcrit_screw()
            self.vcrit_all[1:] = self.vcrit_screw
            if self.cl==0:
                self.init_all()
            self.vcrit_all[3] = self.cl
        else:
            self.vcrit_all[1:] = np.sort(self.computevcrit_barnett(theta_list=np.asarray(theta),setvcrit=setvcrit),axis=1).T
        if indices[0] is not None:
            self.computevcrit_screw()
            if CheckReflectionSymmetry(self.C2_aligned_screw):
                self.vcrit_all[1:,indices[0]] = self.vcrit_screw
            elif set_screwedge:
                self.vcrit_screw = self.vcrit_all[1,indices[0]]
        if indices[1] is not None:
            self.computevcrit_edge()
            if CheckReflectionSymmetry(self.C2_aligned_edge) and self.vcrit_edge is not None:
                self.vcrit_all[2,indices[1]] = self.vcrit_all[1,indices[1]] = self.vcrit_edge
                if len(indices) == 3:
                    self.vcrit_all[2,indices[2]] = self.vcrit_all[1,indices[2]] = self.vcrit_edge
            elif set_screwedge:
                self.vcrit_edge = self.vcrit_all[1,indices[1]]
                if len(indices) == 3:
                    self.vcrit_edge = min(self.vcrit_edge,self.vcrit_all[1,indices[2]])
        return self.vcrit_all[1]
    
    def findvcrit_smallest(self,xatol=1e-2):
        '''Computes the smallest critical velocity, which subsequently is stored as attribute .vcrit_smallest and the full result of scipy.minimize_scalar is returned
           (as type 'OptimizeResult' with its 'fun' being vcrit_smallest and 'x' the associated character angle theta).
           The absolute tolerance for theta can be passed via xatol; in order to improve accuracy and speed of this routine, we make use of computevcrit with Ntheta>=11 resolution
           in order to be able to pass tighter bounds to the subsequent call to minimize_scalar(). If .vcrit_all already exists in sufficient resolution from an earlier call,
           this step is skipped.'''
        backupvcrit = self.vcrit_all
        if self.vcrit_all is None or self.vcrit_all.shape[1]<11:
            self.computevcrit(theta=np.linspace(self.theta[0],self.theta[-1],11),set_screwedge=False,setvcrit=False)
        vcrit_smallest = np.nanmin(self.vcrit_all[1])
        thind = np.where(self.vcrit_all[1]==vcrit_smallest)[0][0] ## find index of theta so that we may pass tighter bounds to minimize_scalar below for more accurate (and faster) results
        bounds=(max(-np.pi/2,self.vcrit_all[0][max(0,thind-1)]),min(np.pi/2,self.vcrit_all[0][min(thind+1,len(self.vcrit_all[0])-1)]))
        def f(x):
            return np.min(self.computevcrit_barnett(theta_list=[x],setvcrit=False))
        if self.sym=='iso':
            result = vcrit_smallest
        else: result = optimize.minimize_scalar(f,method='bounded',bounds=bounds,options={'xatol':xatol})
        if self.sym=='iso':
            self.vcrit_smallest = vcrit_smallest
        elif result.success: self.vcrit_smallest = min(result.fun,vcrit_smallest)
        if backupvcrit is not None: self.vcrit_all = backupvcrit ## don't change vcrit_all, restore from our backup
        return result
    
    def findRayleigh(self):
        '''Computes the Rayleigh wave speed for every dislocation character self.theta.'''
        Rayleigh=np.zeros((self.Ntheta))
        norm = self.C2[3,3] # use c44
        C2norm = UnVoigt(self.C2/norm)
        if self.vcrit_all is None or len(self.vcrit_all[0])!=self.Ntheta or np.any(self.vcrit_all[0]!=self.theta):
            self.computevcrit(set_screwedge=False) ## need it as an upper bound on the Rayleigh speed
        vcrit = self.vcrit_all[1]
        for th in range(self.Ntheta):
            def Rayleighcond(B):
                return abs((B[0,0]+B[1,1])/2-np.sqrt((B[0,0]-B[1,1])**2/4 + B[0,1]**2))
            def findrayleigh(x):
                tmpC = C2norm - self.Cv[:,:,:,:,th]*x**2
                M=self.M[:,th].T
                N=self.N[:,th].T
                MM = elbrak1d(M,M,tmpC)
                MN = elbrak1d(M,N,tmpC)
                NM = elbrak1d(N,M,tmpC)
                NN = elbrak1d(N,N,tmpC)
                NNinv = np.linalg.inv(NN)
                S = - NNinv @ NM
                B = MM + MN @ S
                return Rayleighcond(integrate.trapezoid(B,x=self.phi,axis=0)/(4*np.pi**2))
            bounds=(0.0,vcrit[th]*np.sqrt(self.rho/norm))
            result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=bounds,options={'xatol':1e-12})
            # if result.fun>=1e-3: print(f"{bounds}\n{result}")  ## if this failed, try enlarging the search interval slightly above vcrit (there was some numerical uncertainty there too):
            if result.success and result.fun>=1e-3: result = optimize.minimize_scalar(findrayleigh,method='bounded',bounds=(0.5*bounds[1],1.25*bounds[1]),options={'xatol':1e-12})
            if result.fun>=1e-3 or not result.success: print(f"Failed: Rayleigh not found in [{bounds[0]},{1.25*bounds[1]}]\n",result)
            if result.success and result.fun<1e-3: Rayleigh[th] = result.x * np.sqrt(norm/self.rho)
        self.Rayleigh = Rayleigh
        return Rayleigh
    
    def computeuij_acc_screw(self,a,beta,burgers=None,rho=None,C2_aligned=None,phi=None,r=None,eta_kw=None,etapr_kw=None,t=None,shift=None,**kwargs):
        '''Computes the displacement gradient of an accelerating screw dislocation (based on  J. Mech. Phys. Solids 152 (2021) 104448, resp. arxiv.org/abs/2009.00167).
           For now, it is implemented only for slip systems with the required symmetry properties, that is the plane perpendicular to the dislocation line must be a reflection plane.
           In particular, a=acceleration, beta=v/c_A is a normalized velocity where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a),
           and normalization c_A defaults to sqrt(C2_aligned[4,4]/rho) which can be overridden by setting the option beta_normalization.
           Keywords burgers and rho denote the Burgers vector magnitude and material density, respectively.
           C2_aligned is the tensor of SOECs in Voigt notation rotated into coordinates aligned with the dislocation.
           r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core, i.e.
           x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi).
           Finally, more general dislocation motion can be defined via function eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
           likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
           Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
        self.beta = beta
        scrind = self.findedgescrewindices()[0]
        if C2_aligned is None:
            C2_aligned = self.C2_aligned
        elif C2_aligned.shape==(6,6): ## check if we received C2_aligned only for screw rather than all characters
            C2_aligned=[C2_aligned]
            scrind=0
        if burgers is None:
            burgers = self.burgers
        else:
            self.burgers = burgers
        if r is None:
            if self.r is None:
                r=burgers*np.linspace(0,1,250)
            else: r = burgers*self.r
        else:
            self.r = r/burgers
        if rho is None:
            rho = self.rho
        else:
            self.rho = rho
        if phi is None: phi=self.phi
        test = np.abs(self.C2_aligned[scrind]/self.C2_aligned[scrind,3,3]) ## check for symmetry requirements
        if test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4] > 1e-12:
            raise ValueError("not implemented - slip plane is not a reflection plane")
        ## change sign to match Stroh convention of steady state counter part:
        self.uij_acc_screw_aligned = -computeuij_acc_screw(a,beta,burgers,C2_aligned[scrind],rho,phi,r,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift,**kwargs)

    def computeuij_acc_edge(self,a,beta,burgers=None,rho=None,C2_aligned=None,phi=None,r=None,eta_kw=None,etapr_kw=None,t=None,shift=None,beta_normalization=1,force_static=False,**kwargs):
        '''Computes the displacement gradient of an accelerating edge dislocation.
           For now, it is implemented only for slip systems with the required symmetry properties, that is the plane perpendicular to the dislocation line must be a reflection plane.
           In particular, a=acceleration, beta=v/c_A is a normalized velocity where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a).,
           and normalization c_A defaults to sqrt(C2_aligned[3,3]/rho) which can be overridden by setting the option beta_normalization.
           Keywords burgers and rho denote the Burgers vector magnitude and material density, respectively.
           C2_aligned is the tensor of SOECs in Voigt notation rotated into coordinates aligned with the dislocation.
           r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core, i.e.
           x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi).
           Finally, more general dislocation motion can be defined via function eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
           likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
           Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
        edgeind = self.findedgescrewindices()[1]
        if C2_aligned is None:
            C2_aligned = self.C2_aligned
        elif C2_aligned.shape==(6,6): ## check if we received C2_aligned only for edge rather than all characters
            C2_aligned=[C2_aligned]
            edgeind=0
        if burgers is None:
            burgers = self.burgers
        else:
            self.burgers = burgers
        if r is None:
            if self.r is None:
                r=burgers*np.linspace(0,1,250)
            else: r = burgers*self.r
        else:
            self.r = r/burgers
        if rho is None:
            rho = self.rho
        else:
            self.rho = rho
        if phi is None: phi=self.phi
        test = np.abs(self.C2_aligned[edgeind]/self.C2_aligned[edgeind,3,3]) ## check for symmetry requirements
        if test[0,3]+test[1,3]+test[0,4]+test[1,4]+test[5,3]+test[5,4] > 1e-12:
            raise ValueError("not implemented - slip plane is not a reflection plane")
        self.uij_acc_edge_aligned = computeuij_acc_edge(a,beta,burgers,C2_aligned[edgeind],rho,phi,r,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift,beta_normalization=beta_normalization,**kwargs)
        ## workaround while static part is not yet implemented:
        if beta_normalization == 1:
            C2 = UnVoigt(self.C2/C2_aligned[edgeind][3,3])
        else:
            C2 = UnVoigt(self.C2/(self.rho*beta_normalization**2))
        if self.uij_static_aligned is None or force_static or self.uij_static_aligned.shape[-2] != len(r):
            self.computeuij(0,C2=C2,r=r/burgers)
            self.alignuij()
            self.uij_static_aligned = self.uij_aligned.copy()
        self.beta = beta
        self.uij_acc_edge_aligned += self.uij_static_aligned[:,:,edgeind]
        
    def find_vRF(self,fast=True,verbose=False,resolution=50,thetaind=None,partial_burgers=None):
        '''Compute the radiation-free velocity for dislocations (default: edge dislocations) in the transonic regime.
           For details on the method, see Gao, Huang, Gumbsch, Rosakis, JMPS 47 (1999) 1941.
           If the slip system has orthotropic symmetry, the analytic solution derived in that paper is used.
           Otherwise, we use a numerical method.
           In the first transonic regime, there may be a range of radiation free velocities; option resolution
           determines how many values to probe for in this regime. In this regime, we also support searching for
           radiation-free velocities for mixed dislocations: option 'thetaind' may be any index pointing to an
           element of self.theta. Furthermore, partial_burgers may be passed using Miller indices to check for vRF
           of a partial dislocation (an instance of the Dislocation class always represents a perfect dislocation
           with normalized Burgers vector self.b).
           Option 'fast=False' is only used for orthotropic slip systems in which case it will bypass the
           analytic solution in favor of the numerical one in order to facilitate testing the latter;
           this option may be removed in future versions.'''
        if self.C2_aligned is None:
            self.alignC2()
        if self.vcrit_all is None or len(self.vcrit_all[0])!=self.Ntheta or np.any(self.vcrit_all[0]!=self.theta):
            self.computevcrit()
        if thetaind is None:
            edgind = self.findedgescrewindices()[1]
        else:
            edgind=thetaind
            fast=False ## always use numeric code for mixed or partial disloc.
        if partial_burgers is not None: fast=False
        c = self.C2_aligned[edgind]
        test = np.abs(c/self.C2[3,3]) ## check for symmetry requirements
        if fast and CheckReflectionSymmetry(c,strict=True) and test[0,5]+test[1,5]+test[2,5]+test[3,4] < 1e-12:
            self.vRF = out = np.sqrt((c[0,0]*c[1,1]-c[0,1]**2)/(c[0,1]+c[1,1])/self.rho)
            if verbose: print("orthotropic symmetry detected, using analytic solution")
        else:
            rv2 = sp.symbols('rv2') # abbreviation for rho*v^2;
            delta = np.identity(3,dtype=int)
            p = sp.symbols('p')
            m = np.array([1,0,0]) ## we already rotated C2 so that now x1 is in the direction of v and x2 is the slip plane normal
            n = np.array([0,1,0])
            l = m + sp.I*p*n
            norm = self.C2[3,3] ## use c44 to normalize some numbers below
            C2eC = UnVoigt(c/norm)
            if CheckReflectionSymmetry(c):
                delta = delta[:2,:2] ## only need 2x2 submatrix for edge in this case
                n = n[:2]
                l = l[:2]
                C2eC = C2eC[:2,:2,:2,:2]
            C2M = sp.simplify(sp.Matrix(np.dot(l,np.dot(C2eC,l)) - rv2*delta))
            thedet = sp.simplify(sp.det(C2M))
            fct = sp.lambdify((p,rv2),thedet,modules=[np.emath])
            vlim = self.vcrit_all[1:,edgind]
            burg = None
            if thetaind is not None:
                burg = self.rot[edgind] @ self.b
            if partial_burgers is not None:
                burg = self.rot[edgind] @ self.Miller_to_Cart(partial_burgers)
            if burg is not None:
                x = np.array([1,0,0])
                v = np.cross(x,burg)
                sv = np.sqrt(np.vdot(v,v))
                rot = rotaround(v/sv,sv,x@burg)
            bounds = (self.rho*(vlim[1])**2/norm,self.rho*(vlim[2])**2/norm)
            def L2_of_beta2(beta2,comp=1):
                '''Finds eigenvector L in the 2nd transonic regime; this function needs as input beta2 = (rho/c44)*v^2'''
                def f(x):
                    return float(np.abs(fct(x,beta2)))
                p1 = None
                for x0 in [0.5,1,1.5]:
                    psol = optimize.root(f,x0)
                    if psol.success:
                        p1 = abs(float(psol.x))
                        break
                if p1 is not None:
                    C2Mp = C2M.subs({rv2:beta2,p:p1})
                    Ak = C2Mp.eigenvects()
                    p0 = abs(Ak[0][0])
                    Ap = Ak[0][2][0]
                    if abs(Ak[1][0])<p0:
                        p0 = abs(Ak[1][0])
                        Ap = Ak[1][2][0]
                    if len(Ak)>2 and abs(Ak[2][0])<p0:
                        p0 = abs(Ak[2][0])
                        Ap = Ak[2][2][0]
                    if p0 > 1e-5:
                        Ap = np.repeat(np.inf,len(Ak))
                    L = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap)))).subs(p,p1).evalf()
                else:
                    L = [1e12,1e12] ## random high number so that minimize_scalar() below will not pick this velocity
                if comp=='all':
                    out = L.evalf()
                else:
                    out = float(np.abs(L[comp]))
                return out
            def L1_of_beta2(beta2,comp=1,burg=burg):
                '''Finds L1+L2 in the 1st transonic regime; this function needs as input beta2 = (rho/c44)*v^2'''
                @np.vectorize
                def f(x):
                    return float(np.abs(fct(x,beta2)))
                p1 = None
                p2 = None
                psol = optimize.root(f,np.array([0.9,1.5]))
                if psol.success:
                    if len(set(np.abs(np.round(psol.x,12))))==2:
                        p1 = abs(float(psol.x[0]))
                        p2 = abs(float(psol.x[1]))
                if p1 is not None and p2 is not None:
                    C2Mp1 = C2M.subs({rv2:beta2,p:p1})
                    C2Mp2 = C2M.subs({rv2:beta2,p:p2})
                    Ak1 = C2Mp1.eigenvects()
                    Ak2 = C2Mp2.eigenvects()
                    p01 = abs(Ak1[0][0])
                    Ap1 = Ak1[0][2][0]
                    p02 = abs(Ak2[0][0])
                    Ap2 = Ak2[0][2][0]
                    if abs(Ak1[1][0])<p01:
                        p01 = abs(Ak1[1][0])
                        Ap1 = Ak1[1][2][0]
                    if abs(Ak1[2][0])<p01:
                        p01 = abs(Ak1[2][0])
                        Ap1 = Ak1[2][2][0]
                    if p01 > 1e-5:
                        Ap1 = np.array([np.inf,np.inf,np.inf])
                    if abs(Ak2[1][0])<p02:
                        p02 = abs(Ak2[1][0])
                        Ap2 = Ak2[1][2][0]
                    if abs(Ak2[2][0])<p02:
                        p02 = abs(Ak2[2][0])
                        Ap2 = Ak2[2][2][0]
                    if p02 > 1e-5:
                        Ap2 = np.array([np.inf,np.inf,np.inf])
                    L1 = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap1)))).subs(p,p1).evalf()
                    L2 = sp.Matrix(-np.dot(n,np.dot(np.dot(C2eC,l),np.asarray(Ap2)))).subs(p,p2).evalf()
                    factor = complex(-L1[1]/L2[1])
                else:
                    L1 = L2 = sp.Matrix([1e12,1e12,1e12]) ## random high number so that minimize_scalar() below will not pick this velocity
                    factor = 1
                if comp=='all':
                    out = (L1+factor*L2).evalf()
                else:
                    L1plusL2 = np.array((L1+factor*L2).evalf(),dtype=complex)
                    if burg is not None:
                        L1plusL2 = rot @ L1plusL2
                    out = float(np.abs((L1plusL2)[1]) + abs((L1plusL2).imag[0]*(L1plusL2).real[2] - (L1plusL2).imag[2]*(L1plusL2).real[0]))
                return out
            def checkprops(vRF):
                '''checks properties (such as boundary conditions) for the solution-candidate vRF'''
                out = None
                tol = 1e-5
                if vRF is None: return out
                if vRF.success and vRF.fun < tol:
                    out_norm = np.sqrt(vRF.x*norm/self.rho)
                    checkL = np.array(roundcoeff(L2_of_beta2(vRF.x,comp='all'),int(-np.log10(tol))),dtype=complex)
                    if CheckReflectionSymmetry(c):
                        checkL[-1].imag = checkL[0].imag
                        checkL[-1].real=0 ## condition below reduces to Re(L[0])=0 or Im(L[0])=0 in this case
                    if not (abs(checkL[0].real*checkL[-1].imag - checkL[-1].real*checkL[0].imag)<tol) or sp.Matrix(checkL).norm()<tol:
                        if verbose: print(f"Error: eigenvector does not meet required properties: \n{checkL=}")
                    else:
                        # double check number of real eigenvalues (method above only works for one pair out of three)
                        checkdet = roundcoeff(thedet.subs(rv2,vRF.x))
                        eigenvals = np.array(roundcoeff(sp.Matrix(sp.solve(checkdet.subs({p**6:rv2**3,p**4:rv2**2,p**2:rv2}),rv2))),dtype=complex)
                        if (int(sum(eigenvals.real>0)) != 1) and (float(sum(eigenvals.imag**2)) > 1e-15):
                            print(f"Error: unexpected number of real eigenvalues detected: \n{eigenvals=}")
                            if verbose: print(f"{out_norm}")
                        else:
                            if verbose: print("success")
                            out = out_norm
                elif verbose: print(f"condition for disloc. glide not met (L[1]!=0);\n{vRF=} ")
                return out
            out = self.vRF = None
            if burg is None: #bypass for mixed or partial disloc. (not supported yet) TODO: check if we really never have a solution here for mixed dislocs.
                if verbose: print("searching in the 2nd transonic regime ...")
                vRF = optimize.minimize_scalar(L2_of_beta2,method='bounded',bounds=bounds)
                if (out:=checkprops(vRF)) is not None:
                    self.vRF = out
            bounds_fst = (self.rho*self.vcrit_edge**2/norm,self.rho*(vlim[1])**2/norm)
            if np.isclose(bounds_fst[0],bounds_fst[1]):
                if verbose and thetaind is None: print("found only one transonic regime for this gliding edge dislocation")
            else:
                if verbose: print("searching in the 1st transonic regime ...")
                vels = np.linspace(bounds_fst[0],bounds_fst[1],resolution)
                vRF_fst = []
                for v in vels:
                    if L1_of_beta2(v)<1e-9:
                        if (L1_of_beta2(v,'all').norm())>1e-9: # make sure we found a non-trivial eigenvector
                            vRF_fst.append(np.sqrt(v*norm/self.rho))
                if len(vRF_fst)==0:
                    vRF_fst = optimize.minimize_scalar(L1_of_beta2,method='bounded',bounds=bounds_fst)
                    if (vRF_fst.success and vRF_fst.fun < 1e-9):
                        vRF_fst = np.sqrt(vRF_fst.x*norm/self.rho)
                    else:
                        vRF_fst = None
                if vRF_fst is not None:
                    if verbose: print("success")
                    if self.vRF is None:
                        self.vRF = vRF_fst
                    else:
                        self.vRF = [vRF_fst,self.vRF]
                elif verbose: print("nothing found in 1st transonic regime")
            if self.vRF is None:
                print(f"Failed: could not find a solution for vRF of {self.name}")
        return self.vRF
    
    def plotdisloc(self,beta=None,character='screw',component=[2,0],a=None,eta_kw=None,etapr_kw=None,t=None,shift=None,fastapprox=False,Nr=250,nogradient=False,skipcalc=False,showplt=False,savefig=True,**kwargs):
        '''Calculates and generates a plot of the requested component of the dislocation displacement gradient; plotting is done by the function plotuij().
           Optional arguments are: the normalized velocity 'beta'=v/self.ct (defaults to self.beta, assuming one of the .computeuij() methods were called earlier).
           'character' is either 'edge', 'screw' (default), or an index of self.theta, and 'component' is
           a list of two indices indicating which component of displacement gradient u[ij] to plot.
           The steady-state solution is plotted unless an acceleration 'a' (or a more general function eta_kw) is passed. In the latter case,
           'slipsystem' is required except for those metals where its keyword coincides with self.sym (see documentation of self.computeuij_acc_screw()
           and self.computeuij_acc_edge() for details on capabilities and limitations of the current implementation of the accelerating solution).
           Option nogradient=True will plot the displacement field instead of its gradient; this option must be combined with an integer value for 'component'
           and is currently only implemented for steady-state solutions (a=None).
           Option skipcalc=True (implied when beta is not set) may be passed to plot results of an earlier calculation with the same input parameters (useful
           for plotting multiple components of the dislocation field).
           If option 'showplt' is set to 'True', the figure is shown in an interactive session in addition to being saved to a file. Warning: this will only work
           if the user sets matplotlib's backend to an interactive one after PyDislocDyn was loaded (e.g. by calling %matplotlib inline). Saving the figure to
           a file can be suppressed with option 'savefig=False'.
           See the documentation of plotting function plotuij() for additional options that may be passed to it via kwargs.'''
        if beta is None:
            beta = self.beta
            skipcalc = True
        ## make sure everything we need has been initialized:
        if self.ct==0:
            self.ct = np.sqrt(self.mu/self.rho)
        if np.count_nonzero(self.C2norm) == 0:
            self.C2norm = UnVoigt(self.C2/self.mu)
        if self.C2_aligned is None:
            self.alignC2()
        if skipcalc and self.r is not None:
            r = self.r
            Nr = len(r)
        else:
            r = np.linspace(0.0001,1,Nr)
        xylabel = {0:'x',1:'y',2:'z'}
        if a is None and eta_kw is None:
            if not skipcalc:
                if not nogradient:
                    self.computeuij(beta=beta)
                    self.alignuij() ## self.rot was computed as a byproduct of .alignC2() above
                else:
                    self.computeuk(beta=beta, r=r)
                    self.alignuk()
            if character == 'screw':
                index = self.findedgescrewindices()[0]
            elif character == 'edge':
                index = self.findedgescrewindices()[1]
            else:
                index=character
            if not nogradient:
                namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}{character}_{self.name}_v{beta*self.ct:.0f}"
                uijtoplot = self.uij_aligned[component[0],component[1],index]
                uijtoplot = np.outer(1/r,uijtoplot)
            else:
                namestring = f"u{xylabel[component]}{character}_{self.name}_v{beta*self.ct:.0f}"
                uijtoplot = self.uk_aligned[component,index]
        elif character=='screw' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_screw(a,beta,burgers=self.burgers,fastapprox=fastapprox,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None:
                acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}screw_{self.name}_v{beta*self.ct:.0f}_a{acc:}"
            uijtoplot = self.uij_acc_screw_aligned[component[0],component[1]]
        elif character=='edge' and not nogradient:
            if not skipcalc:
                self.computeuij_acc_edge(a,beta,burgers=self.burgers,r=r*self.burgers,beta_normalization=self.ct,eta_kw=eta_kw,etapr_kw=etapr_kw,t=t,shift=shift)
            if a is None:
                acc = '_of_t'
            else: acc = f"{a:.0e}"
            namestring = f"u{xylabel[component[0]]}{xylabel[component[1]]}edge_{self.name}_v{beta*self.ct:.0f}_a{acc:}"
            uijtoplot = self.uij_acc_edge_aligned[component[0],component[1]]
        else:
            raise ValueError("not implemented")
        if savefig: savefig=namestring+".pdf"
        plotuij(uijtoplot,r,self.phi,**kwargs,showplt=showplt,title=namestring,savefig=savefig)
        
    def __repr__(self):
        return "DISLOCATION\n" + metal_props.__repr__(self) + f"\n burgers:\t {self.burgers}\n" + StrohGeometry.__repr__(self)

def readinputfile(fname,init=True,theta=None,Nphi=500,Ntheta=2,symmetric=True,isotropify=False):
    '''Reads an inputfile like the one generated by writeinputfile() defined in metal_data.py (some of these data are only needed by other parts of PyDislocDyn),
       and returns a populated instance of the Dislocation class. If option init=True, some derived quantities are computed immediately via the classes .init_all() method.
       Array theta contains all dislocation characters to be considered, and integer Nphi denotes the resolution to be used for polar angle phi.
       Alternatively, instead of passing theta explicitly, the number of characters angles between 0 and pi/2 can be passed via keyword Ntheta.
       In this case, keyword 'symmetric' will determine whether the generated theta array ranges from 0 to pi/2 (True) or from -pi/2 to pi/2 (False).
       The latter keyword can also be read from file 'fname'.
       If option isotropify is set to True, we calculate isotropic averages of the elastic constants and return an instance of the Dislocation class
       with sym=iso and using those averages.'''
    inputparams = loadinputfile(fname)
    sym = inputparams['sym']
    name = inputparams.get('name',str(fname))
    if 'Millerb' in inputparams or 'Millern0' in inputparams:
        temp = metal_props(sym,name) ## need a metal_props method to convert to Cartesian b, n0
        temp.populate_from_dict(inputparams)
        b = temp.b
        n0 = temp.n0
    else:
        b = np.asarray(inputparams['b'].split(','),dtype=float)
        n0 = np.asarray(inputparams['n0'].split(','),dtype=float)
    if theta is None:
        symmetric = inputparams.get('symmetric',symmetric)
        if symmetric is True or symmetric == 'True' or Ntheta<=2:
            theta = np.linspace(0,np.pi/2,Ntheta)
        else: theta = np.linspace(-np.pi/2,np.pi/2,2*Ntheta-1)
    out = Dislocation(sym=sym, name=name, b=b, n0=n0, theta=theta, Nphi=Nphi)
    out.populate_from_dict(inputparams)
    out.filename = fname ## remember which file we read
    if isotropify and sym != 'iso': # bypass if we're already isotropic
        inputparams['sym'] = 'iso'
        inputparams['name'] = name+'_ISO'
        if 'lam' in inputparams: inputparams.pop('lam') ## ignore if read from file, use averages instead
        if 'mu' in inputparams: inputparams.pop('mu')
        out.lam = out.mu = None
        out.init_all()
        inputparams['c12'] = out.lam
        inputparams['c44'] = out.mu
        inputparams['a'] = np.cbrt(out.Vc)
        if 'c123' in inputparams:
            print("Warning: there is no good averaging scheme for TOECs, calculating (unreliable) Hill averages for the Murnaghan constants.")
            out_toec = out.compute_Lame(include_TOEC=True)
            inputparams['c123'] = float(out_toec['c123'])
            inputparams['c144'] = float(out_toec['c144'])
            inputparams['c456'] = float(out_toec['c456'])
        out = Dislocation(sym='iso', name=name+'_ISO', b=b, n0=n0, theta=theta, Nphi=Nphi)
        out.populate_from_dict(inputparams)
    if init:
        out.init_all()
        out.C2norm = UnVoigt(out.C2/out.mu)
    return out

@jit(nopython=True)
def accscrew_xyintegrand(x,y,t,xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw,xcomp):
    '''subroutine of computeuij_acc_screw'''
    Rpr = np.sqrt((x-xpr)**2 - (x-xpr)*y*B/C + y**2/Ct)
    if eta_kw is None:
        eta = np.sqrt(2*xpr/a)
        etatilde = np.sign(x)*np.sqrt(2*abs(x)/a)*0.5*(1+xpr/x)
    else:
        eta = eta_kw(xpr)
        etatilde = eta_kw(x) + (xpr-x)*etapr_kw(x)
    tau = t - eta
    tau_min_R = np.sqrt(abs(tau**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct = heaviside(t - eta - Rpr/(cA*np.sqrt(ABC)))
    tau2 = t - etatilde
    tau_min_R2 = np.sqrt(abs(tau2**2*ABC/Ct - Rpr**2/(Ct*cA**2)))
    stepfct2 = heaviside(t - etatilde - Rpr/(cA*np.sqrt(ABC)))
    if xcomp:
        integrand = stepfct*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R + tau**2*(ABC/Ct)/tau_min_R)
        integrand -= stepfct2*((x-xpr-y*B/(2*C))*y/Rpr**4)*(tau_min_R2 + tau2**2*(ABC/Ct)/tau_min_R2) ## subtract pole
    else:
        integrand = stepfct*(1/Rpr**4)*((tau**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R - (x-xpr)**2*(tau_min_R))
        integrand -= stepfct2*(1/Rpr**4)*((tau2**2*y**2*ABC/Ct**2 - (x-xpr)*y*(B/(2*C))*Rpr**2/(Ct*cA**2))/tau_min_R2 - (x-xpr)**2*tau_min_R2)
    return integrand

# @jit(nopython=True) ## cannot compile while using scipy.integrate.quad() inside this function
def computeuij_acc_screw(a,beta,burgers,C2_aligned,rho,phi,r,eta_kw=None,etapr_kw=None,t=None,shift=None,deltat=1e-3,fastapprox=False,beta_normalization=1,epsilon=2e-16):
    '''For now, only pure screw is implemented for slip systems with the required symmetry properties.
       a=acceleration, beta=v/c_A where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a).
       C2_aligned is the tensor of SOECs in Voigt notation rotated into coordinates aligned with the dislocation.
       Furthermore, x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi),
       i.e. r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core.
       Finally, more general dislocation motion can be defined via functions eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
       likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
       Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
    A = C2_aligned[4,4]
    B = 2*C2_aligned[3,4]
    C = C2_aligned[3,3]
    cA = np.sqrt(A/rho)
    ABC = 1-B**2/(4*A*C)
    Ct = C/A
    if beta_normalization==1:
        v = beta*cA
    else:
        v = beta*beta_normalization
    if eta_kw is None:
        t = v/a
        shift = a*t**2/2  ## = v**2/(2*a) # distance covered by the disloc. when achieving target velocity
        # print(f"time we reach {beta=}: {t=}; distance covered: {shift=}")
    uxz = np.zeros((len(r),len(phi),2))
    uyz = np.zeros((len(r),len(phi),2))
    R = np.zeros((len(r),len(phi)))
    X = np.zeros((len(r),len(phi)))
    Y = np.zeros((len(r),len(phi)))
    uij = np.zeros((3,3,len(r),len(phi)))
    ### integrate.quad options (to trade-off accuracy for speed in the kinetic eqns.)
    quadepsabs=1.49e-04 ## absolute error tolerance; default: 1.49e-08
    quadepsrel=1.49e-04 ## relative error tolerance; default: 1.49e-08
    quadlimit=30 ## max no of subintervals; default: 50
    ###
    tv = t*np.array([1-deltat/2,1+deltat/2])
    for ri, rx in enumerate(r):
        if abs(rx) < 1e-25:
            rx=1e-25
        for ph, phx in enumerate(phi):
            x = rx*np.cos(phx) + shift ### shift x to move with the dislocations
            y = rx*np.sin(phx)
            R[ri,ph] = np.sqrt(x**2 - x*y*B/C + y**2/Ct)
            X[ri,ph] = x
            Y[ri,ph] = y
            if not fastapprox: ## allow bypassing
                for ti in range(2):
                    uxz[ri,ph,ti] = integrate.quad(lambda xpr: accscrew_xyintegrand(x,y,tv[ti],xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw,True), 0, np.inf, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)[0]
                    uyz[ri,ph,ti] = integrate.quad(lambda xpr: accscrew_xyintegrand(x,y,tv[ti],xpr,a,B,C,Ct,ABC,cA,eta_kw,etapr_kw,False), 0, np.inf, epsabs=quadepsabs, epsrel=quadepsrel, limit=quadlimit)[0]
    ##
    if eta_kw is None:
        eta = np.sign(X)*np.sqrt(2*np.abs(X)/a)
        etapr = eta/(2*X)
        tau = t-0.5*eta
    else:
        eta = eta_kw(X)
        etapr = etapr_kw(X)
        tau = t-(eta-etapr*X)
    denom = tau*(tau-2*etapr*(X-Y*B/(2*C))) + (etapr*R)**2 - Y**2/(Ct*cA**2)
    heaviadd = heaviside(tau - R/(cA*np.sqrt(ABC)))
    rootadd = np.sqrt(np.abs(tau**2*ABC/Ct-R**2/(Ct*cA**2)))
    ## supersonic part:
    p0 = (X-Y*B/(2*C))/(R*cA*np.sqrt(1-B**2/(4*A*C)))
    with np.errstate(invalid='ignore'): ## don't need to know about nan from sqrt(-...) as these will always occur in the subsonic regime (and filtered with np.nan_to_num)
        deltaterms = (burgers/2)*np.sign(Y*B/(2*C)-X)*heaviside(p0/etapr-1)*np.nan_to_num(deltadistri(tau-(X-Y*B/(2*C))*etapr-np.abs(Y)*np.sqrt(1/(cA**2*Ct)-etapr**2*ABC/Ct),epsilon=epsilon))
    uxz_supersonic = deltaterms*np.sign(Y)*etapr
    uyz_supersonic = deltaterms*(np.sqrt(1/(cA*Ct)-etapr**2*ABC/Ct)-np.sign(Y)*B*etapr/(2*C))
    ##
    uxz_added = heaviadd*(Y/rootadd)\
        *(2*etapr*((X-Y*B/(2*C))/Ct)*(tau**2*ABC-(R**2)/(2*cA**2)) - tau*(tau**2-Y**2/(Ct*cA**2))*ABC/Ct)\
        /(R**2*(denom))
    uxz_static = Y*np.sqrt(ABC/Ct)/R**2
    uyz_added = (heaviadd/rootadd)\
        *(tau**2*(etapr)*ABC*(Y**2/Ct-X**2) + (X*etapr-tau)*(R**2/cA**2)*(X-Y*B/(2*C)) + X*tau*ABC*(tau**2-Y**2/(Ct*cA**2)))\
        /(R**2*Ct*(denom))
    uyz_static = - X*np.sqrt(ABC/Ct)/R**2
    uij[2,0] = (burgers/(2*np.pi))*((uxz[:,:,1]-uxz[:,:,0])/deltat + uxz_static + uxz_added) + uxz_supersonic
    uij[2,1] = (burgers/(2*np.pi))*((uyz[:,:,1]-uyz[:,:,0])/deltat + uyz_static + uyz_added) + uyz_supersonic
    return uij

def accedge_theroot(y,rv2,c11,c12,c16,c22,c26,c66):
    '''subroutine of computeuij_acc_edge()'''
    ## rv2 = rho/lambda^2
    ## y = mu/ (s lambda) and s lambda = i s alpha = i k, so Im(y)=0 leads to poles, i.e. equivalent to Re(mu)=0
    K4 = c66*c22-c26**2
    K3 = 2*(c26*c12-c16*c22)
    K2 = (c11*c22-c12**2-2*c12*c66+2*c16*c26) - (c22+c66)*rv2
    K1 = 2*(c16*c12-c26*c11) + 2*rv2*(c16+c26)
    K0 = (c11-rv2)*(c66-rv2)-c16**2
    return K0+K1*y+K2*y**2+K3*y**3+K4*y**4

def computeuij_acc_edge(a,beta,burgers,C2p,rho,phi,r,eta_kw=None,etapr_kw=None,t=None,shift=None,beta_normalization=1):
    '''For now, only pure edge is implemented for slip systems with the required symmetry properties.
       a=acceleration, beta=v/c_A where v=a*t (i.e. time is represented in terms of the current normalized velocity beta as t=v/a = beta*c_A/a).
       C2_aligned is the tensor of SOECs in Voigt notation rotated into coordinates aligned with the dislocation.
       Furthermore, x = r*cos(phi)+a*t**2/2 = r*cos(phi)+v**2/(2*a) = r*cos(phi)+(beta*c_A)**2/(2*a) and y=r*sin(phi),
       i.e. r, phi are polar coordinates in a frame moving with the dislocation so that r=0 represents its core.
       Finally, more general dislocation motion can be defined via functions eta_kw(x) (which is the inverse of core position as a function of time eta=l^{-1}(t)),
       likewise etapr_kw is the derivative of eta and is also a function. Acceleration a and velocity beta are ignored (and may be set to None) in this case.
       Instead, we require the time t at which to evaluate the dislocation field as well as the current dislocation core position 'shift' at time t.'''
    spmodules = ["mpmath"]
    if beta_normalization==1:
        v = beta*np.sqrt(C2p[3,3]/rho)
    else:
        v = beta*beta_normalization
    if eta_kw is None:
        t = v/a
        shift = a*t**2/2  ## = v**2/(2*a) # distance covered by the disloc. when achieving target velocity
        # print(f"time we reach {beta=}: {t=}; distance covered: {shift=}")
    y2,rv2,lambd, xs, ys, taus = sp.symbols('y2,rv2,lambd, xs, ys, taus')
    norm = C2p[3,3]
    C2p = C2p/norm
    ysol = sp.solve(accedge_theroot(y2,rv2,C2p[0,0],C2p[0,1],C2p[0,5],C2p[1,1],C2p[1,5],C2p[5,5]),y2) ## 4 complex roots as fcts of rv2=rho/lambda**2
    rv2subs = rho/lambd**2 / norm
    # ysol_all = [sp.lambdify((lambd),ysol[i].subs(rv2,rv2subs),modules=spmodules) for i in range(4)]
    # lameqn_all = [sp.lambdify((lambd,xs,ys,taus),lambd*xs + (ys*lambd*ysol[i].subs(rv2,rv2subs)) - taus,modules=spmodules,cse=True) for i in range(4)]
    # dLdT_all = [sp.lambdify((lambd,xs,ys), (1 / (xs+sp.diff(ys*lambd*ysol[i].subs(rv2,rv2subs),lambd))),modules=spmodules) for i in range(4)]
    ## work around a python 3.13 (or sympy?) bug where lambdify inside a certain list comprehension (see above) triggers Segmentation fault: 11
    ysol_all = [0,0,0,0]
    lameqn_all = [0,0,0,0]
    dLdT_all = [0,0,0,0]
    for i in range(4):
        ysol_all[i] = sp.lambdify((lambd),ysol[i].subs(rv2,rv2subs),modules=spmodules)
        lameqn_all[i] = sp.lambdify((lambd,xs,ys,taus),lambd*xs + (ys*lambd*ysol[i].subs(rv2,rv2subs)) - taus,modules=spmodules,cse=True)
        dLdT_all[i] = sp.lambdify((lambd,xs,ys), (1 / (xs+sp.diff(ys*lambd*ysol[i].subs(rv2,rv2subs),lambd))),modules=spmodules)
    # end of workaround
    uij = np.zeros((3,3,len(r),len(phi)))
    for ri, rx in enumerate(r):
        if abs(rx) < 1e-25:
            rx=1e-25
        # if abs(C2p[0,5])>1e-3 or abs(C2p[1,5])>1e-3: print(f"{rx=}")
        for ph, phx in enumerate(phi):
            x = -rx*np.cos(phx) + shift ### shift x to move with the dislocations (step fct automatically fulfilled near disloc. core)
            if np.abs(np.sin(phx)) < 1e-15:
                y = 0 ## avoid numerical issues for multiples of pi
            else:
                y = rx*np.sin(phx)
            if eta_kw is None:
                eta = np.sign(x)*np.sqrt(2*np.abs(x)/a)
                etapr = eta/(2*x)
                tau = t-0.5*eta
            else:
                eta = eta_kw(x)
                etapr = etapr_kw(x)
                tau = t-(eta-etapr*x)
            lam=np.zeros((4),dtype=complex)
            Jacobian=np.zeros((4),dtype=complex)
            mu=np.zeros((4),dtype=complex)
            muovlam=np.zeros((4),dtype=complex)
            coeff = np.ones((4),dtype=bool)
            for i in range(4):
                lamsol = findroot(lambda lamb: lameqn_all[i](lamb,x,y,tau),0.001+0.001j,solver='muller') ## solver recommended in docs for complex roots
                lam[i] = complex(lamsol)
                muovlam[i] = complex(ysol_all[i](lam[i]))
                Jacobian[i] = complex(dLdT_all[i](lam[i],x,y))
                mu[i] = complex(muovlam[i] * lam[i])
                if np.sign(y)>=0 and np.sign(np.imag(muovlam[i])) == np.sign(np.imag(lam[i])):
                    coeff[i] = False
                elif np.sign(y)<0 and np.sign(np.imag(muovlam[i])) != np.sign(np.imag(lam[i])):
                    coeff[i] = False
                elif np.sign(np.imag(lam[i]))==0 and np.sign(np.imag(muovlam[i]))>0 and abs(y)<1e-18:
                    coeff[i] = False # treat special case at origin / y=0 where lam can become real
            lam = lam[coeff]
            mu = mu[coeff]
            muovlam = muovlam[coeff]
            Jacobian = Jacobian[coeff]
            if len(lam)!=2:
                print(f"Warning: something went wrong at point (r={rx/burgers:.4f}*b, phi={phx:.6f}), skipping.")
                continue ## skip this point (will remain zero)
            ## all four arrays should now be length 2 instead of 4 after applying the mask 'coeff'
            lamJac = lam*Jacobian
            muJac = mu*Jacobian
            etaminlam = 1 / (etapr - lam)
            amdenom = (C2p[0,5] - (C2p[0,1]+C2p[5,5])*muovlam + C2p[1,5]*muovlam**2)
            am = - (C2p[0,0] - 2*C2p[0,5]*muovlam + C2p[5,5]*muovlam**2 - rho/lam**2/norm) /amdenom
            A11denom = C2p[1,1]*(am[0]*mu[0]-am[1]*mu[1]) + C2p[1,5]*(-lam[0]*am[0]+lam[1]*am[1]+mu[0]-mu[1]) \
                       + C2p[0,1]*(lam[1]-lam[0])
            A11 = -(-lam[1]*C2p[0,1] + C2p[1,1]*mu[1]*am[1] + C2p[1,5]*(-lam[1]*am[1]+mu[1])) /A11denom
            A12 = 1 -A11
            A1m = np.array([A11,A12],dtype=complex) ## build the array
            A2m = am*A1m
            ## choose overall sign to match Stroh convention of steady state counter part:
            uij[0,0,ri,ph] = np.sum(np.imag(-np.sign(y)*lamJac*A1m*etaminlam))
            uij[0,1,ri,ph] = np.sum(np.imag(np.sign(y)*muJac*A1m*etaminlam))
            uij[1,0,ri,ph] = np.sum(np.imag(-np.sign(y)*lamJac*A2m*etaminlam))
            uij[1,1,ri,ph] = np.sum(np.imag(np.sign(y)*muJac*A2m*etaminlam))
    ## TODO: implement higher order corrections (tend to zero for 'small' acceleration)
    return (burgers/(2*np.pi))*uij
