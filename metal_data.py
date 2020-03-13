# Compilation of various useful data for metals; all numbers are given in SI units
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - Mar. 12, 2020
#################################
from __future__ import division
from __future__ import print_function
import numpy as np

## effective isotropic SOEC (in Pa) at room temperature for polycrystals taken from Hertzberg 2012 and from Kaye & Laby online (kayelaby.npl.co.uk)
ISO_c44 = {'Ag':30.3e9, 'Al':26.1e9, 'Au':27.0e9, 'Cd':19.2e9, 'Cr':115.4e9, 'Cu':48.3e9,  'Fe':81.6e9, 'Mg':17.3e9, 'Nb':37.5e9, 'Ni':76.0e9, 'Sn':18.4e9, 'Ta':69.2e9, 'Ti':43.8e9, 'W':160.6e9, 'Zn':43.4e9}
ISO_nu = {'Ag':0.367, 'Al':0.345, 'Au':0.440, 'Cd':0.300, 'Cr':0.210, 'Cu':0.343, 'Fe':0.293, 'Mg':0.291, 'Nb':0.397, 'Ni':0.312, 'Sn':0.357, 'Ta':0.342, 'Ti':0.321, 'W':0.280, 'Zn':0.249}
ISO_bulk = {}
ISO_c11 = {}
ISO_c12 = {}
for X in ISO_nu.keys():
    ISO_c12[X] = round(ISO_c44[X]*2*ISO_nu[X]/(1-2*ISO_nu[X]),-8)  ## round to same precision that we have for c44 above, i.e. 0.1GPa
    ISO_c11[X] = ISO_c12[X]+2*ISO_c44[X]
for X in ISO_bulk.keys():
    ISO_c12[X] = round(ISO_bulk[X] - 2*ISO_c44[X]/3,-8)
    ISO_c11[X] = ISO_c12[X]+2*ISO_c44[X]
for X in set(ISO_c11.keys()).difference(ISO_bulk.keys()):
    ISO_bulk[X] = round(ISO_c12[X]+2*ISO_c44[X]/3,-8)
for X in set(ISO_c11.keys()).difference(ISO_nu.keys()):
    ISO_nu[X] = round(ISO_c12[X]/(2*(ISO_c12[X]+ISO_c44[X])),3)
ISO_young = {} ## calculate Young's modulus
for X in ISO_nu.keys():
    ISO_young[X] = round(2*ISO_c44[X]*(1+ISO_nu[X]),-8)
    
## effective isotropic TOEC (in Pa) at room temperature for polycrystals Reddy 1976 (Al), Seeger & Buck 1960 (Cu, Fe), and Graham, Nadler, & Chang 1968 (Nb)
ISO_l = {'Al':-143e9, 'Cu':-160e9,  'Fe':-170e9, 'Nb':-610e9}
ISO_m = {'Al':-297e9, 'Cu':-620e9,  'Fe':-770e9, 'Nb':-220e9}
ISO_n = {'Al':-345e9, 'Cu':-1590e9,  'Fe':-1520e9, 'Nb':-300e9}
# derive cijk from these Murnaghan constants
ISO_c123 = {}
ISO_c144 = {}
ISO_c456 = {}
ISO_c111 = {}
ISO_c112 = {}
ISO_c166 = {}
for X in ISO_l.keys():
    ISO_c123[X] = 2*ISO_l[X] - 2*ISO_m[X] + ISO_n[X]
    ISO_c144[X] = ISO_m[X] - ISO_n[X]/2
    ISO_c456[X] = ISO_n[X]/4
    ISO_c112[X] = ISO_c123[X] + 2*ISO_c144[X]
    ISO_c166[X] = ISO_c144[X] + 2*ISO_c456[X]
    ISO_c111[X] = ISO_c112[X] + 4*ISO_c166[X]
###

## numbers at room temperature from the CRC handbook (hbcponline.com), 98th ed. ("Crystal Structure Parameters of the Elements" for a, "Thermal and Physical Properties of Metals" for alpha_a and rho, and "Elastic Constants for Single Crystals" for cii and rho_sc (single crystal))
CRC_c11 = {'Ag':123.99e9, 'Al':106.75e9, 'Au':192.44e9, 'Be':292.3e9, 'Cd':114.50e9, 'Cr':339.8e9, 'Cu':168.3e9, 'In':44.50e9, 'Ni':248.1e9, 'Fe':226e9, 'K':3.7e9, 'Mg':59.50e9, 'Mo':463.7e9, 'Nb':246.5e9, 'Sn':75.29e9, 'Ta':260.2e9, 'Ti':162.40e9, 'W':522.39e9, 'Zn':163.68e9, 'Zr':143.4e9}
CRC_c12 = {'Ag':93.67e9, 'Al':60.41e9, 'Au':162.98e9, 'Be':26.7e9, 'Cd':39.50e9, 'Cr':58.6e9, 'Cu':121.2e9, 'In':39.50e9, 'Ni':154.9e9, 'Fe':140e9, 'K':3.14e9, 'Mg':26.12e9, 'Mo':157.8e9, 'Nb':134.5e9, 'Sn':61.56e9, 'Ta':154.4e9, 'Ti':92.00e9, 'W':204.37e9, 'Zn':36.40e9, 'Zr':72.8e9}
CRC_c44 = {'Ag':46.12e9, 'Al':28.34e9, 'Au':42.0e9, 'Be':162.5e9, 'Cd':19.85e9, 'Cr':99.0e9, 'Cu':75.7e9, 'In':6.55e9,  'Ni':124.2e9, 'Fe':116e9, 'K':1.88e9, 'Mg':16.35e9, 'Mo':109.2e9, 'Nb':28.73e9, 'Sn':21.93e9, 'Ta':82.55e9, 'Ti':46.70e9, 'W':160.58e9, 'Zn':38.79e9, 'Zr':32.0e9}
CRC_c13 = {'Be':14.0e9, 'Cd':39.90e9, 'In':40.50e9, 'Mg':21.805e9, 'Sn':44.00e9, 'Ti':69.00e9, 'Zn':53.00e9, 'Zr':65.3e9}
CRC_c33 = {'Be':336.4e9, 'Cd':50.85e9, 'In':44.40e9, 'Mg':61.55e9, 'Sn':95.52e9, 'Ti':180.70e9, 'Zn':63.47e9, 'Zr':164.8e9}
CRC_c66 = {'In':12.20e9, 'Sn':23.36e9}
## errata: c12['Cu'] is corrected using the original reference, Epstein & Carlson 1965; c44['W'] is corrected using the original reference, Lowrie & Gonas 1967
CRC_ZenerA = {} ## Zener anisotropy for cubic metals, determined from their SOECs

CRC_a = {'Ag':4.0857e-10, 'Al':4.0496e-10, 'Au':4.0782e-10, 'Be':2.2859e-10, 'Cd':2.9793e-10, 'Cr':2.8848e-10, 'Cu':3.6146e-10, 'In':3.253e-10, 'Ni':3.5240e-10, 'Fe':2.8665e-10, 'K':5.321e-10, 'Mg':3.2094e-10, 'Mo':3.1470e-10, 'Nb':3.3004e-10, 'Sn':5.8318e-10, 'Ta':3.3030e-10, 'Ti':2.9506e-10, 'W':3.1652e-10, 'Zn':2.665e-10, 'Zr':3.2316e-10} # lattice constant in m
CRC_c = {'Be':3.5845e-10, 'Cd':5.6196e-10, 'In':4.9470e-10, 'Mg':5.2107e-10, 'Sn':3.1818e-10, 'Ti':4.6835e-10, 'Zn':4.947e-10, 'Zr':5.1475e-10}
CRC_alpha_a = {'Ag':18.9e-6, 'Al':23.1e-6, 'Au':14.2e-6, 'Be':11.3e-6, 'Cd':30.8e-6, 'Cr':4.9e-6, 'Cu':16.5e-6, 'In':32.1e-6, 'Ni':13.4e-6, 'Fe':11.8e-6, 'K':83.3e-6, 'Mg':24.8e-6, 'Mo':4.8e-6, 'Nb':7.3e-6, 'Sn':22.0e-6, 'Ta':6.3e-6, 'Ti':8.6e-6, 'W':4.5e-6, 'Zn':30.2e-6, 'Zr':5.7e-6} # coefficient of linear thermal expansion in [K^-1]
CRC_rho_sc = {'Ag':10500, 'Al':26970, 'Au':19283, 'Be':1850, 'Cd':8690, 'Cr':7200, 'Cu':8932, 'In':7300, 'Ni':8910, 'Fe':7867.2, 'K':851, 'Mg':1740, 'Mo':10228.4, 'Nb':8578, 'Sn':7290, 'Ta':16626, 'Ti':4506, 'W':19257, 'Zn':7134, 'Zr':6520} # density in kg/m^3
CRC_rho = {'Ag':10500, 'Al':2700, 'Au':19300, 'Be':1850, 'Cd':8690, 'Cr':7150, 'Cu':8960, 'In':7310, 'Ni':8900, 'Fe':7870, 'K':890, 'Mg':1740, 'Mo':10200, 'Nb':8570, 'Sn':7287, 'Ta':16400, 'Ti':4506, 'W':19300, 'Zn':7134, 'Zr':6520} # density in kg/m^3

## sets containing names of all metals of a certain crystal structure (cubic fcc/bcc, hexagonal close packed, tetragonal) at room temperature
fcc_metals = {'Ag', 'Al', 'Au', 'Cu', 'Ni'}
bcc_metals = {'Cr', 'Fe', 'K', 'Mo', 'Nb', 'Ta', 'W'}
hcp_metals = {'Be', 'Cd', 'Mg', 'Ti', 'Zn', 'Zr'}
tetr_metals = {'In', 'Sn'}

## SOEC (in Pa) at room temperature from Thomas 1968 (Al), Hiki & Granato 1966 (Ag, Au, Cu), Leese & Lord 1968 (Fe), Voronov et al. 1978 (Mo), Graham, Nadler, & Chang 1968 (Nb) and Alers, Neighbours, & Sato 1960 (Ni)
THLPG_c11 = {'Ag':122.2e9, 'Al':106.75e9, 'Au':192.9e9, 'Cu':166.1e9, 'Fe':226e9, 'Mo':461.7e9, 'Nb':246.5e9, 'Ni':250.8e9}
THLPG_c12 = {'Ag':90.7e9, 'Al':60.41e9, 'Au':163.8e9, 'Cu':119.9e9, 'Fe':140e9, 'Mo':164.7e9, 'Nb':133.3e9, 'Ni':150e9}
THLPG_c44 = {'Ag':45.4e9, 'Al':28.34e9, 'Au':41.5e9, 'Cu':75.6e9,  'Fe':116e9, 'Mo':108.7e9, 'Nb':28.4e9, 'Ni':123.5e9}
## TOEC (in Pa) at room temperature from Thomas 1968 (Al), Saunders & Yogurtcu 1986 (Cd), Hiki & Granato 1966 (Ag, Au, Cu), Powell & Skove 1984 (Fe), Naimon 1971 (Mg), Voronov et al. 1978 (Mo), Graham, Nadler, & Chang 1968 (Nb), Riley & Skove 1973 (Ni),
## Swartz, Chua, & Elbaum 1972 (Sn),  Ramji Rao & Menon 1972 (Ti), Swartz & Elbaum 1970 (Zn), and Singh, Rathore & Agrawal 1992 (Zr)
c111 = {'Ag':-843e9, 'Al':-1076e9, 'Au':-1729e9, 'Cd':-2060e9, 'Cu':-1271e9, 'Fe':-2720e9, 'Mg':-663e9, 'Mo':-3557e9, 'Nb':-2564e9, 'Ni':-2040e9, 'Sn':-410e9, 'Ti':-1358e9, 'Zn':-1760e9, 'Zr':-767.4e9}
c112 = {'Ag':-529e9, 'Al':-315e9, 'Au':-922e9, 'Cd':-114e9, 'Cu':-814e9, 'Fe':-608e9, 'Mg':-178e9, 'Mo':-1333e9, 'Nb':-1140e9, 'Ni':-1030e9, 'Sn':-583e9, 'Ti':-1105e9, 'Zn':-440e9, 'Zr':-697e9}
c113 = {'Cd':-197e9, 'Mg':30e9, 'Sn':-467e9, 'Ti':17e9, 'Zn':-270e9, 'Zr':-95.9e9}
c123 = {'Ag':189e9, 'Al':36e9, 'Au':-233e9, 'Cd':-110e9, 'Cu':-50e9, 'Fe':-578e9, 'Mg':-76e9, 'Mo':-617e9, 'Nb':-467e9, 'Ni':-210e9, 'Sn':128e9, 'Ti':-162e9, 'Zn':-210e9, 'Zr':37.2e9}
c133 = {'Cd':-268e9, 'Mg':-86e9, 'Sn':-186e9, 'Ti':-383e9, 'Zn':-350e9, 'Zr':-270.6e9}
c144 = {'Ag':56e9, 'Al':-23e9, 'Au':-13e9, 'Cd':227e9, 'Cu':-3e9, 'Fe':-836e9, 'Mg':-30e9, 'Mo':-269e9, 'Nb':-343e9, 'Ni':-140e9, 'Sn':-162e9, 'Ti':-263e9, 'Zn':-10e9, 'Zr':37.2e9}
c155 = {'Cd':-332e9, 'Mg':-58e9, 'Sn':-177e9, 'Ti':117e9, 'Zn':250e9, 'Zr':-270.6e9}
c166 = {'Ag':-637e9, 'Al':-340e9, 'Au':-648e9, 'Cu':-780e9, 'Fe':-530e9, 'Mo':-893e9, 'Nb':-167.7e9, 'Ni':-920e9, 'Sn':-191e9}
c222 = {'Cd':-2020e9, 'Mg':-864e9, 'Ti':-2306e9, 'Zn':-2410e9, 'Zr':-1450e9}
c333 = {'Cd':-516e9, 'Mg':-726e9, 'Sn':-1427e9, 'Ti':-1617e9, 'Zn':-720e9, 'Zr':-2154e9}
c344 = {'Cd':-171e9, 'Mg':-193e9, 'Sn':-212e9, 'Ti':-383e9, 'Zn':-440e9, 'Zr':-270.6e9}
c366 = {'Sn':-78e9}
c456 = {'Ag':83e9, 'Al':-30e9, 'Au':-12e9, 'Cu':-95e9, 'Fe':-720e9, 'Mo':-555e9, 'Nb':136.6e9, 'Ni':-70e9, 'Sn':-52e9}

### set to "None" non-independent elastic constants (and compute the Zener anisotropy ratio for cubic metals)
for X in fcc_metals.union(bcc_metals).intersection(CRC_c11.keys()):
    CRC_c13[X] = None
    CRC_c33[X] = None
    CRC_c66[X] = None
    CRC_ZenerA[X] = 2*CRC_c44[X]/(CRC_c11[X] - CRC_c12[X])
    
for X in fcc_metals.union(bcc_metals).intersection(c111.keys()):
    c113[X] = None
    c133[X] = None
    c155[X] = None
    c222[X] = None
    c333[X] = None
    c344[X] = None
    c366[X] = None
    
for X in hcp_metals.intersection(CRC_c11.keys()):
    CRC_c66[X] = None

for X in hcp_metals.intersection(c111.keys()):
    c166[X] = None
    c366[X] = None
    c456[X] = None    
    
for X in tetr_metals.intersection(c111.keys()):
    c222[X] = None


#####################################################################################
def writeinputfile(X,fname,iso=False,bccslip='110',hcpslip='basal'):
    '''Write selected data of metal X to a text file in a format key = value that can be read and understood by other parts of PyDislocDyn.
       Boolean option 'iso' is used to choose between writing single crystal values (default) and polycrystal (isotropic) averages.
       To choose between various predefined slip systems, use options 'bccslip'='110' (default), '112', or '123' and 'hcpslip'='basal' (default),
       'prismatic', or 'pyramidal'.'''
    with open(fname,"w") as outf:
        outf.write("# input parameters for {} at ambient conditions\n\n".format(X))
        outf.write("name = {}\n".format(fname))
        if X in fcc_metals:
            outf.write("sym = fcc\n\n")
            outf.write("# example slip system:\nb = "+", ".join(map("{}".format,(np.array([1,1,0])/np.sqrt(2))))+"\n")
            outf.write("burgers = {} \t# a/sqrt(2)\n".format(CRC_a[X]/np.sqrt(2)))
            outf.write("n0 = "+", ".join(map("{}".format,(np.array([-1,1,-1])/np.sqrt(3))))+"\n\n")
        elif X in bcc_metals:
            outf.write("sym = bcc\n\n")
            outf.write("# example slip system:\nb = "+", ".join(map("{}".format,(np.array([1,-1,1])/np.sqrt(3))))+"\n")
            outf.write("burgers = {} \t# a*sqrt(3)/2\n".format(CRC_a[X]*np.sqrt(3)/2))
            if bccslip=='112':
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([1,-1,-2])/np.sqrt(6))))+"\t# slip in 112 plane\n\n")
            elif bccslip=='123':
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([1,-2,-3])/np.sqrt(14))))+"\t# slip in 123 plane\n\n")
            else:
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([1,1,0])/np.sqrt(2))))+"\t# slip in 110 plane\n\n")
        elif X in hcp_metals:
            outf.write("sym = hcp\n\n")
            outf.write("# example slip systems:\n")
            outf.write("b = "+", ".join(map("{}".format,(np.array([-1,0,0]))))+"\n")
            outf.write("burgers = {} \t# a\n".format(CRC_a[X]))
            if hcpslip=='prismatic':
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([0,-1,0]))))+"\t# prismatic slip\n\n")
            elif hcpslip=='pyramidal':
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([0,-CRC_a[X],CRC_c[X]])/np.sqrt(CRC_a[X]**2+CRC_c[X]**2))))+"\t# pyramidal slip\n\n")
            else:
                outf.write("n0 = "+", ".join(map("{}".format,(np.array([0,0,1]))))+"\t# basal slip\n\n")
        elif X in tetr_metals:
            outf.write("sym = tetr\n\n")
            outf.write("# example slip system:\nb = "+", ".join(map("{}".format,(np.array([0,0,-1]))))+"\n")
            outf.write("burgers = {} \t# c\n".format(CRC_c[X]))
            outf.write("n0 = "+", ".join(map("{}".format,(np.array([0,1,0]))))+"\n\n")
        outf.write("# temperature, latticeconstant(s), density, and thermal expansion coefficient:\nT = 300\na = {}\n".format(CRC_a[X]))
        if X in CRC_c.keys():
            outf.write("c = {}\n".format(CRC_c[X]))
        outf.write("rho = {}\n".format(CRC_rho[X]))
        outf.write("alpha_a = {}\n".format(CRC_alpha_a[X]))
        outf.write("\n#soec\n")
        if iso==True:
            outf.write("\nsym = iso\t# (overwrites previous entry)\n\n")
            soec = {"c11":ISO_c11, "c12":ISO_c12, "c44":ISO_c44}
        else:
            soec = {"c11":CRC_c11, "c12":CRC_c12, "c44":CRC_c44, "c13":CRC_c13, "c33":CRC_c33, "c66":CRC_c66}
        for c2 in soec.keys():
            val = soec[c2][X]
            if val != None:
                outf.write("{} = {:e}\n".format(c2,val))
        outf.write("\n#toec\n")
        if iso==True:
            toec = {"c123":ISO_c123, "c144":ISO_c144, "c456":ISO_c456}
        else:
            toec = {"c111":c111, "c112":c112, "c113":c113, "c123":c123, "c133":c133, "c144":c144, "c155":c155, "c166":c166, "c222":c222, "c333":c333, "c344":c344, "c366":c366, "c456":c456}
        for c3 in toec.keys():
            if X in toec[c3].keys():
                val = toec[c3][X]
                if val != None:
                    outf.write("{} = {:e}\n".format(c3,val))
        if X in ISO_c44.keys() and iso==False:
            outf.write("\n## optional - if omitted, averages will be used:\n")
            outf.write("lam = {:e}\n".format(ISO_c12[X]))
            outf.write("mu = {:e}\n".format(ISO_c44[X]))
            
