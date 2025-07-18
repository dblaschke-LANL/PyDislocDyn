# Compilation of various useful data for metals; all numbers are given in SI units
# Author: Daniel N. Blaschke
# Copyright (c) 2018, Triad National Security, LLC. All rights reserved.
# Date: Nov. 3, 2017 - June 26, 2025
'''This module contains dictionaries of various material properties. Use function 'writeinputfile' to write a PyDislocDyn input file for a specific metal predefined in this module.

References for the data included in these dictionaries (see the manual and its bibliography for further details):
Dictionaries containing effective isotropic elastic constants for polycrystals start with "ISO_" in their names.
Effective isotropic SOEC (in Pa) at room temperature for polycrystals taken from Hertzberg 2012, from Kaye & Laby online (https://web.archive.org/web/20190506031327/http://www.kayelaby.npl.co.uk/),
from the CRC handbook (hbcp.chemnetbase.com), from Smith, Stern, Stephens 1966 (Mo), and from G.V. Samsonov, Handbook of the Physiochemical Properties of the Elements 1968 (Be, Co, Pd, Sc, Zr).
Effective isotropic TOEC (in Pa) at room temperature for polycrystals in Murnaghan notation: Reddy 1976 (Al), Seeger & Buck 1960 (Cu, Fe), and Graham, Nadler, & Chang 1968 (Nb);
in standard notation taken from Smith, Stern, Stephens 1966 (Mg, Mo, W) and Kiewel, Fritsche, Reinert 1996 (Ag, Au, Ni, Co, Zn, Sn).
(We convert all sets to the other notation so that they are available in all dictionaries on isotropic TOEC.)

Various quantities at room temperature are from the CRC handbook (hbcp.chemnetbase.com), 98th--104th ed. ("Crystal Structure Parameters of the Elements" for a, "Thermal and Physical Properties of Metals" for alpha_a, T_m, and rho,
and "Elastic Constants for Single Crystals" for cii and rho_sc (single crystal), except Sc from Singh, Rathore & Agrawal 1992).
Errata: c12['Cu'] is corrected using the original reference, Epstein & Carlson 1965; c44['W'] is corrected using the original reference, Lowrie & Gonas 1967.
The dictionaries containing single crystal SOEC from the CRC handbook start with "CRC_" in their names.
An alternative set of single crystal SOEC (in Pa) at room temperature for selected cubic crystals is given by dictionaries starting with "THLPG_" in their names.
The latter are from Thomas 1968 (Al), Hiki & Granato 1966 (Ag, Au, Cu), Leese & Lord 1968 (Fe), Voronov et al. 1978 (Mo), Graham, Nadler, & Chang 1968 (Nb) and Alers, Neighbours, & Sato 1960 (Ni)

TOEC (in Pa) at room temperature from Thomas 1968 (Al), Saunders & Yogurtcu 1986 (Cd), Yogurtcu, Saunders, Riedi 1985 (Co), Hiki & Granato 1966 (Ag, Au, Cu), Jiles & Palmer 1981 (Er), Powell & Skove 1984 (Fe), Naimon 1971 (Mg),
Voronov et al. 1978 (Mo), Graham, Nadler, & Chang 1968 (Nb), Riley & Skove 1973 (Ni), Swartz, Chua, & Elbaum 1972 (Sn),  Ramji Rao & Menon 1973 (Ti), Swartz & Elbaum 1970 (Zn), and Singh, Rathore & Agrawal 1992 (Sc, Zr),
Srinivasan & Girirajan 1973 (K), Ramji Rao & Ramanan 1980 (Be), Vekilov, Krasilnikov, Lugovskoy, & Lozovik 2016 (W), Suzuki 1971 (Pb), Krasilnikov, Vekilov & Mosyagin 2012 (Ta)
TOEC data for Be, Pb, Sc, Ta, Zr, and W are from ab initio calculations at 0 or low pressure, all others were measured, see resp. references above.
'''
#################################
import numpy as np

ISO_c44 = {'Ag':30.3e9, 'Al':26.1e9, 'Au':27.0e9, 'Be':132e9, 'Cd':19.2e9, 'Co':74.8e9, 'Cr':115.4e9, 'Cu':48.3e9, 'Er':28.3e9, 'Fe':81.6e9, 'Mg':17.3e9, 'Mo':124e9,
           'Nb':37.5e9, 'Ni':76.0e9, 'Pb':5.59e9, 'Pd':43.7e9, 'Pt':61.0e9, 'Sc':29.4e9, 'Sn':18.4e9, 'Ta':69.2e9, 'Ti':43.8e9, 'V':46.7e9, 'W':160.6e9, 'Zn':43.4e9, 'Zr':36.0e9}
ISO_nu = {'Ag':0.367, 'Al':0.345, 'Au':0.440, 'Be':0.032, 'Co':0.310, 'Cd':0.300, 'Cr':0.210, 'Cu':0.343, 'Fe':0.293, 'Mg':0.291, 'Nb':0.397,
          'Ni':0.312, 'Pd':0.390, 'Sc':0.309, 'Sn':0.357, 'Ta':0.342, 'Ti':0.321, 'W':0.280, 'Zn':0.249, 'Zr':0.330}
ISO_bulk = {'Er':44.4e9, 'Mo':178e9+(2/3)*124e9, 'Pb':45.8e9, 'Pt':228.0e9, 'V':158e9}
ISO_c11 = {}
ISO_c12 = {}
for X, Y in ISO_nu.items():
    ISO_c12[X] = round(ISO_c44[X]*2*Y/(1-2*Y),-8)  ## round to same precision that we have for c44 above, i.e. 0.1GPa
    ISO_c11[X] = ISO_c12[X]+2*ISO_c44[X]
for X, Y in ISO_bulk.items():
    ISO_c12[X] = round(Y - 2*ISO_c44[X]/3,-8)
    ISO_c11[X] = ISO_c12[X]+2*ISO_c44[X]
for X in set(ISO_c11.keys()).difference(ISO_bulk.keys()):
    ISO_bulk[X] = round(ISO_c12[X]+2*ISO_c44[X]/3,-8)
for X in set(ISO_c11.keys()).difference(ISO_nu.keys()):
    ISO_nu[X] = round(ISO_c12[X]/(2*(ISO_c12[X]+ISO_c44[X])),3)
ISO_young = {} ## calculate Young's modulus
for X, Y in ISO_nu.items():
    ISO_young[X] = round(2*ISO_c44[X]*(1+Y),-8)
ISO_poisson = ISO_nu

ISO_l = {'Al':-143e9, 'Cu':-160e9, 'Fe':-170e9, 'Nb':-610e9}
ISO_m = {'Al':-297e9, 'Cu':-620e9, 'Fe':-770e9, 'Nb':-220e9}
ISO_n = {'Al':-345e9, 'Cu':-1590e9, 'Fe':-1520e9, 'Nb':300e9}

ISO_c123 = {'Ag':-12e9, 'Au':-411e9, 'Co':-1090e9, 'Mg':-65.4e9, 'Mo':194e9, 'Ni':-185e9, 'Sn':-5e9, 'W':-429e9, 'Zn':-180e9}
ISO_c144 = {'Ag':-160e9, 'Au':-169e9, 'Co':70e9, 'Mg':-57.4e9, 'Mo':-398e9, 'Ni':-466e9, 'Sn':-167e9, 'W':-258e9, 'Zn':-66e9}
ISO_c456 = {'Ag':-86e9, 'Au':-127e9, 'Co':-618e9, 'Mg':-42.1e9, 'Mo':-227e9, 'Ni':0e9, 'Sn':11e9, 'W':-267e9, 'Zn':-89e9}
ISO_c111 = {}
ISO_c112 = {}
ISO_c166 = {}
for X, Y in ISO_c123.items(): ## convert to Murnaghan constants
    ISO_l[X] = Y/2 + ISO_c144[X]
    ISO_m[X] = ISO_c144[X] + 2*ISO_c456[X]
    ISO_n[X] = 4*ISO_c456[X]
# derive cijk (resp. Toupin & Bernstein constants) from Murnaghan constants
for X, Y in ISO_l.items():
    ISO_c123[X] = 2*Y - 2*ISO_m[X] + ISO_n[X] ## =nu1
    ISO_c144[X] = ISO_m[X] - ISO_n[X]/2              ## =nu2
    ISO_c456[X] = ISO_n[X]/4                         ## =nu3
    ISO_c112[X] = ISO_c123[X] + 2*ISO_c144[X]
    ISO_c166[X] = ISO_c144[X] + 2*ISO_c456[X]
    ISO_c111[X] = ISO_c112[X] + 4*ISO_c166[X]
###

CRC_c11 = {'Ag':123.99e9, 'Al':106.75e9, 'Au':192.44e9, 'Be':292.3e9, 'Cd':114.50e9, 'Co':307.1e9, 'Cr':339.8e9, 'Cu':168.3e9, 'Er':86.34e9, 'In':44.50e9, 'Ni':248.1e9, 'Fe':226e9, 'K':3.7e9, 'Mg':59.50e9,
           'Mo':463.7e9, 'Nb':246.5e9, 'Pb':49.66e9, 'Pd':2.2710e11, 'Pt':346.7e9, 'Sc':0.993e11, 'Sn':75.29e9, 'Ta':260.2e9, 'Ti':162.40e9, 'V':2.287e11, 'W':522.39e9, 'Zn':163.68e9, 'Zr':143.4e9}
CRC_c12 = {'Ag':93.67e9, 'Al':60.41e9, 'Au':162.98e9, 'Be':26.7e9, 'Cd':39.50e9, 'Co':165.0e9, 'Cr':58.6e9, 'Cu':121.2e9, 'Er':30.50e9, 'In':39.50e9, 'Ni':154.9e9, 'Fe':140e9, 'K':3.14e9, 'Mg':26.12e9,
           'Mo':157.8e9, 'Nb':134.5e9, 'Pb':42.31e9, 'Pd':1.7604e11, 'Pt':250.7e9, 'Sc':0.457e11, 'Sn':61.56e9, 'Ta':154.4e9, 'Ti':92.00e9, 'V':1.190e11, 'W':204.37e9, 'Zn':36.40e9, 'Zr':72.8e9}
CRC_c44 = {'Ag':46.12e9, 'Al':28.34e9, 'Au':42.0e9, 'Be':162.5e9, 'Cd':19.85e9, 'Co':75.5e9, 'Cr':99.0e9, 'Cu':75.7e9, 'Er':28.09e9, 'In':6.55e9, 'Ni':124.2e9, 'Fe':116e9, 'K':1.88e9, 'Mg':16.35e9,
           'Mo':109.2e9, 'Nb':28.73e9, 'Pb':14.98e9, 'Pd':0.7173e11, 'Pt':76.5e9, 'Sc':0.277e11, 'Sn':21.93e9, 'Ta':82.55e9, 'Ti':46.70e9, 'V':0.432e11, 'W':160.58e9, 'Zn':38.79e9, 'Zr':32.0e9}
CRC_c13 = {'Be':14.0e9, 'Cd':39.90e9, 'Co':102.7e9, 'Er':22.70e9, 'In':40.50e9, 'Mg':21.805e9, 'Sc':0.294e11, 'Sn':44.00e9, 'Ti':69.00e9, 'Zn':53.00e9, 'Zr':65.3e9}
CRC_c33 = {'Be':336.4e9, 'Cd':50.85e9, 'Co':358.1e9, 'Er':85.54e9, 'In':44.40e9, 'Mg':61.55e9, 'Sc':1.069e11, 'Sn':95.52e9, 'Ti':180.70e9, 'Zn':63.47e9, 'Zr':164.8e9}
CRC_c66 = {'In':12.20e9, 'Sn':23.36e9}

CRC_ZenerA = {} ## Zener anisotropy for cubic metals, determined from their SOECs

CRC_a = {'Ag':4.0857e-10, 'Al':4.0496e-10, 'Au':4.0782e-10, 'Be':2.2859e-10, 'Cd':2.9793e-10, 'Co':2.5071e-10, 'Cr':2.8848e-10, 'Cu':3.6146e-10, 'Er':3.5592e-10, 'In':3.253e-10, 'Ni':3.5240e-10, 'Fe':2.8665e-10, 'K':5.321e-10, 'Mg':3.2094e-10,
         'Mo':3.1470e-10, 'Nb':3.3004e-10, 'Pb':4.9502e-10, 'Pd':3.8903e-10, 'Pt':3.9242e-10, 'Sc':3.3088e-10, 'Sn':5.8318e-10, 'Ta':3.3030e-10, 'Ti':2.9506e-10, 'V':3.0240e-10, 'W':3.1652e-10, 'Zn':2.665e-10, 'Zr':3.2316e-10} # lattice constant in m
CRC_c = {'Be':3.5845e-10, 'Cd':5.6196e-10, 'Co':4.0686e-10, 'Er':5.5850e-10, 'In':4.9470e-10, 'Mg':5.2107e-10, 'Sc':5.2680e-10, 'Sn':3.1818e-10, 'Ti':4.6835e-10, 'Zn':4.947e-10, 'Zr':5.1475e-10}
CRC_alpha_a = {'Ag':18.9e-6, 'Al':23.1e-6, 'Au':14.2e-6, 'Be':11.3e-6, 'Cd':30.8e-6, 'Co':13.0e-6, 'Cr':4.9e-6, 'Cu':16.5e-6, 'Er':12.2e-6, 'In':32.1e-6, 'Ni':13.4e-6, 'Fe':11.8e-6, 'K':83.3e-6, 'Mg':24.8e-6,
               'Mo':4.8e-6, 'Nb':7.3e-6, 'Pb':28.9e-6, 'Pd':11.8e-6, 'Pt':8.8e-6, 'Sc':10.2e-06, 'Sn':22.0e-6, 'Ta':6.3e-6, 'Ti':8.6e-6, 'V':8.4e-6, 'W':4.5e-6, 'Zn':30.2e-6, 'Zr':5.7e-6} # coefficient of linear thermal expansion in [K^-1]
CRC_rho_sc = {'Ag':10500, 'Al':26970, 'Au':19283, 'Be':1850, 'Cd':8690, 'Co':8836, 'Cr':7200, 'Cu':8932, 'Er':9064, 'In':7300, 'Ni':8910, 'Fe':7867.2, 'K':851, 'Mg':1740,
              'Mo':10228.4, 'Nb':8578, 'Pb':11340, 'Pd':12038, 'Pt':21500, 'Sc':2990, 'Sn':7290, 'Ta':16626, 'Ti':4506, 'V':6022, 'W':19257, 'Zn':7134, 'Zr':6520} # density in kg/m^3
CRC_rho = {'Ag':10500, 'Al':2700, 'Au':19300, 'Be':1850, 'Cd':8690, 'Co':8860, 'Cr':7150, 'Cu':8960, 'Er':9070, 'In':7310, 'Ni':8900, 'Fe':7870, 'K':890, 'Mg':1740,
           'Mo':10200, 'Nb':8570, 'Pb':11300, 'Pd':12000, 'Pt':21500, 'Sc':2990, 'Sn':7287, 'Ta':16400, 'Ti':4506, 'V':6000, 'W':19300, 'Zn':7134, 'Zr':6520} # density in kg/m^3
CRC_T_m = {'Ag':1234.93, 'Al':933.47, 'Au':1337.33, 'Be':1560.15, 'Cd':594.22, 'Co':1768.15, 'Cr':2180.15, 'Cu':1357.77, 'Er':1802.15, 'Fe':1811.15, 'In':429.75, 'K':336.65, 'Mg':923.15,
           'Mo':2895.15, 'Nb':2750.15, 'Ni':1728.15, 'Pb':600.61, 'Pd':1827.95, 'Pt':2041.35, 'Sc':1814.15, 'Sn':505.08, 'Ta':3290.15, 'Ti':1943.15, 'V':2183.15, 'W':3687.15, 'Zn':692.68, 'Zr':2127.15} ## melting temperature in K at ambient conditions

## sets containing names of all metals of a certain crystal structure (cubic fcc/bcc, hexagonal close packed, tetragonal) at room temperature
fcc_metals = {'Ag', 'Al', 'Au', 'Cu', 'Ni', 'Pb', 'Pd', 'Pt'}
bcc_metals = {'Cr', 'Fe', 'K', 'Mo', 'Nb', 'Ta', 'V', 'W'}
hcp_metals = {'Be', 'Cd', 'Co', 'Er', 'Mg', 'Ti', 'Sc', 'Zn', 'Zr'}
tetr_metals = {'In', 'Sn'}
cubic_metals = fcc_metals | bcc_metals
all_metals = cubic_metals | hcp_metals | tetr_metals

THLPG_c11 = {'Ag':122.2e9, 'Al':106.75e9, 'Au':192.9e9, 'Cu':166.1e9, 'Fe':226e9, 'Mo':461.7e9, 'Nb':246.5e9, 'Ni':250.8e9}
THLPG_c12 = {'Ag':90.7e9, 'Al':60.41e9, 'Au':163.8e9, 'Cu':119.9e9, 'Fe':140e9, 'Mo':164.7e9, 'Nb':133.3e9, 'Ni':150e9}
THLPG_c44 = {'Ag':45.4e9, 'Al':28.34e9, 'Au':41.5e9, 'Cu':75.6e9, 'Fe':116e9, 'Mo':108.7e9, 'Nb':28.4e9, 'Ni':123.5e9}

c111 = {'Ag':-843e9, 'Al':-1076e9, 'Au':-1729e9, 'Be':-2190e9, 'Cd':-2060e9, 'Co':-6710e9, 'Cu':-1271e9, 'Er':-384e9, 'Fe':-2720e9, 'K':-38.5e9, 'Mg':-663e9,
        'Mo':-3557e9, 'Nb':-2564e9, 'Ni':-2040e9, 'Pb':-547e9, 'Pd':-17.54e11, 'Pt':-24.21e11, 'Sc':-7.430e11, 'Sn':-410e9, 'Ta':-2445e9, 'Ti':-1358e9, 'W':-5230e9, 'Zn':-1760e9, 'Zr':-767.4e9}
c112 = {'Ag':-529e9, 'Al':-315e9, 'Au':-922e9, 'Be':-650e9, 'Cd':-114e9, 'Co':-1454e9, 'Cu':-814e9, 'Er':-340e9, 'Fe':-608e9, 'K':-11.1e9, 'Mg':-178e9,
        'Mo':-1333e9, 'Nb':-1140e9, 'Ni':-1030e9, 'Pb':-316e9, 'Pd':-11.22e11, 'Pt':-16.14e11, 'Sc':-3.89e11, 'Sn':-583e9, 'Ta':-703e9, 'Ti':-1105e9, 'W':-932.3e9, 'Zn':-440e9, 'Zr':-697e9}
c113 = {'Be':-530e9, 'Cd':-197e9, 'Co':-766e9, 'Er':-30e9, 'Mg':30e9, 'Sc':-0.320e11, 'Sn':-467e9, 'Ti':17e9, 'Zn':-270e9, 'Zr':-95.9e9}
c123 = {'Ag':189e9, 'Al':36e9, 'Au':-233e9, 'Be':10e9, 'Cd':-110e9, 'Co':-429e9, 'Cu':-50e9, 'Er':711e9, 'Fe':-578e9, 'K':-12.7e9, 'Mg':-76e9,
        'Mo':-617e9, 'Nb':-467e9, 'Ni':-210e9, 'Pb':72e9, 'Pd':-0.35e11, 'Pt':-0.58e11, 'Sc':-0.526e11, 'Sn':128e9, 'Ta':-30.3e9, 'Ti':-162e9, 'W':-742.1e9, 'Zn':-210e9, 'Zr':37.2e9}
c133 = {'Be':-780e9, 'Cd':-268e9, 'Co':-511e9, 'Er':-300e9, 'Mg':-86e9, 'Sc':-0.261e11, 'Sn':-186e9, 'Ti':-383e9, 'Zn':-350e9, 'Zr':-270.6e9}
c144 = {'Ag':56e9, 'Al':-23e9, 'Au':-13e9, 'Be':-110e9, 'Cd':227e9, 'Co':133e9, 'Cu':-3e9, 'Er':-80e9, 'Fe':-836e9, 'K':-13.2e9, 'Mg':-30e9,
        'Mo':-269e9, 'Nb':-343e9, 'Ni':-140e9, 'Pb':41e9, 'Pd':1.05e11, 'Pt':1.73e11, 'Sc':-0.526e11, 'Sn':-162e9, 'Ta':-437.9e9, 'Ti':-263e9, 'W':-786.4e9, 'Zn':-10e9, 'Zr':37.2e9}
c155 = {'Be':-490e9, 'Cd':-332e9, 'Co':-1486e9, 'Er':9.5e9, 'Mg':-58e9, 'Sc':-2.61e11, 'Sn':-177e9, 'Ti':117e9, 'Zn':250e9, 'Zr':-270.6e9}
c166 = {'Ag':-637e9, 'Al':-340e9, 'Au':-648e9, 'Cu':-780e9, 'Fe':-530e9, 'K':-9.4e9, 'Mo':-893e9, 'Nb':-167.7e9, 'Ni':-920e9, 'Pb':-323e9, 'Pd':-9.82e11, 'Pt':-13.83e11, 'Sn':-191e9, 'Ta':-320.6e9, 'W':-924.6e9}
c222 = {'Be':-2440e9, 'Cd':-2020e9, 'Co':-5788e9, 'Er':-95e9, 'Mg':-864e9, 'Sc':-13.69e11, 'Ti':-2306e9, 'Zn':-2410e9, 'Zr':-1450e9}
c333 = {'Be':-2590e9, 'Cd':-516e9, 'Co':-6347e9, 'Er':-150e9, 'Mg':-726e9, 'Sc':-12.26e11, 'Sn':-1427e9, 'Ti':-1617e9, 'Zn':-720e9, 'Zr':-2154e9}
c344 = {'Be':-780e9, 'Cd':-171e9, 'Co':-210e9, 'Er':-220e9, 'Mg':-193e9, 'Sc':-2.41e11, 'Sn':-212e9, 'Ti':-383e9, 'Zn':-440e9, 'Zr':-270.6e9}
c366 = {'Sn':-78e9}
c456 = {'Ag':83e9, 'Al':-30e9, 'Au':-12e9, 'Cu':-95e9, 'Fe':-720e9, 'K':-12.0e9, 'Mo':-555e9, 'Nb':136.6e9, 'Ni':-70e9, 'Pb':61e9, 'Pd':1.75e11, 'Pt':2.88e11, 'Sn':-52e9, 'Ta':-179.1e9, 'W':-934.7e9}

### set to "None" non-independent elastic constants (and compute the Zener anisotropy ratio for cubic metals)
for X in cubic_metals.intersection(CRC_c11.keys()):
    CRC_c13[X] = None
    CRC_c33[X] = None
    CRC_c66[X] = None
    CRC_ZenerA[X] = 2*CRC_c44[X]/(CRC_c11[X] - CRC_c12[X])

for X in cubic_metals.intersection(c111.keys()):
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
    
### compute the unit cell volumes:
CRC_Vc = {}
for X in cubic_metals:
    CRC_Vc[X] = CRC_a[X]**3
for X in hcp_metals:
    CRC_Vc[X] = CRC_a[X]*CRC_a[X]*CRC_c[X]*3*np.sin(np.pi/3)
for X in tetr_metals:
    CRC_Vc[X] = CRC_a[X]*CRC_a[X]*CRC_c[X]

X=Y=None
#####################################################################################
def writeinputfile(X,fname='auto',iso=False,bccslip='110',hcpslip='basal',alt_soec=False,alt_rho=False):
    '''Write selected data of metal X to a text file in a format key = value that can be read and understood by other parts of PyDislocDyn.
       Boolean option 'iso' is used to choose between writing single crystal values (default) and polycrystal (isotropic) averages.
       To choose between various predefined slip systems, use options 'bccslip'='110' (default), '112', or '123' and 'hcpslip'='basal' (default),
       'prismatic', or 'pyramidal'.
       By setting 'alt_soec=True', one may swap out the CRC handbook values of SOECs for some fcc metals with alternative ones (see THLPG dictionaries).
       Likewise, 'alt_rho=True' will use the CRC_rho_sc dictionary instead of the CRC_rho one.'''
    if fname=='auto': fname = X
    with open(fname,"w", encoding="utf8") as outf:
        outf.write(f"# this input file requires PyDislocDyn >=1.2.7\n# input parameters for {X} at ambient conditions\n\n")
        outf.write(f"name = {fname}\n")
        if X in fcc_metals:
            outf.write("sym = fcc\n\n")
            outf.write("# example slip system:\nMillerb = 1/2, 1/2, 0 \t# normalization applied automatically upon reading (after calculating 'burgers')\n")
            outf.write(f"# burgers = {CRC_a[X]/np.sqrt(2)} \t# a/sqrt(2) (determined from Millerb above), this optional line can be used as an override\n")
            outf.write("Millern0 = -1, 1, -1 \t# normalization 1/sqrt(3) applied automatically upon reading\n\n")
        elif X in bcc_metals:
            outf.write("sym = bcc\n\n")
            outf.write("# example slip system:\nMillerb = 1/2, -1/2, 1/2 \t# normalization applied automatically upon reading (after calculating 'burgers')\n")
            outf.write(f"# burgers = {CRC_a[X]*np.sqrt(3)/2} \t# a*sqrt(3)/2 (determined from Millerb above), this optional line can be used as an override\n")
            if bccslip=='112':
                outf.write("Millern0 = 1, -1, -2 \t# slip in 112 plane, normalization 1/sqrt(6) applied automatically upon reading\n\n")
            elif bccslip=='123':
                outf.write("Millern0 = 1, -2, -3 \t# slip in 123 plane, normalization 1/sqrt(14) applied automatically upon reading\n\n")
            else:
                outf.write("Millern0 = 1, 1, 0 \t# slip in 110 plane, normalization 1/sqrt(2) applied automatically upon reading\n\n")
        elif X in hcp_metals:
            outf.write("sym = hcp\n\n")
            outf.write("# example slip systems:\n")
            outf.write("Millerb = -2/3, 1/3, 1/3, 0\n") ## Miller indices are converted to normalized Cartesian upon reading
            outf.write(f"# burgers = {CRC_a[X]} \t# a (determined from Millerb above), this optional line can be used as an override\n")
            if hcpslip=='prismatic':
                outf.write("Millern0 = -1, 0, 1, 0 \t# prismatic slip\n\n")
            elif hcpslip=='pyramidal':
                outf.write("Millern0 = -1, 0, 1, 1 \t# pyramidal slip, normalization applied automatically upon reading\n\n")
            else:
                outf.write("Millern0 = 0, 0, 0, 1 \t# basal slip\n\n")
            ### slip directions for hcp are the [1,1,bar-2,0] directions; the SOEC are invariant under rotations about the z-axis
            ### caveat: TOEC are only invariant under rotations about the z-axis by angles of n*pi/3; measurement was done with x-axis aligned with one of the slip directions
            ### therefore, may choose b parallel to x-axis
        elif X in tetr_metals:
            outf.write("sym = tetr\n\n")
            outf.write("# example slip system:\nMillerb = 0, 0, -1\n")
            outf.write(f"# burgers = {CRC_c[X]} \t# c (determined from Millerb above), this optional line can be used as an override\n")
            outf.write("Millern0 = 0, 1, 0\n\n")
            ## just one of many possible slip systems in tetragonal crystals such as Sn (see Jpn J Appl Phys 32:3214 for a list):
            ## we choose here the simplest one with the shortest burgers vector in Sn (i.e. energetically most favorable),
            ## slip plane normal may be parallel to either x or y as C2,C3 are invariant under rotations by pi/2 about the z axis
        outf.write("# temperature, lattice constant(s), density, thermal expansion coefficient, and melting temperature:\n")
        outf.write(f"T = 300\na = {CRC_a[X]}\n")
        if X in CRC_c:
            outf.write(f"c = {CRC_c[X]}\n")
        if alt_rho:
            outf.write(f"rho = {CRC_rho_sc[X]}\n")
        else:
            outf.write(f"rho = {CRC_rho[X]}\n")
        outf.write(f"alpha_a = {CRC_alpha_a[X]}\n")
        outf.write(f"Tm = {CRC_T_m[X]}\n")
        outf.write("\n#soec\n")
        if iso is True:
            outf.write("\nsym = iso\t# (overwrites previous entry)")
            outf.write(f"\na = {np.cbrt(CRC_Vc[X])}\t# replace by average lattice constants such that a^3 is the true unit cell volume\n\n")
            soec = {"c11":ISO_c11, "c12":ISO_c12, "c44":ISO_c44}
        elif alt_soec and X in THLPG_c44:
            soec = {"c11":THLPG_c11, "c12":THLPG_c12, "c44":THLPG_c44}
        else:
            soec = {"c11":CRC_c11, "c12":CRC_c12, "c44":CRC_c44, "c13":CRC_c13, "c33":CRC_c33, "c66":CRC_c66}
        for c2, soec_c2 in soec.items():
            val = soec_c2[X]
            if val is not None:
                outf.write(f"{c2} = {val:e}\n")
        outf.write("\n#toec\n")
        if iso is True:
            toec = {"c123":ISO_c123, "c144":ISO_c144, "c456":ISO_c456}
        else:
            toec = {"c111":c111, "c112":c112, "c113":c113, "c123":c123, "c133":c133, "c144":c144, "c155":c155, "c166":c166, "c222":c222, "c333":c333, "c344":c344, "c366":c366, "c456":c456}
        for c3, toec_c3 in toec.items():
            if X in toec_c3:
                val = toec_c3[X]
                if val is not None:
                    outf.write(f"{c3} = {val:e}\n")
        if X in ISO_c44 and not iso:
            outf.write("\n## optional - if omitted, averages will be used:\n")
            outf.write(f"lam = {ISO_c12[X]:e}\n")
            outf.write(f"mu = {ISO_c44[X]:e}\n")
        outf.write("\n\n")

def writeallinputfiles(iso=False,alt_soec=False,alt_rho=False):
    '''Calls writeinputfile() for all metals and slip systems defined in the dictionaries of metal_data.py.
       Additional options 'iso', 'alt_soec', and 'alt_rho' are passed on to that function.'''
    if iso:
        for X in all_metals.intersection(ISO_c44):
            writeinputfile(X,iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
    else:
        for X in fcc_metals.union(tetr_metals):
            writeinputfile(X,iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
        for X in bcc_metals:
            writeinputfile(X,X+"110",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
            writeinputfile(X,X+"112",bccslip="112",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
            writeinputfile(X,X+"123",bccslip="123",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
        for X in hcp_metals:
            writeinputfile(X,X+"basal",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
            writeinputfile(X,X+"prismatic",hcpslip="prismatic",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
            writeinputfile(X,X+"pyramidal",hcpslip="pyramidal",iso=iso,alt_soec=alt_soec,alt_rho=alt_rho)
