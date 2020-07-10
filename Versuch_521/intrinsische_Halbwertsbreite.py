#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate
from scipy import odr
import uncertainties.unumpy as unp
from uncertainties import ufloat

plt.figure(dpi=400)
def lin_func(p, x):
    a, b = p
    return a*x+b


data = genfromtxt("FWHM_H.txt", delimiter=';')
x = unp.uarray(data[3:, 2], data[3:,3])
y = (unp.uarray(data[3:, 6],data[3:,7]))**2
plt.plot(unp.nominal_values(x), unp.nominal_values(y), '.')

model = odr.Model(lin_func)
data = odr.RealData(unp.nominal_values(x), unp.nominal_values(y), sx=unp.std_devs(x), sy=unp.std_devs(y))
out = odr.ODR(data, model, beta0=[1.,1.]).run()
popt = out.beta
perr = out.sd_beta
x_fit = np.linspace(min(unp.nominal_values(x)), max(unp.nominal_values(x)), 10000)
y_fit = lin_func(popt, x_fit)
print(popt)
plt.errorbar(unp.nominal_values(x), unp.nominal_values(y),xerr=unp.std_devs(x), yerr=unp.std_devs(y),linestyle = 'None')
plt.plot(x_fit, y_fit, color='black', zorder=-1,label = "$(\Delta E(E_\gamma))^2=\Delta E_D(E_\gamma)^2+(\Delta E_e)^2=c\cdot E_\gamma+(\Delta E_e)^2$\n$=(%.3f\pm%.3f)$keV$\cdot10^{-3}\cdot E_\gamma+(%.3f\pm%.3f$)(keV)$^2$"%(popt[0]*10**3, perr[0]*10**3,popt[1],perr[1]))

print("Const.=",unp.uarray(popt[0],perr[0]),"\nConst sqrt=",unp.sqrt(unp.uarray(popt[0],perr[0])),"\n$E_e^2$=", unp.uarray(popt[1],perr[1]), "\n$E_e$=", unp.sqrt(unp.uarray(popt[1],perr[1])))
plt.xlabel('Energie $E_\gamma$ / keV')
plt.ylabel('FWHM$^2$ / (keV)$^2$')
plt.legend()
plt.savefig("../pics/intrinsische_Halbwertsbreite.png")
plt.show()
