#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate
from scipy import odr

plt.figure(dpi=400)
def lin_func(p, x):
    a, b = p
    return a*x+b


file = sys.argv[1]
data = genfromtxt(file, delimiter=';')
file = file[:-4]
x = data[:, 2]
y = data[:, 6]
x_error = data[:,3]
y_error = data[:,7]
#plt.plot(x, y, '.')

model = odr.Model(lin_func)
data = odr.RealData(x, y, sx=x_error, sy=y_error)
out = odr.ODR(data, model, beta0=[1.,1.]).run()
popt = out.beta
perr = out.sd_beta
x_fit = np.linspace(min(x), max(x), 10000)
y_fit = lin_func(popt, x_fit)
plt.errorbar(x,y,x_error, y_error,zorder=0, fmt='x')
plt.plot(x_fit, y_fit, color='black', zorder=-1, label = "$E(c)= (%.3f\pm%.3f)10^{-2}keV \cdot c + (%.3f\pm%.3f)keV$"%(popt[0]*10**2, perr[0]*10**2, popt[1], perr[1]))

print('E(c)=a*c+b')
print('a='+str(round(popt[0],5))+';'+str(round(perr[0],5)))
print('b='+str(round(popt[1],3))+';'+str(round(perr[1],3)))

# f = interpolate.Rbf(x, y, smooth=1)
# x = np.arange(122, 1408, 1)
# s = f(x)

# plt.plot(x, s, '--', color='grey')

# plt.title(file)
plt.xlabel('Energie / keV')
plt.ylabel('FWHM / keV')
plt.legend()
plt.savefig("../pics/Energie-Diagramm_"+file+".png")
plt.show()
