#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate
import mylib

plt.figure(figsize=(10, 5))
data = genfromtxt('./Datensätze/Vakuum_Erzeugen-Table 1.csv', delimiter=';')
x = data[2:-1, 0]
y= data[2:-1, 1]
# plt.plot(x, y, '-')
plt.errorbar(x,y,yerr=y*0.1,xerr=np.ones(len(x)),linestyle = "None")

x=x[3:]
y=y[3:]
p=[-0.05,150,30]
popt,perr=mylib.anpassung_double_err(mylib.func_exp,x,x*0.1,y,np.ones(len(y)),p,True,False)
print("p(t)=e^{(%.3f\pm%.3f)10^{-2}\cdot(t-(%.1f\pm%.1f))}+(%.2f\pm%.2f)" %(popt[0]*10,perr[0]*10,popt[1],perr[1],popt[2],perr[2]))
# x=np.linspace(0,100,1000)
# plt.plot(x, mylib.func_exp(p,x))

# f = interpolate.interp1d(x, y)
# x = np.arange(x[0], x[-1], 0.1)
# s = f(x)

# f = interpolate.Rbf(x, y, smooth=10)
# s = f(x)
#
# plt.plot(x, s, '-')
plt.xlabel("Zeit / s")
plt.ylabel("p / mbar")
plt.savefig("../pics/zeit-druck-kurve.pdf")
plt.show()
plt.figure(figsize=(10, 5))
x = data[2:-1, 0]
y = data[2:-1, 1]

plt.errorbar(x,np.log(y),yerr=0.1,xerr=np.ones(len(x)),linestyle = "None")
x=x[3:]
y=y[3:]
popt,perr=mylib.anpassung_double_err(mylib.func_lin, x, np.ones(len(x)), np.log(y), 1/y*0.1, [1,1], True, False)
# print(popt)
print("log(p(t))=(%.2f\pm%.2f)10^{-2}\cdot t+(%.2f\pm%.2f)" %(popt[0]*100,perr[0]*100,popt[1],perr[1]))
plt.xlabel("Zeit / s")
plt.ylabel("$\log(p)$ / $\log$(mbar)")
plt.savefig("../pics/zeit-druck-kurve_log.pdf")
plt.show()
