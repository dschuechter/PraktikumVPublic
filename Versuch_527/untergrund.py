#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import mylib
plt.figure(figsize=(10, 5))
file = sys.argv[1]
data = genfromtxt(file, delimiter=';')
x = data[1:-1, 0]
x_error=np.ones(len(x))*2
y= data[1:-1, 1]/400
y_error=np.sqrt(y*400)/400
plt.errorbar(x,y,yerr=y_error,xerr=x_error,linestyle = "None")
mylib.anpassung_double_err(mylib.func_gauss,x,x_error,y,y_error,[200,100,100,1400],True,False)

y= data[1:-1, 2]/400
y_error=np.sqrt(y*400)/400
plt.errorbar(x,y,yerr=y_error,xerr=x_error,linestyle = "None")
mylib.anpassung_double_err(mylib.func_gauss,x,x_error,y,y_error,[200,270,100,1400],True,False)


# f = interpolate.interp1d(x, y)
# x = np.arange(x[0], x[-1], 0.1)
# s = f(x)

# f = interpolate.Rbf(x, y, smooth=10)
# s = f(x)
#
# plt.plot(x, s, '-')

plt.xlabel("$\phi$ / grad")
plt.ylabel("Zählrate / cps")
plt.savefig("../pics/Untergrund.pdf")
plt.show()
