#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import odr
import mylib

plt.figure(dpi=100, figsize=(5,5))
file = sys.argv[1]
data = genfromtxt(file, delimiter=';')
x = data[2:, 0]*1000
x_error = x*0.01
y= data[2:, 1]
y_error = data[2:, 2]
plt.errorbar(x,y,xerr=x_error,yerr=y_error,fmt='x')
# plt.plot(x, y, '-')
print(mylib.anpassung_double_err(mylib.func_func,x,x_error,y,y_error,[0.01,1.,0.],True))
# plt.plot(x,mylib.func_func([3,1.,0.],x))


plt.xlabel('Hochspannung / V')
plt.ylabel('Strom / nA')
# plt.title(file)
plt.savefig('../../pics/Strom-Spannung_mitQuelle.pdf')
plt.show()

plt.figure(dpi=100, figsize=(5,5))
y= data[2:, 3]
y_error = data[2:, 4]
plt.errorbar(x,y,xerr=x_error,yerr=y_error,fmt='x')
# plt.plot(x, y, '-')
print(mylib.anpassung_double_err(mylib.func_func,x,x_error,y,y_error,[0.001,1.,0.],True))
# plt.plot(x,mylib.func_func([3,1.,0.],x))


plt.xlabel('Hochspannung / V')
plt.ylabel('Strom / nA')
# plt.title(file)
plt.savefig('../../pics/Strom-Spannung_ohneQuelle.pdf')
plt.show()
