#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import odr
import matplotlib.pyplot as plt
import numpy as np
import mylib

def func_cos_square(p,x):
    a,b,c,d=p
    return a*(np.cos((x-b)*np.pi/180))**2+np.sqrt(d**2)

plt.figure(dpi=100, figsize=(10,5))


file = "all"
data = genfromtxt(file, delimiter=';')
x = data[:,1]
w = np.arctan(1/-x)*180/np.pi
n = data[:,4]
winkel = np.hstack((w,w[n < 2],w[n < 3],w[n < 4]))
hight, bin_left, bla=plt.hist(winkel, bins=250, zorder=-2, label='all')
bin_center=np.zeros(len(bin_left)-1)
for i in range(0, len(bin_left)-1):
    bin_center[i]=(bin_left[i]+bin_left[i+1])/2
hight[hight==0]=0.1
p=[1000.,0,0.02,0]
p,perr=mylib.anpassung_yerr(func_cos_square,bin_center,hight,np.sqrt(hight),p,True)
print(p)
x_new=np.linspace(-90, 90, 1000)
y_new=func_cos_square(p,x_new)
plt.plot(x_new, y_new,zorder=3, color='grey')


# x=np.linspace(-60,40,1000)
# plt.plot(x, np.cos(0.03*(x-10))**2*1000)


# plt.title("Gaussian Histogram")
plt.xlabel("Winkel")
plt.ylabel("Trefferzahl x 4")
# plt.legend()
plt.savefig('../pics/zusatz.pdf')
plt.show()
