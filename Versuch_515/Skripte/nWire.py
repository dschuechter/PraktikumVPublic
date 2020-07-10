#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import odr
import mylib

def func_cos_square(p,x):
    a,b,c,d=p
    return a*(np.cos((x-b)*np.pi/180))**2+d**2


plt.figure(dpi=100, figsize=(10,5))
file = "nWire"
data = genfromtxt(file, delimiter=';')
x = (data[3:-2, 3])*8.5
y= data[3:-2, 1]
y_error = np.sqrt(y)
plt.errorbar(x,y,yerr=y_error,fmt='x',zorder=4)
# plt.plot(x, y, '-')
p=[10000.,300.,100,1.]
print(mylib.anpassung_yerr(mylib.func_gauss,x,y,y_error,p,True))
x_new=np.linspace(min(x), max(x), 1000)
y_new=mylib.func_gauss(p,x_new)
# plt.plot(x_new, y_new,zorder=3, color='grey')


plt.xlabel('horizontale koordinate / mm')
plt.ylabel('n')
# plt.title(file)
plt.savefig('../../pics/bestimme_maximum.pdf')
plt.show()

plt.figure(dpi=100, figsize=(10,5))
x = data[3:-2, 3]
x_error = x*0.01
y= data[3:-2, 1]
y_error = np.sqrt(y)
w=np.zeros(len(x))
v=np.zeros(len(x))

loc_szinti = 299.95648657

for i in x:
    i=int(i-3)
    if i % 2 == 0:
        w[i] = np.arctan((i*8.5-loc_szinti)/(125-7.5))*180/np.pi
    else:
        w[i] = np.arctan((i*8.5-loc_szinti)/(125+7.5))*180/np.pi

# for i in [28,29,30,31]:
#     w = np.delete(w,i)
#     y = np.delete(y,i)
# print(w[:28],w[23:])
# w=np.hstack((w[:28],w[32:]))
# y=np.hstack((y[:28],y[32:]))

# v=np.arctan((x*8.5-loc_szinti)/(125))*180/np.pi
# print(np.stack((x,w,v),axis=1))

p=[10000.,0,0.02,0]
p,perr=mylib.anpassung_yerr(func_cos_square,w,y,np.sqrt(y),p,True)
print(p)
# print(mylib.anpassung_yerr(func_cos_square,v,y,y_error,p,True))
# print(mylib.anpassung_no_err(func_cos_square,w,y,p,True))
x_new=np.linspace(-90, 90, 1000)
y_new=func_cos_square(p,x_new)
plt.plot(x_new, y_new,zorder=-3, color='grey',alpha=.5)
plt.errorbar(w,y,yerr=np.sqrt(y), fmt='x-',zorder=4)
# plt.errorbar(v,y,yerr=y_error,fmt='x-',zorder=4,label='plump')
plt.xlabel('Winkel')
plt.ylabel('n')
plt.savefig('../../pics/Winkelverteilung.pdf')
plt.xlim(-90,90)
# plt.legend()
# plt.title(file)
plt.show()
