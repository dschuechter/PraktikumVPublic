#!/usr/bin/env python3

import sys
import argparse
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.odr as sodr
from numpy import genfromtxt
from scipy.optimize import curve_fit
from scipy import stats
from scipy import odr

def lin_func(p, x):
    a, b = p
    return a*x+b

data0 = genfromtxt('Peak_channels/Co_H.txt',delimiter=';')
x0 = data0[:, 1]
y0 = data0[:, 3]
x0_error = data0[:, 2]
plt.errorbar(x0, y0, xerr=x0_error, zorder=0, label='Co', fmt='x')
# plt.plot(x0, y0, 'x', label="Co")

data1 = genfromtxt('Peak_channels/Cs_H.txt',delimiter=';')
x1 = data1[1]
y1 = data1[3]
x1_error = data1[2]
plt.errorbar(x1, y1, xerr=x1_error, zorder=0, label='Cs', fmt='x')
# plt.plot(x1, y1, 'x', label='Cs')

data2 = genfromtxt('Peak_channels/Eu_H.txt',delimiter=';')
x2 = data2[:, 1]
y2 = data2[:, 3]
x2_error = data2[:, 2]
plt.errorbar(x2, y2, xerr=x2_error, zorder=0, label='Eu', fmt='x')
# plt.plot(x2, y2, 'x', label='Eu')

x=np.append(x0,x1)
y=np.append(y0,y1)
x_error=np.append(x0_error,x1_error)
x=np.concatenate([x,x2])
y=np.concatenate([y,y2])
x_error=np.concatenate([x_error,x2_error])
# Create a RealData object
model = odr.Model(lin_func)
data = odr.RealData(x, y, sx=x_error)

# Set up ODR with the model and data.
out = odr.ODR(data, model, beta0=[1.,1.]).run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta
# print('fit parameter 1-sigma error')
# print('———————————–')
# for j in range(len(popt)):
#     print(str(round(popt[j],3))+' +- '+str(round(perr[j],3)))
# print('E(c)=a$\cdot$c+b')
# print('a='+str(round(popt[0],5))+';'+str(round(perr[0],5)))
# print('b='+str(round(popt[1],3))+';'+str(round(perr[1],3)))
x_fit = np.linspace(min(x), max(x), 10000)
y_fit = lin_func(popt, x_fit)
plt.plot(x_fit, y_fit, color='black', zorder=-1)

plt.fill_between(x_fit, lin_func(popt+perr, x_fit), lin_func(popt-perr, x_fit), alpha=.25)

def E(c):
	return popt[0]*c+popt[1]

def deltaE(c,delta_c):
	return np.sqrt((c*perr[0])**2+(perr[1])**2+(popt[0]*delta_c)**2)

plt.ylabel('Energie')
plt.xlabel('Kanal(c)')
plt.legend()
# plt.show()
plt.cla()
plt.clf()
