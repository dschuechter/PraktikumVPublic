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
from uncertainties import ufloat
import uncertainties.unumpy as unp
import energiekalibrierung_H as ecal
from rel_eff_eichung import korrektur
from scipy import interpolate

def gauss(p, x):
    a, b, c, d, e = p
    return np.exp(-(x-a)**2/(2*b*b))*c+d*x+e

def simplegauss(p,x):
    a, b, c = p
    # return b/(x-a)+c
    return np.exp(-(x-a)**2/(2*b*b))*c

#datensatz auswahl
datensatz='Bodenprobe'

#Imortiere rohe daten N(kanal)
eckdaten = genfromtxt('Cs_eckdaten.txt', delimiter=':')
data = genfromtxt(datensatz+'.txt',delimiter=';')
hintergrund_data = genfromtxt(datensatz+'_Hintergrund.txt',delimiter=';')
x0 = data[:, 0]
y0 = data[:, 1]
hintergrund = hintergrund_data[:, 1]
#plt.plot(x0, hintergrund)
#plt.plot(x0, y0)
hintergrund=75702/72878*hintergrund
# y0 = y0-hintergrund
y0[y0 < 0] = 0
y0_error = np.sqrt(y0)
x0_error = np.ones(len(x0))*0.5

# rechne Kanal in Energie um
x0=ecal.E(x0)
x0_error=ecal.deltaE(x0,x0_error)
# plt.errorbar(x0, y0, yerr=y0_error, xerr=x0_error, zorder=0, label='data', fmt='none')


# Plotte an alle i Peaks mit den grenzen aus den eckdaten eine Gaussfunktion(gauss) mit hilfe von ODR
def gausanpassung(x1,y1,x1_error,y1_error):
    model = odr.Model(gauss)
    width=0.7
    height=1.

    for i in range(0,int(eckdaten[0,1])):
        # print(eckdaten[3+i*2,:])
        x = x1[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        y = y1[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        x_error = x1_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        y_error = y1_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]

        # Create a RealData object
        data = odr.RealData(x, y, sx=x_error, sy=y_error)

        # Set up ODR with the model and data.
        presets=[ecal.E(eckdaten[3+i*2,1]), 10, 20, 0. ,25.]
        # presets=[eckdaten[3+i*2,1], width, y1[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
        out = odr.ODR(data, model, beta0=presets).run()

        popt = out.beta
        perr = out.sd_beta
        # maxima[0,i]=popt[0]#a
        # maxima[1,i]=perr[0]#a_err
        # maxima[2,i]=popt[2]#d
        # maxima[3,i]=perr[2]#d_err
        # print(popt,min(x), max(x))
        maxima[i]=np.vstack((popt,perr))
        b=ufloat(popt[1],perr[1])
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = gauss(maxima[i,0], x_fit)
        print(popt[0],perr[0])
        # print(np.hstack(np.stack((popt,perr),axis=1)))
        plt.plot(x_fit, y_fit, label='137-Cs Peak ('+str(round(popt[0],2))+str(round(perr[0],2))+'+/-'+str(round(perr[0],2))+') keV')
        plt.errorbar(x=popt[0], y=popt[2]+popt[3]*popt[0]+popt[4], yerr=perr[2] ,xerr=perr[0], marker='x',linestyle = 'None', color='red',zorder=3)
        # plt.plot(x_fit, gauss(presets, x_fit), color='grey', zorder=-3)
    return maxima

# plt.figure(dpi=400, figsize=(10,5))
y0_uf=unp.uarray(y0,y0_error)
# maxima = np.zeros(shape=(4,int(eckdaten[0,1])))
maxima = np.zeros(shape=(int(eckdaten[0,1]),2,5))
# print(maxima)
ykorr, ykorr_error=korrektur(y0_uf)
# plt.plot(x0,y0)
# plt.plot(x0,ykorr)
plt.errorbar(x0,ykorr,xerr=x0_error,yerr=ykorr_error,fmt='x', alpha=.5)
ymaxima=gausanpassung(x0,ykorr,x0_error,ykorr_error)
# print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
# print(ymaxima)
plt.legend()#prop={'size': 8})
plt.ylabel('rel. I(E)')
plt.xlabel('Energie / keV')
plt.xlim(645,680)
plt.ylim(10,60)
plt.savefig('../../pics/Cs_Bodenprobe.png')
plt.show()

# plt.figure(dpi=400, figsize=(10,5))
# maxima = np.zeros(shape=(4,int(eckdaten[0,1])))
maxima = np.zeros(shape=(int(eckdaten[0,1]),2,5))
hintergrund_uf=unp.uarray(hintergrund,np.sqrt(hintergrund))
hkorr,hkorr_error=korrektur(hintergrund_uf)
plt.errorbar(x0,hkorr,xerr=x0_error,yerr=hkorr_error,fmt='x', alpha=.5)
hmaxima=gausanpassung(x0,hkorr,x0_error,hkorr_error)
# print(hmaxima[2,:])
# print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')

plt.legend()#prop={'size': 8})
plt.ylabel('rel. I(E)')
plt.xlabel('Energie E / keV')
plt.xlim(645,680)
plt.ylim(10,60)
plt.savefig('../../pics/Cs_Hintergrund.png')
plt.show()
plt.close()
