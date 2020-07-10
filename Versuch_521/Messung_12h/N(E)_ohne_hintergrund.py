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
    a, b, c, e = p
    return np.exp(-(x-a)**2/(2*b*b))*c+e

def func(p,x):
    a, b, c = p
    # return b/(x-a)+c
    return np.exp(1)**(a*(x-b))+c

#datensatz auswahl
datensatz='Bodenprobe'

#Imortiere rohe daten N(kanal)
eckdaten = genfromtxt('eckdaten.txt', delimiter=':')
data = genfromtxt(datensatz+'.txt',delimiter=';')
hintergrund_data = genfromtxt(datensatz+'_Hintergrund.txt',delimiter=';')
x0 = data[:, 0]
y0 = data[:, 1]
hintergrund = hintergrund_data[:, 1]
#plt.plot(x0, hintergrund)
#plt.plot(x0, y0)
# hintergrund=75702/72878*hintergrund
# # y0 = y0-hintergrund
# y0[y0 < 0] = 0
# y0_error = np.sqrt(y0)
# x0_error = np.ones(len(x0))*0.5
#
# # rechne Kanal in Energie um
# x0=ecal.E(x0)
# x0_error=ecal.deltaE(x0,x0_error)
# plt.errorbar(x0, y0, yerr=y0_error, xerr=x0_error, zorder=0, label='data', fmt='none')
maxima = np.zeros(shape=(4,int(eckdaten[0,1])))


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
        presets=[ecal.E(eckdaten[3+i*2,1]), width, y1[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
        # presets=[eckdaten[3+i*2,1], width, y1[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
        out = odr.ODR(data, model, beta0=presets).run()

        popt = out.beta
        perr = out.sd_beta
        maxima[0,i]=popt[0]#a
        maxima[1,i]=perr[0]#a_err
        maxima[2,i]=popt[2]#d
        maxima[3,i]=perr[2]#d_err
        # print(perr[2])
        b=ufloat(popt[1],perr[1])
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = gauss(popt, x_fit)

        plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak ('+str(round(popt[0],2))+'+/-'+str(round(perr[0],2))+') keV')
        # plt.plot(x_fit, gauss(presets, x_fit), color='grey', zorder=-3)
    return maxima

hintergrund=75702/72878*hintergrund
y0_error = np.sqrt(y0)
x0_error = np.ones(len(x0))*0.5
y0_uf=unp.uarray(y0,y0_error)
x0_uf=unp.uarray(ecal.E(x0),ecal.deltaE(x0,x0_error))
hintergrund_uf=unp.uarray(hintergrund,np.sqrt(hintergrund))
y0_uf = y0_uf-hintergrund_uf
y0_uf[y0_uf < 0] = 0
# y0_uf=unp.uarray(y0,y0_error)

plt.figure(dpi=400, figsize=(10,5))
ykorr, ykorr_error=korrektur(y0_uf)
# plt.plot(x0,y0)
plt.errorbar(ecal.E(x0),ykorr,xerr=x0_error,yerr=ykorr_error)
pltdata=genfromtxt('data.txt',delimiter=';')
# for i in range(0,int(eckdaten[0,1])):
#     x_fit = np.linspace(pltdata[i,4],pltdata[i,5], 10000)
#     y_fit = gauss([pltdata[i,0],pltdata[i,1],pltdata[i,2],0], x_fit)
#
#     plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak ('+str(round(pltdata[i,0],2))+') keV')
# ymaxima=gausanpassung(x0,ykorr,x0_error,ykorr_error)
# print(ymaxima[2,:])
# plt.legend()
plt.ylabel('rel. I(E)')
plt.xlabel('Energie / keV')
plt.savefig('../../pics/Bodenprobe_ohne_hintergrund.png')
plt.show()
