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

def gauss(p, x):
    a, b, c, d, e = p
    return np.exp(-(x-a)**2/(2*b*b))*c+d*x+e

#datensatz auswahl
element=sys.argv[1]
detektortyp=sys.argv[2]
vermesser = 'Dominic/'
if detektortyp == 'H':
	vermesser = 'Nikolas/'
	import energiekalibrierung_H as ecal
else:
	import energiekalibrierung_S as ecal

datensatz=element+'_'+detektortyp

plt.figure(dpi=400, figsize=(10,5))


#Imortiere rohe daten N(kanal)
eckdaten = genfromtxt('./eckdaten/'+datensatz+'.txt', delimiter=':')
data = genfromtxt('./Messung_'+vermesser+datensatz+'.txt',delimiter=';')
hintergrund_data = genfromtxt('./Messung_'+vermesser+'Hintergrund_'+detektortyp+'.txt',delimiter=';')
x0 = data[:, 0]
y0 = data[:, 1]
hintergrund = hintergrund_data[:, 1]
#plt.plot(x0, hintergrund)
#plt.plot(x0, y0)
y0 = y0 - hintergrund
y0_error = np.sqrt(y0)
x0_error = np.ones(len(x0))*5
if detektortyp == 'H':
	y0_error = np.sqrt(y0)
	x0_error = np.ones(len(x0))*0.5

# rechne Kanal in Energie um
x0=ecal.E(x0)
x0_error=ecal.deltaE(x0,x0_error)
plt.errorbar(x0, y0, yerr=y0_error, xerr=x0_error, zorder=0, label='data', fmt='none')

# plt.plot(x0, y0)
model = odr.Model(gauss)

width=3.
height=1.

if detektortyp == 'H':
	width=0.7
	height=1.

# Plotte an alle i Peaks mit den grenzen aus den eckdaten eine Gaussfunktion(gauss) mit hilfe von ODR
for i in range(0,int(eckdaten[0,1])):
    # print('\n~~~~~~~~~'+str(i+1)+'. Peak~~~~~~~~~~~')
    x = x0[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
    y = y0[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
    x_error = x0_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
    y_error = y0_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]

    # Create a RealData object
    data = odr.RealData(x, y, sx=x_error, sy=y_error)

    # Set up ODR with the model and data.
    presets=[ecal.E(eckdaten[3+i*2,1]), width, y0[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
    out = odr.ODR(data, model, beta0=presets).run()

    #print fit parameters and 1-sigma estimates
    popt = out.beta
    perr = out.sd_beta
    # print('fit parameter 1-sigma error')
    # print('———————————–')
    # for j in range(len(popt)):
    #     print(str(round(popt[j],3))+' +- '+str(round(perr[j],3)))
    b=ufloat(popt[1],perr[1])
    FWHM=2*np.sqrt(2*np.log(2))*b
    print(datensatz+';'+str(i+1)+'. Peak;'+str(popt[0])+';'+str(perr[0])+';'+str(b)+';'+str(FWHM))
    x_fit = np.linspace(min(x), max(x), 10000)
    y_fit = gauss(popt, x_fit)

    plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak (E=('+str(round(popt[0],2))+'$\pm$'+str(round(perr[0],2))+')keV, FWHM=('+str(round(popt[1]*2*np.sqrt(2*np.log(2)),2))+'$\pm$'+str(round(perr[1]*2*np.sqrt(2*np.log(2)),2))+')keV)')
    # plt.plot(x_fit, gauss(presets, x_fit), color='grey')


plt.ylabel('N(E)')
plt.xlabel('Energie E / keV')
plt.legend(loc=1)
plt.savefig("../pics/FWHM_"+element+"_"+detektortyp+".png")
plt.show()
