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

def gauss(p, x):
    a, b, c, d, e = p
    return np.exp(-(x-a)**2/(2*b*b))*c+d*x+e

def func(p,x):
    a, b, c = p
    # return b/(x-a)+c
    return np.exp(1)**(a*(x-b))+c



#datensatz auswahl
element='Eu'
detektortyp=sys.argv[1]
vermesser = 'Dominic/'
if detektortyp == 'H':
	vermesser = 'Nikolas/'
	import energiekalibrierung_H as ecal
else:
	import energiekalibrierung_S as ecal

datensatz=element+'_'+detektortyp

#Imortiere rohe daten N(kanal)
eckdaten = genfromtxt('./eckdaten/'+datensatz+'_releff.txt', delimiter=':')
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
	x0_error = np.ones(len(x0))*0.5

# rechne Kanal in Energie um
x0=ecal.E(x0)
x0_error=ecal.deltaE(x0,x0_error)
plt.errorbar(x0, y0, yerr=y0_error, xerr=x0_error, zorder=0, label='data', fmt='none')

maxima = np.zeros(shape=(4,int(eckdaten[0,1])))
relint=np.zeros(int(eckdaten[0,1]))

# Plotte an alle i Peaks mit den grenzen aus den eckdaten eine Gaussfunktion(gauss) mit hilfe von ODR
def gausanpassung(x1,y1,x1_error,y1_error):
    model = odr.Model(gauss)
    width=3.
    height=1.
    if detektortyp == 'H':
    	width=0.7
    	height=1.

    for i in range(0,int(eckdaten[0,1])):
        x = x1[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        y = y1[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        x_error = x1_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]
        y_error = y1_error[int(eckdaten[3+i*2,0]):int(eckdaten[3+i*2,2])]

        # Create a RealData object
        data = odr.RealData(x, y, sx=x_error, sy=y_error)

        # Set up ODR with the model and data.
        presets=[ecal.E(eckdaten[3+i*2,1]), width, y1[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
        out = odr.ODR(data, model, beta0=presets).run()

        popt = out.beta
        perr = out.sd_beta
        b=ufloat(popt[1],perr[1])
        maxima[0,i]=popt[0]#a
        maxima[1,i]=perr[0]#a_err
        print(perr[0])
        maxima[2,i]=popt[2]#d
        maxima[3,i]=perr[2]#d_err
        relint[i]=eckdaten[3+i*2,4]
        FWHM=2*np.sqrt(2*np.log(2))*b
        # print(str(i+1)+'. Peak;'+str(round(popt[0],2))+'plusundminus'+str(round(perr[0],2))+';'+str(round(popt[2],2))+'plusundminus'+str(round(perr[2],2)))
        # print(datensatz+';'+str(i+1)+'. Peak;'+str(b)+';'+str(FWHM))
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = gauss(popt, x_fit)

        plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak (E=('+str(round(popt[0],2))+'$\pm$'+str(round(perr[0],2))+')keV)')
        # plt.plot(x_fit, gauss(presets, x_fit), color='grey', zorder=-3)

gausanpassung(x0,y0,x0_error,y0_error)
# plt.show()
plt.close()


plt.figure(dpi=400, figsize=(10,5))
plt.errorbar(maxima[0,:],maxima[2,:]/relint, xerr=maxima[1,:], yerr=maxima[3,:]/relint,  fmt='x')
model = odr.Model(func)
y=maxima[2,:]/relint
y_error=maxima[3,:]/relint
x=maxima[0,:]
x_error=maxima[1,:]
p=[-0.01,500.,1.]
data = odr.RealData(x, y, sx=x_error, sy=y_error)
out = odr.ODR(data, model, beta0=p).run()
popt = out.beta
perr = out.sd_beta
print(unp.uarray(popt,perr))
x_fit = np.linspace(min(x), max(x), 10000)
y_fit = func(popt, x_fit)
plt.plot(x_fit, y_fit, label= "$N(E)=e^{a\cdot (x-b)}+c=e^{(%.3f\pm%.3f)10^{-3}\cdot (E-(%.3f\pm%.3f))}+(%.3f\pm%.3f)$"%(popt[0]*10**3,perr[0]*10**3,popt[1],perr[1],popt[2],perr[2]))
plt.ylabel('N / rel. Intensität')
plt.xlabel('Energie / keV')
plt.legend()
plt.savefig("../pics/Relative_Effizienz_korrektur.png")
plt.show()
plt.close()

plt.figure(dpi=400, figsize=(10,5))

a=ufloat(popt[0],perr[0])
b=ufloat(popt[1],perr[1])
c=ufloat(popt[2],perr[2])

y_korrektur=func([a,b,c],x0)
ykorr_ufloat= y0/y_korrektur
ykorr_ufloat=1000./949.66*ykorr_ufloat
ykorr=unp.nominal_values(ykorr_ufloat)
ykorr_error=unp.std_devs(ykorr_ufloat)
# plt.plot(x0,y0)
plt.errorbar(x0,ykorr, xerr=x0_error, yerr=ykorr_error, fmt='none')

gausanpassung(x0,ykorr,x0_error,ykorr_error)

plt.ylabel('Intensität I(E)')
plt.xlabel('Energie E / keV')
plt.legend(prop={'size': 6})
plt.savefig("../pics/Relative_Effizienz_"+element+"_"+detektortyp+".png")
plt.show()
