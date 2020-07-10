#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import odr

def gauss(p, x):
    a, b, c, d, e = p
    return np.exp(-(x-a)**2/(2*b*b))*c+d*x+e

#datensatz auswahl
element=sys.argv[1]
detektortyp=sys.argv[2]
vermesser = 'Dominic/'
if detektortyp == 'H':
	vermesser = 'Nikolas/'
datensatz=element+'_'+detektortyp

#Imortiere alle daten und bereite sie zum plotten vor
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
plt.errorbar(x0, y0, yerr=y0_error, xerr=x0_error, zorder=0, label='data', fmt='none')
# plt.plot(x0, y0)
model = odr.Model(gauss)


width=30.
height=1.

if detektortyp == 'H':
	width=3.
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
	presets=[eckdaten[3+i*2,1], width, y0[int(eckdaten[3+i*2,1])]*height, 0. ,1.]
	out = odr.ODR(data, model, beta0=presets).run()

	#print fit parameters and 1-sigma estimates
	popt = out.beta
	perr = out.sd_beta
	# print('fit parameter 1-sigma error')
	# print('———————————–')
	# for j in range(len(popt)):
	#     print(str(round(popt[j],3))+' +- '+str(round(perr[j],3)))
	print(str(i+1)+'. Peak;'+str(round(popt[0],3))+';'+str(round(perr[0],3)))
	x_fit = np.linspace(min(x), max(x), 10000)
	y_fit = gauss(popt, x_fit)

	#plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak ('+str(round(popt[0],3))+','+str(round(gauss(popt, popt[0]),3))+')')
	plt.plot(x_fit, y_fit, label=str(i+1)+'. Peak ('+str(round(popt[0],3))+'$\pm$'+str(round(perr[0],3))+')')
    #plt.plot(x_fit, gauss(presets, x_fit), label=str(i+1)+'. Peak presets')


plt.ylabel('N(c)')
plt.xlabel('Kanal (c)')
plt.legend()
plt.savefig("../pics/Kanal_Gauss_"+element+"_"+detektortyp+".png")
plt.show()
