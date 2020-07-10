import sys
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat

#datensatz auswahl
element=sys.argv[1]
if(element != 'Cs'):
	print("Dieses Skript ist nur für Cs relevant")
	sys.exit()
detektortyp=sys.argv[2]
vermesser = 'Dominic'
if detektortyp == 'H':
	vermesser = 'Nikolas'

datensatz=element+'_'+detektortyp

#Importiere Datensatz
eckdaten = genfromtxt('./eckdaten/PoT_'+datensatz+'.txt', delimiter=':')
raumdaten = genfromtxt('./raumdaten/'+datensatz+'_'+vermesser+'.txt', delimiter=':')
data = genfromtxt('./Messung_'+vermesser+'/'+datensatz+'.txt',delimiter=';')
hintergrund_data = genfromtxt('./Messung_'+vermesser+'/'+'Hintergrund_'+detektortyp+'.txt',delimiter=';')
x = data[:, 0]
y = data[:, 1]
hintergrund = hintergrund_data[:, 1]
hintergrund_error = hintergrund*0.05+2
x_error = np.ones(len(x))*5
y_error = np.sqrt(y)
if detektortyp == 'H':
    x_error = np.ones(len(x))*0.5
    y_error = np.sqrt(y)
    hintergrund_error = hintergrund_error+20
x = unp.uarray(x,x_error)
y = unp.uarray(y,y_error)
hintergrund = unp.uarray(hintergrund,hintergrund_error)

y = y - hintergrund
y[y < 0] = 0

xmin = int(eckdaten[1,0])
xmax = int(eckdaten[1,1])


#Finde Peak to total Verhältnis
total_counts = np.sum(y)
y_error = np.array(unp.std_devs(y))
y = unp.nominal_values(y)

def peak_integral(a, b, delta):
	peak_int = np.sum(y[a:b])
	print("peak-int",peak_int)
	peak_int_error = (((y[a]*y[a])+(y[b]*y[b]))*delta**2)**(1/2)
	print("peak-int-error",peak_int_error)
	peak_int = unp.uarray(peak_int, peak_int_error)
	return peak_int


####
# Absolute Peakeffizienz berechnen
####

Cs_activity = 25 * 10**(-6)*3.7*10**10
#Alle angaben in cm
messdauer = unp.uarray(raumdaten[0],0.001)
dist = unp.uarray(raumdaten[1],1)

if(detektortyp == 'H'):
	if vermesser == "Dominic":
		detektor_durchmesser = 7.62
	else:
		detekor_durchmesser = 4.8
if(detektortyp == 'S'):
	detekor_durchmesser = 5.57

events_per_surface = Cs_activity/(4*np.pi*dist**2)

theoretische_zaehlrate = events_per_surface*np.pi*(0.5*detekor_durchmesser)**2

absolute_peakeffizienz = peak_integral(xmin,xmax,eckdaten[1,2])/(theoretische_zaehlrate*messdauer)

print("Messdauer: ",messdauer, "\nDistance: ", dist, "\nQuanten/cm*s:", events_per_surface, "\nTheoretische Zählrate: ", theoretische_zaehlrate,"\nGemessene Zählrate", peak_integral(xmin,xmax,eckdaten[1,2])/messdauer, "\nAbsolute Peakeffizienz: ", absolute_peakeffizienz)
