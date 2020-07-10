import sys
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat
plt.figure(dpi=400)
#datensatz auswahl
element=sys.argv[1]
detektortyp=sys.argv[2]
vermesser = 'Dominic/'
if detektortyp == 'H':
	vermesser = 'Nikolas/'

datensatz=element+'_'+detektortyp

#Importiere Datensatz
eckdaten = genfromtxt('./eckdaten/PoT_'+datensatz+'.txt', delimiter=':')
data = genfromtxt('./Messung_'+vermesser+datensatz+'.txt',delimiter=';')
hintergrund_data = genfromtxt('./Messung_'+vermesser+'Hintergrund_'+detektortyp+'.txt',delimiter=';')
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

xmin = int(eckdaten[1,0])
xmax = int(eckdaten[1,1])

#Finde Peak to total Verhältnis
total_counts = np.sum(y)

def peak_to_total(a, b):
    peak_int = np.sum(y[a:b])
    peak_to_total = peak_int/total_counts
    return peak_to_total

if datensatz == 'Co_H':
    def peak_to_total(a, b):
        peak_int = np.sum(y[a:6195])+np.sum(y[6990:b])
        peak_to_total = peak_int/total_counts
        return peak_to_total

print("Peak-to-Total: ",peak_to_total(xmin,xmax))
x_error = np.array(unp.std_devs(x))
y_error = np.array(unp.std_devs(y))

# print(np.sort(unp.nominal_values(y)))
# print(np.sort(unp.std_devs(y)))
# plt.errorbar(x=unp.nominal_values(x), y=unp.nominal_values(y), xerr=x_error, yerr=y_error, marker='.',linestyle = 'None')
plt.plot(unp.nominal_values(x), unp.nominal_values(y), marker='.',linestyle = 'None')
plt.fill_between(unp.nominal_values(x[:xmin+1]),unp.nominal_values(y[:xmin+1]),np.zeros(len(unp.nominal_values(y[:xmin+1]))), alpha=.25, color='blue')
plt.fill_between(unp.nominal_values(x[xmax:]),unp.nominal_values(y[xmax:]),np.zeros(len(unp.nominal_values(y[xmax:]))), alpha=.25, color='blue')
plt.fill_between(unp.nominal_values(x[xmin:xmax+1]),unp.nominal_values(y[xmin:xmax+1]),np.zeros(len(unp.nominal_values(y[xmin:xmax+1]))), alpha=.25, color='orange')
#plt.plot(x_data,y_data)
#plt.plot(x[0], z[0])
#plt.plot(x[0], y_background)
plt.xlabel("Kanal (c)")
plt.ylabel("Zählrate N")
plt.savefig("../pics/Peak-to-Total_"+element+'_'+detektortyp+".png")
plt.show()
