import sys
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties import ufloat

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
y[y < 0] = 0

xmin = int(eckdaten[1,0])
xmax = int(eckdaten[1,1])

#Finde Peak to total Verhältnis
total_counts = np.sum(y)
y_error = np.array(unp.std_devs(y))
y = unp.nominal_values(y)
def peak_to_total(a, b, delta):
	peak_int = np.sum(y[a:b])
	peak_int_error = (((y[a]*y[a])+(y[b]*y[b]))*delta**2)**(1/2)
	if datensatz == 'Co_H':
		peak_int = np.sum(y[a:6195])+np.sum(y[6990:b])
		peak_int_error = (((y[a]*y[a])+(y[6195]*y[6195]))*delta**2)**(1/2)+(((y[6990]*y[6990])+(y[b]*y[b]))*delta**2)**(1/2)
	peak_to_total = peak_int/unp.nominal_values(total_counts)
	peak_to_total_error = np.sqrt((peak_int_error/unp.nominal_values(total_counts))**2+(peak_int/unp.nominal_values(total_counts)**2*unp.std_devs(total_counts))**2)
	# return peak_to_total
	return peak_to_total,peak_to_total_error

print("Peak-to-Total: ",peak_to_total(xmin,xmax,eckdaten[1,2]))
x_error = np.array(unp.std_devs(x))
y = unp.uarray(y,y_error)

# print(np.sort(unp.nominal_values(y)))
# print(np.sort(unp.std_devs(y)))
# plt.errorbar(x=unp.nominal_values(x), y=unp.nominal_values(y), xerr=x_error, yerr=y_error, marker='.',linestyle = 'None')
plt.errorbar(unp.nominal_values(x), unp.nominal_values(y), xerr=unp.std_devs(x), yerr=unp.std_devs(y),linestyle = 'None')
plt.fill_between(unp.nominal_values(x[:xmin+1]),unp.nominal_values(y[:xmin+1]),np.zeros(len(unp.nominal_values(y[:xmin+1]))), alpha=.25, color='blue')
plt.fill_between(unp.nominal_values(x[xmax:]),unp.nominal_values(y[xmax:]),np.zeros(len(unp.nominal_values(y[xmax:]))), alpha=.25, color='blue')
if datensatz == 'Co_H':
	plt.fill_between(unp.nominal_values(x[xmin:6195]),unp.nominal_values(y[xmin:6195]),np.zeros(len(unp.nominal_values(y[xmin:6195]))), alpha=.25, color='orange')
	plt.fill_between(unp.nominal_values(x[6194:6991]),unp.nominal_values(y[6194:6991]),np.zeros(len(unp.nominal_values(y[6194:6991]))), alpha=.25, color='blue')
	plt.fill_between(unp.nominal_values(x[6990:xmax+1]),unp.nominal_values(y[6990:xmax+1]),np.zeros(len(unp.nominal_values(y[6990:xmax+1]))), alpha=.25, color='orange')
else:
	plt.fill_between(unp.nominal_values(x[xmin:xmax+1]),unp.nominal_values(y[xmin:xmax+1]),np.zeros(len(unp.nominal_values(y[xmin:xmax+1]))), alpha=.25, color='orange')
#plt.plot(x_data,y_data)
#plt.plot(x[0], z[0])
#plt.plot(x[0], y_background)
plt.xlabel("Energie in keV")
plt.ylabel("Zählrate N")
plt.savefig("../pics/Peak-to-Total_"+element+'_'+detektortyp+".png")
plt.show()
