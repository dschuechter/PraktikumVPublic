#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate
import mylib

plt.figure(dpi=100, figsize=(5,3))

file = sys.argv[1]
data = genfromtxt(file, delimiter=';')
x = data[0:, 0]
x_error=np.ones(len(x))*3
y= data[0:, 1]
y_error=np.sqrt(y)
# plt.plot(x, y, '-',linewidth=1,alpha=.5)
plt.errorbar(x,y,xerr=x_error,yerr=y_error,linewidth=1,alpha=.25,zorder=-10)

#-------------------------
# Ermittlung der Kanal Maxima
#-------------------------

#links
p=[[2500,500,100,100],[1000,1240,1,200],[100,4200,1,100],[200,4950,1,150],[100,6900,1,50]]
a=[250,1130,4000,4640,6630]
b=[650,1300,4430,5150,7400]

#rechts
# p=[[2000,200,100,200],[1000,950,100,200],[100,4000,1,100],[200,4600,1,150],[100,6900,1,50]]
# a=[140,870 ,3700,4200,6200]
# b=[240,1100,4200,4900,7200]
print("a&b&c&d\\")

for i in range(0,len(a)):
    # plt.plot(x[a[i]:b[i]],mylib.func_gauss(p[i],x[a[i]:b[i]]),color='grey')
    popt,perr=mylib.anpassung_double_err(mylib.func_gauss,x[a[i]:b[i]],x_error[a[i]:b[i]],y[a[i]:b[i]],y_error[a[i]:b[i]],p[i],False,False)
    mylib.anpassung_double_err(mylib.func_gauss,x[a[i]:b[i]],x_error[a[i]:b[i]],y[a[i]:b[i]],y_error[a[i]:b[i]],p[i],True,"k(max)="+str(round(popt[1],1)))
    print("$"+str(round(popt[0],2))+"\pm"+str(round(perr[0],2))+"$&$"+str(round(popt[1],2))+"\pm"+str(round(perr[1],2))+"$&$"+str(round(popt[2],2))+"\pm"+str(round(perr[2],2))+"$&$"+str(round(popt[3],2))+"\pm"+str(round(perr[3],2))+"$&\\\\")
    # print(popt[1],perr[1])

# plt.title(file)
plt.xlabel('Kanal')
plt.ylabel('N')
plt.legend()
# plt.savefig('../pics/NaBaSpektrum_links.pdf')
# plt.show()
plt.close()



#-------------------------
# Energiekalibration
#-------------------------

file = "Ekal_links.txt"
data = genfromtxt(file, delimiter=';')
t = 6
s = t+5
x = data[t:s, 0]
x_error=data[t:s,1]
y= data[t:s, 2]
plt.errorbar(x, y,xerr=x_error, fmt='x',label="linker Aufbau")
popt_links,perr_links=mylib.anpassung_xerr(mylib.func_lin,x,y,x_error,[1.,1.],True,"linker Aufbau")
print(str(round(popt_links[0]*100,2))+"\pm"+str(round(perr_links[0]*100,2))+"$\\cdot k$"+str(round(popt_links[1],2))+"\pm"+str(round(perr_links[1],2)))

file = "Ekal_rechts.txt"
data = genfromtxt(file, delimiter=';')
x = data[t:s, 0]
x_error=data[t:s,1]
y= data[t:s, 2]
plt.errorbar(x, y,xerr=x_error, fmt='x',label="rechter Aufbau")
popt_rechts,perr_rechts=mylib.anpassung_xerr(mylib.func_lin,x,y,x_error,[1.,1.],True,"rechter Aufbau")
print(str(round(popt[0]*100,2))+"\pm"+str(round(perr[0]*100,2))+"$\\cdot k$"+str(round(popt[1],2))+"\pm"+str(round(perr[1],2)))
plt.legend()
plt.xlabel('Kanal k')
plt.ylabel("Energie E / keV")
# plt.savefig("../pics/Energiekalibration.pdf")
# plt.show()
plt.close()



#-------------------------
# Energiespektren und FWHM bestimmung von 81 356 511
#-------------------------
lamda=[81,356,511]

plt.figure(dpi=100, figsize=(10,5))
file = "Dominic/datensätze/Energiedspektrum der NA-Quelle identifizieren und Verstärkung einstellen/Energiespektrum_links_mit_Ba.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_links[0]*x0+popt_links[1]
x_error=np.sqrt((popt_links[0]*x0_error)**2+(perr_links[0]*x0)**2+(perr_links[1])**2)
# plt.plot(x,y, label='linker Aufbau',alpha=.5,zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='linker Aufbau',alpha=.25,zorder=-100,color='mediumaquamarine')

#links
p=[[1000,81,10,0,250],[200,356,100,0,150],[100,511,100,0,50]]
a=[1030,4540,6530]
b=[1400,5150,7500]

for i in range(0,len(a)):
    # plt.plot(x[a[i]:b[i]],mylib.func_gauss(p[i],x[a[i]:b[i]]),color='grey')
    popt,perr=mylib.anpassung_double_err(mylib.func_gauss_lin,x[a[i]:b[i]],x_error[a[i]:b[i]],y[a[i]:b[i]],y_error[a[i]:b[i]],p[i],True,"$\\lambda=$"+str(lamda[i]))
    # print("links&$"+str(round(popt[0],1))+"\pm"+str(round(perr[0],1))+"$&$"+str(round(popt[1],2))+"\pm"+str(round(perr[1],2))+"$&$"+str(round(popt[2],2))+"\pm"+str(round(perr[2],2))+"$&$"+str(round(2.35482*np.sqrt(popt[2]),2))+"\\pm"+str(round(2.35482/2/np.sqrt(popt[2])*perr[2],2))+"$&$"+str(round(popt[3],1))+"\pm"+str(round(perr[3],1))+"$&$"+str(round(popt[4],1))+"\pm"+str(round(perr[4],1))+"$\\\\")
    # print("&$"+str(round(2.35482*np.sqrt(popt[2]),2))+"\\pm"+str(round(2.35482/2/np.sqrt(popt[2])*perr[2],2))+"$")
    # print(popt[1],perr[1])

file = "Dominic/datensätze/Energiedspektrum der NA-Quelle identifizieren und Verstärkung einstellen/Energiespektrum_rechts_mit_Ba.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_rechts[0]*x0+popt_rechts[1]
x_error=np.sqrt((popt_rechts[0]*x0_error)**2+(perr_rechts[0]*x0)**2+(perr_rechts[1])**2)
# plt.plot(x,y, label='rechter Aufbau',alpha=.5,zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='rechter Aufbau',alpha=.25,zorder=-300,color='lightcoral')

#rechts
p=[[1000,81,10,0,250],[200,356,100,0,150],[100,511,100,0,50]]
a=[770 ,4200,6200]
b=[1150,4900,7200]

for i in range(0,len(a)):
    # plt.plot(x[a[i]:b[i]],mylib.func_gauss(p[i],x[a[i]:b[i]]),color='grey')
    popt,perr=mylib.anpassung_double_err(mylib.func_gauss_lin,x[a[i]:b[i]],x_error[a[i]:b[i]],y[a[i]:b[i]],y_error[a[i]:b[i]],p[i],True,"$\\lambda=$"+str(lamda[i]))
    # print("rechts&$"+str(round(popt[0],1))+"\pm"+str(round(perr[0],1))+"$&$"+str(round(popt[1],2))+"\pm"+str(round(perr[1],2))+"$&$"+str(round(popt[2],2))+"\pm"+str(round(perr[2],2))+"$&$"+str(round(2.35482*np.sqrt(popt[2]),2))+"\\pm"+str(round(2.35482/2/np.sqrt(popt[2])*perr[2],2))+"$&$"+str(round(popt[3],1))+"\pm"+str(round(perr[3],1))+"$&$"+str(round(popt[4],1))+"\pm"+str(round(perr[4],1))+"$\\\\")
    # print("&$"+str(round(2.35482*np.sqrt(popt[2]),1))+"\\pm"+str(round(2.35482/2/np.sqrt(popt[2])*perr[2],1))+"$")
    # print(popt[1],perr[1])

plt.legend()
plt.xlabel('Energie / keV')
plt.ylabel(r"N")
# plt.savefig('../pics/Energiespektrum.pdf')
# plt.show()
plt.close()

####
# Spektrum SCA 511keV
####

plt.figure(dpi=100, figsize=(10,5))
file = "Dominic/datensätze/Einkanalfenster/511_links.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_links[0]*x0+popt_links[1]
x_error=np.sqrt((popt_links[0]*x0_error)**2+(perr_links[0]*x0)**2+(perr_links[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='linker Aufbau',alpha=1,zorder=-100)

file = "Dominic/datensätze/Einkanalfenster/511_rechts.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_rechts[0]*x0+popt_rechts[1]
x_error=np.sqrt((popt_rechts[0]*x0_error)**2+(perr_rechts[0]*x0)**2+(perr_rechts[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='rechter Aufbau',alpha=0.25,zorder=-50)

plt.legend()
plt.xlabel('Energie / keV')
plt.ylabel('N')
plt.xlim(440,580)
# plt.savefig('../pics/SCA_511.pdf')
# plt.show()
plt.close()

####
# Spektrum SCA 511keV
####

file = "Dominic/datensätze/Einfallsfenster für Ba einstellen/Einkanalfenster_Ba_links.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_links[0]*x0+popt_links[1]
x_error=np.sqrt((popt_links[0]*x0_error)**2+(perr_links[0]*x0)**2+(perr_links[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='linker Aufbau',alpha=1,zorder=-100)

plt.xlabel('Energie / keV')
plt.ylabel('N')
plt.xlim(310,420)
# plt.savefig('../pics/SCA_356.pdf')
# plt.show()

file = "Dominic/datensätze/Einfallsfenster für Ba einstellen/Einkanalfenster_Ba_rechts.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_rechts[0]*x0+popt_rechts[1]
x_error=np.sqrt((popt_rechts[0]*x0_error)**2+(perr_rechts[0]*x0)**2+(perr_rechts[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='rechter Aufbau',alpha=1,zorder=-50)

# plt.legend()
plt.xlabel('Energie / keV')
plt.ylabel('N')
plt.xlim(60,100)
# plt.savefig('../pics/SCA_81.pdf')
# plt.show()
plt.close()

####
# CDF
####

file = "Dominic/datensätze/CFD-Diskriminatorschwelle/Diskriminatorschwelle_links.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_links[0]*x0+popt_links[1]
x_error=np.sqrt((popt_links[0]*x0_error)**2+(perr_links[0]*x0)**2+(perr_links[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='linker Aufbau',alpha=.25,zorder=-10)

file = "Dominic/datensätze/CFD-Diskriminatorschwelle/Diskriminatorschwelle_rechts.txt"
data = genfromtxt(file, delimiter=';')
x0 = data[0:, 0]
x0_error=np.ones(len(x0))*3
y= data[0:, 1]
y_error=np.sqrt(y)
x=popt_rechts[0]*x0+popt_rechts[1]
x_error=np.sqrt((popt_rechts[0]*x0_error)**2+(perr_rechts[0]*x0)**2+(perr_rechts[1])**2)
# plt.plot(x,y, label='linker Aufbau',zorder=-3)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, label='rechter Aufbau',alpha=1,zorder=-50)

plt.legend()
plt.xlabel('Energie / keV')
plt.ylabel('N')
plt.xlim(0,600)
plt.savefig('../pics/CFD.pdf')
plt.show()
