#!/usr/bin/env python3

import sys
import argparse
import uncertainties.unumpy as unp
from uncertainties import ufloat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.odr as sodr
from numpy import genfromtxt
import argparse
import mylib
time_untergrund = unp.uarray(400, 1)
plt.figure(figsize=(10, 5))
untergrund = "./Datensätze/Untergrund-Table 1.csv"
untergrund_data = genfromtxt(untergrund, delimiter=';')
untergrund_l = unp.uarray(np.array(untergrund_data[1:-1,1]),np.sqrt(untergrund_data[1:-1,1]))/time_untergrund
untergrund_r = unp.uarray(np.array(untergrund_data[1:-1,2]),np.sqrt(untergrund_data[1:-1,2]))/time_untergrund

asymmetrie_data = genfromtxt("./Datensätze/apparative_asymmetrie_faktor.txt", delimiter=';')
asymmetrie = unp.uarray(asymmetrie_data[1:,2],asymmetrie_data[1:,3])

my_data = genfromtxt('./Datensätze/Mottsreuung_duenne_Folie-Table 1.csv', delimiter=';')
x = unp.uarray(my_data[1:, 0]*np.pi, np.ones(len(my_data[1:, 0]))*5*np.pi)/180
n_links = unp.uarray(my_data[1:, 1],np.sqrt(my_data[1:,1]))-untergrund_l*200
n_rechts = unp.uarray(my_data[1:,2],np.sqrt(my_data[1:,2]))-untergrund_r*200
y=n_links/n_rechts/asymmetrie

plt.errorbar(unp.nominal_values(x)*180/np.pi,unp.nominal_values(y),yerr=unp.std_devs(y),xerr=unp.std_devs(x)*180/np.pi, linestyle = "None")

popt, perr = mylib.anpassung_double_err(mylib.func_cos_delta_527,unp.nominal_values(x), unp.std_devs(x), unp.nominal_values(y), unp.std_devs(y),[1,0.3,0],True,False)
print(popt, perr)
print("\delta=(%.3f\pm%.3f)\n\Theta=(%.3f\pm%.3f)\na=(%.3f\pm%.3f)"%(popt[1],perr[1],np.degrees(popt[2]),np.degrees(perr[2]),popt[0],perr[0]))
plt.plot(unp.nominal_values(x), unp.nominal_values(y), lw=2, linestyle = "None")
plt.xlabel("$\phi$ / Grad")
plt.ylabel("L/R")
plt.savefig("../pics/DünneFolie.pdf")
plt.show()


phi = my_data[1:, 0]
print("\nkorrigiert \n")
print("$\phi$ / Grad & L/R\\\ \n\hline")
for i in range(0, len(y)):
    print(str(np.round(phi[i],0))+"&"+str(np.round(unp.nominal_values(y[i]),2))+"$\pm$"+str(np.round(unp.std_devs(y[i]),2))+"\\\\")
#####
# Nochmal plotten ohne Apparative_Asymmetrie_korrektur
#####
plt.figure(figsize=(10, 5))
untergrund = "./Datensätze/Untergrund-Table 1.csv"
untergrund_data = genfromtxt(untergrund, delimiter=';')
untergrund_l = unp.uarray(np.array(untergrund_data[1:-1,1]),np.sqrt(untergrund_data[1:-1,1]))/400
untergrund_r = unp.uarray(np.array(untergrund_data[1:-1,2]),np.sqrt(untergrund_data[1:-1,2]))/400

asymmetrie_data = genfromtxt("./Datensätze/apparative_asymmetrie_faktor.txt", delimiter=';')
asymmetrie = unp.uarray(asymmetrie_data[1:,2],asymmetrie_data[1:,3])

my_data = genfromtxt('./Datensätze/Mottsreuung_duenne_Folie-Table 1.csv', delimiter=';')

x = unp.uarray(my_data[1:, 0]*np.pi, np.ones(len(my_data[1:, 0]))*5*np.pi)/180
n_links = unp.uarray(my_data[1:, 1],np.sqrt(my_data[1:,1]))-untergrund_l*200
n_rechts = unp.uarray(my_data[1:,2],np.sqrt(my_data[1:,2]))-untergrund_r*200
y=n_links/n_rechts#/asymmetrie

plt.errorbar(unp.nominal_values(x)*180/np.pi,unp.nominal_values(y),yerr=unp.std_devs(y),xerr=unp.std_devs(x)*180/np.pi, linestyle = "None")

mylib.anpassung_double_err(mylib.func_cos_delta_527,unp.nominal_values(x), unp.std_devs(x), unp.nominal_values(y), unp.std_devs(y),[1,0.3,0],True,False)
plt.plot(unp.nominal_values(x), unp.nominal_values(y), lw=2, linestyle = "None")
plt.xlabel("$\phi$ / Grad")
plt.ylabel("L/R")
plt.savefig("../pics/DünneFolie_unkorrigierte_asymmetrie.pdf")
plt.show()
print("unkorrigiert \n\n")
print("$\phi$ / Grad & L/R\\\ \n\hline")
for i in range(0, len(y)):
    print(str(np.round(phi[i],0))+"&"+str(np.round(unp.nominal_values(y[i]),2))+"$\pm$"+str(np.round(unp.std_devs(y[i]),2))+"\\\\")
