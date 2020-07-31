import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy import interpolate
import mylib
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy import odr
import mylib

plt.figure(figsize=(10, 5))

untergrund = "./Datensätze/Untergrund-Table 1.csv"
untergrund_data = genfromtxt(untergrund, delimiter=';')
untergrund_l = unp.uarray(np.array(untergrund_data[1:-1,1]),np.sqrt(untergrund_data[1:-1,1]))
untergrund_r = unp.uarray(np.array(untergrund_data[1:-1,2]),np.sqrt(untergrund_data[1:-1,2]))

asymmetrie = "./Datensätze/Apparative_Asymmetrie-Table 1.csv"
asymmetrie_data = genfromtxt(asymmetrie, delimiter=",")
phi = unp.uarray(asymmetrie_data[1:, 0], 5)
time = unp.uarray(100*np.ones(len(phi)), np.ones(len(phi)))
time_untergrund = unp.uarray(400, 1)
L = unp.uarray(asymmetrie_data[1:, 1], np.sqrt(asymmetrie_data[1:,1]))-untergrund_l*time/time_untergrund
R = unp.uarray(asymmetrie_data[1:, 2], np.sqrt(asymmetrie_data[1:,2]))-untergrund_r*time/time_untergrund
y = L/R

plt.errorbar(x=unp.nominal_values(phi), y=unp.nominal_values(y), xerr=unp.std_devs(phi), yerr=unp.std_devs(y), linestyle="None")
plt.xlabel("$\phi$ / Grad")
plt.ylabel("L/R")
plt.savefig("../pics/Apparative_Asymmetrie.pdf")
plt.show()

print("phi; phi_error; delta; delta_error")
for i in range(0,len(phi)):
    print("%.1f; %.1f; %.3f; %.3f"%(unp.nominal_values(phi[i]), unp.std_devs(phi[i]), unp.nominal_values(y[i]),unp.std_devs(y[i])))
