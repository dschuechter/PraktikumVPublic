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
time_untergrund = unp.uarray(400, 1)
U_pi = unp.uarray(np.array([1281,1254]), np.sqrt([1281,1254]))/time_untergrund
U_0 = unp.uarray(np.array([1224,1235]), np.sqrt([1224,1235]))/time_untergrund

asymm_0 = unp.uarray(1.118 , 0.044)
asymm_pi = unp.uarray(1.159, 0.047)


file = "./Datensätze/Mottasymmetrie.csv"
data = genfromtxt(file, delimiter=',')
thickness = unp.uarray(data[1:, 0], np.ones(len(data[1:, 0]))*0.1)
time = unp.uarray(data[1:, 7], 1)

L_0 = unp.uarray(data[1:, 2], np.sqrt(data[1:,2]))-U_0[0]*time
R_0 = unp.uarray(data[1:, 3], np.sqrt(data[1:,3]))-U_0[1]*time
L_pi = unp.uarray(data[1:, 5], np.sqrt(data[1:,5]))-U_pi[0]*time
R_pi = unp.uarray(data[1:, 6], np.sqrt(data[1:,6]))-U_pi[1]*time

#print(thickness)
#print(L_0)
#print(R_0)
#print(L_pi)
#print(R_pi)
#print(time)

delta = (1-unp.sqrt((L_pi/R_pi)/(L_0/R_0)*asymm_0/asymm_pi))/(1+unp.sqrt((L_pi/R_pi)/(L_0/R_0)*asymm_0/asymm_pi))
#print(delta)
delta=1/delta
popt, perr =mylib.anpassung_double_err(mylib.func_lin_mott, unp.nominal_values(thickness),unp.std_devs(thickness),unp.nominal_values(delta), unp.std_devs(delta), [1,1] , True, "$\frac{1}{\delta^*}=\frac{1}{\delta}+\alpha\cdot t$")
#popt, perr = mylib.anpassung_yerr(mylib.func_lin, unp.nominal_values(thickness),unp.nominal_values(delta), unp.std_devs(delta), [1,1] , True, "$\frac{1}{\delta^*}=\frac{1}{\delta}+\alpha\cdot t$")
plt.errorbar(x=unp.nominal_values(thickness), y=unp.nominal_values(delta),xerr=unp.std_devs(thickness), yerr=unp.std_devs(delta), linestyle="None")
print(popt, perr)
plt.xlim(left=0)
plt.xlabel("Dicke t / $\mu$m")
plt.ylabel("$1/\\delta^*$")
plt.savefig("../pics/Mottstreuung.pdf")
plt.show()

print("$t$ & $\Delta t$ & $1/\delta$ & $\Delta 1/\delta$")
for i in range(0, len(thickness)):
    print(str(np.round(unp.nominal_values(thickness[i]),1))+"&"+str(np.round(unp.std_devs(thickness[i]),1))+"&"+str(np.round(unp.nominal_values(delta[i]),2))+"&"+str(np.round(unp.std_devs(delta[i]),2))+"\\\\")
print("\delta = "+str(1/popt[1])+"$\pm$"+str(1/popt[1]**2*perr[1]))
print("P_{0}=-\delta/0.25="+str(-1/popt[1]/0.25)+"$\pm$"+str(1/popt[1]**2*perr[1]/0.25))
