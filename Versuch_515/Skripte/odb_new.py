#Begin variables
#order:a
#End variables

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import genfromtxt
import scipy.odr as sodr
from scipy.optimize import curve_fit
import chart_studio.plotly as py


plt.figure(figsize=(11,5))
data = genfromtxt("odb.txt", delimiter=';')
xdata = data[1:,1]
ydata = data[1:,0]
xerror = data[1:,2]



# Fitting
z, error = np.polyfit(xdata, ydata, 10, cov=True)
f = np.poly1d(z)
print(f)
print(np.sqrt(np.diag(error)))

x_new = np.linspace(13, 515, 1000)
y_new = f(x_new)

plt.errorbar(xdata,ydata,xerr=xerror,fmt='.', markersize=1, zorder=0)
plt.plot(x_new, y_new, '-', color='red', zorder=10, label="$f(x)=a\cdot x^{10}+b\cdot x^{9}+c\cdot x^{8}+d\cdot x^{7}+e\cdot x^{6}+f\cdot x^{5}+g\cdot x^{4}+h\cdot x^{3}+i\cdot x^{2}+j\cdot x+k$\n$a=%.3f\cdot 10^{-25}$\n$b=%.3f\cdot 10^{-20}$\n$c=%.3f\cdot 10^{-17}$\n$d=%.3f\cdot 10^{-14}$\n$e=%.3f\cdot 10^{-12}$\n$f=%.3f\cdot 10^{-9}$\n$g=%.3f\cdot 10^{-7}$\n$h=%.3f\cdot 10^{-5}$\n$i=%.3f\cdot 10^{-3}$\n$j=%.3f\cdot 10^{-2}$\n$j=%.3f\cdot 10^{-4}$"%(f[10]*10**25,f[9]*10**20,f[8]*10**17,f[7]*10**14,f[6]*10**12,f[5]*10**9,f[4]*10**7,f[3]*10**5,f[2]*10**3,f[1]*10**2,f[0]*10**4))

data = genfromtxt("DTH.txt", delimiter=';')
x = data[1:,3]
y = data[1:,1]

plt.close()
x = f(x)
plt.plot(x,y)

plt.xlabel("Driftzeit / ns")
plt.ylabel("Spurabstand x / mm")
plt.legend(loc="lower right")
plt.show()

#plt.savefig("../../pics/ODB_new.pdf")
