#!/usr/bin/env python3

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

t_zero_delay = 0
t_zero_delay_2 = 1.5

file_nikolas = "./Nikolas/messdaten/prompt.txt"
file_dominic = "./Dominic/datensätze/Prompt_Kurve/Prompt_Kurve.txt"
data_nikolas = genfromtxt(file_nikolas, delimiter=';')
data_dominic = genfromtxt(file_dominic, delimiter=';')
x_nikolas = data_nikolas[0:, 0]
x_nikolas_error = np.ones(len(x_nikolas))*10
y_nikolas = data_nikolas[0:, 1]
y_nikolas_error=np.sqrt(y_nikolas)
y_nikolas_error[y_nikolas_error==0]=1
x_dominic = data_dominic[0:, 0]
x_dominic_error=np.ones(len(x_dominic))*10
y_dominic= data_dominic[0:, 1]
y_dominic_error=np.sqrt(y_dominic)
y_dominic_error[y_dominic_error==0]=1
#plt.show()

####
# Nikolas
####

plt.figure(dpi=100, figsize=(16,9))
plt.errorbar(x=x_nikolas, y=y_nikolas, yerr=y_nikolas_error, xerr=x_nikolas_error,linestyle = 'None')
a_nikolas = [160, 1317, 2576, 3857, 5044, 6367, 7584]
b_nikolas = [584, 1833, 3100, 4290, 5591, 6800, 8080]
p_nikolas = [[30, 440, 10, 10],[30, 1660, 100, 10],[30, 2814, 100, 10],[30, 4047, 100, 10],[30, 5323, 100, 10],[30, 6552, 100, 10],[30, 7822, 100, 10]]
t_nikolas = [16.0, 32.0, 48.0, 64.0, 80.0, 96.0, 112.0]
t_error_nikolas = np.ones(len(t_nikolas))
x_pos_nikolas = [0]*len(a_nikolas)
x_pos_error_nikolas = [0]*len(a_nikolas)
FWHM_nikolas = [0]*len(a_nikolas)
FWHM_error_nikolas = [0]*len(a_nikolas)
print(t_nikolas)

tmp_popt = np.ones(len(a_nikolas))
tmp_perr = np.ones(len(a_nikolas))
for i in range(0,len(a_nikolas)):
    popt,perr=mylib.anpassung_double_err(mylib.func_gauss,x_nikolas[a_nikolas[i]:b_nikolas[i]],x_nikolas_error[a_nikolas[i]:b_nikolas[i]],y_nikolas[a_nikolas[i]:b_nikolas[i]],y_nikolas_error[a_nikolas[i]:b_nikolas[i]],p_nikolas[i],True,"Verzögerung: "+str(np.round(t_nikolas[i]+t_zero_delay,1)))
    tmp_popt[i] = popt[1]
    tmp_perr[i] = perr[1]
    FWHM_nikolas[i]=2.35482*np.sqrt(popt[2])
    FWHM_error_nikolas[i]=2.35482/2/np.sqrt(popt[2])*perr[2]
    plt.plot([popt[1]-FWHM_nikolas[i]/2,popt[1]+FWHM_nikolas[i]/2],[popt[3]+popt[0]/2,popt[3]+popt[0]/2],color='black')
    #plt.plot(x_nikolas[a_nikolas[i]:b_nikolas[i]],mylib.func_gauss(p_nikolas[i],x_nikolas[a_nikolas[i]:b_nikolas[i]]),color='grey')
    x_pos_nikolas[i]=popt[1]
    x_pos_error_nikolas[i]=perr[1]
plt.legend()
plt.xlabel("Channel c")
plt.ylabel("Zählrate n")
plt.savefig("../pics/Promptkurve_Nikolas.pdf")
plt.show()

plt.figure(dpi=100, figsize=(16,9))
print("\n",x_pos_nikolas, x_pos_error_nikolas)
print("\n",t_nikolas, t_error_nikolas)
popt_lin_nikolas,perr_lin_nikolas=mylib.anpassung_double_err(mylib.func_lin,x_pos_nikolas,x_pos_error_nikolas,t_nikolas, t_error_nikolas,[1.,1.],False, "")
print("\n", popt_lin_nikolas, perr_lin_nikolas)
m_lin_nikolas = unp.uarray(popt_lin_nikolas[0], perr_lin_nikolas[0])
b_lin_nikolas = unp.uarray(popt_lin_nikolas[1], perr_lin_nikolas[1])
x_lin_nikolas = np.linspace(0, np.max(x_nikolas), 1000)
y_lin_nikolas = m_lin_nikolas * x_lin_nikolas + b_lin_nikolas +t_zero_delay

Zeitauflösung = unp.uarray(0,0)
ges_Zeitauflösung = unp.uarray(0,0)

print("\n"+"Nr.& $\Delta$ t / ns & $x_c$ & $\Delta x_c$ & FWHM / c & FWHM / ns &  $A_t$ & $\Delta A_t$\\\\\\hline")
for i in range(0, len(a_nikolas)):
    FWHM_time = (unp.uarray(tmp_popt[i], tmp_perr[i])+unp.uarray(FWHM_nikolas[i], FWHM_error_nikolas[i])/2)*m_lin_nikolas-(unp.uarray(tmp_popt[i], tmp_perr[i])-unp.uarray(FWHM_nikolas[i], FWHM_error_nikolas[i])/2)*m_lin_nikolas
    #print(unp.nominal_values(FWHM_time[i]))
    Zeitauflösung = unp.uarray(tmp_popt[i], tmp_perr[i])/unp.uarray(FWHM_nikolas[i], FWHM_error_nikolas[i])

    ges_Zeitauflösung += Zeitauflösung
    print(str(i)+"&"+str(t_nikolas[i]+t_zero_delay)+"&"+str(round(tmp_popt[i],2))+"&"+str(round(tmp_perr[i],2))+"&"+str(round(FWHM_nikolas[i],2))+"$\pm$"+str(round(FWHM_error_nikolas[i],2))+"&"+str(round(float(unp.nominal_values(FWHM_time)),2))+"$\pm$"+str(round(float(unp.std_devs(FWHM_time)),2))+"&"+str(round(float(unp.nominal_values(Zeitauflösung)),4))+"&"+str(round(float(unp.std_devs(Zeitauflösung)),4))+"\\\\")


ges_Zeitauflösung = np.sum(unp.uarray(tmp_popt, tmp_perr)/unp.uarray(FWHM_nikolas, FWHM_error_nikolas))/len(a_nikolas)
print("\nZEITAUFLÖSUNG",ges_Zeitauflösung,"\n")

plt.plot(unp.nominal_values(x_lin_nikolas), unp.nominal_values(y_lin_nikolas), label="$t(c)=(%.2f\pm%.2f)$ps$\cdot c+(%.2f\pm%.2f)$ns"%(popt_lin_nikolas[0]*10**3, perr_lin_nikolas[0]*10**3,popt_lin_nikolas[1], perr_lin_nikolas[1]))
plt.legend()
plt.xlabel("Channel c")
plt.ylabel("Zeit / ns")
plt.errorbar(x=x_pos_nikolas, y=t_nikolas+np.ones(len(a_nikolas))*t_zero_delay, yerr=t_error_nikolas, xerr=x_pos_error_nikolas,linestyle = 'None')
plt.savefig("../pics/Zeitkalibration_Nikolas.pdf")
plt.show()



####
# Dominic
####
plt.figure(dpi=100, figsize=(16,9))
plt.errorbar(x=x_dominic, y=y_dominic, yerr=y_dominic_error, xerr=x_dominic_error,linestyle = 'None')
a_dominic = [160, 1317, 2576, 3857, 5044, 6367]
b_dominic = [739, 2029, 3237, 4456, 5684, 6934]
p_dominic = [[30, 440, 10, 10],[30, 1660, 100, 10],[30, 2814, 100, 10],[30, 4047, 100, 10],[30, 5323, 100, 10],[30, 6552, 100, 10]]
t_dominic = [0.00000000001, 16.0, 32.0, 48.0, 64.0, 80.0]
t_error_dominic = np.ones(len(t_dominic))
x_pos_dominic = [0]*len(a_dominic)
x_pos_error_dominic = [0]*len(a_dominic)
FWHM_dominic = [0]*len(a_dominic)
FWHM_error_dominic = [0]*len(a_dominic)
for i in range(0,len(a_dominic)):
    popt,perr=mylib.anpassung_double_err(mylib.func_gauss,x_dominic[a_dominic[i]:b_dominic[i]],x_dominic_error[a_dominic[i]:b_dominic[i]],y_dominic[a_dominic[i]:b_dominic[i]],y_dominic_error[a_dominic[i]:b_dominic[i]],p_dominic[i],True,"Verzögerung: "+str(np.round(t_dominic[i]+t_zero_delay_2,1)))
    FWHM_dominic[i]=2.35482*np.sqrt(popt[2])
    FWHM_error_dominic[i]=2.35482/2/np.sqrt(popt[2])*perr[2]
    plt.plot([popt[1]-FWHM_dominic[i]/2,popt[1]+FWHM_dominic[i]/2],[popt[3]+popt[0]/2,popt[3]+popt[0]/2],color='black')
    #plt.plot(x_dominic[a_dominic[i]:b_dominic[i]],mylib.func_gauss(p_dominic[i],x_dominic[a_dominic[i]:b_dominic[i]]),color='grey')
    #print("$"+str(round(popt[0],2))+"\pm"+str(round(perr[0],2))+"$&$"+str(round(popt[1],2))+"\pm"+str(round(perr[1],2))+"$&$"+str(round(popt[2],2))+"\pm"+str(round(perr[2],2))+"$&$"+str(round(popt[3],2))+"\pm"+str(round(perr[3],2))+"$&\\\\")
    #print(popt[1],perr[1])
    x_pos_dominic[i]=popt[1]
    x_pos_error_dominic[i]=perr[1]
plt.xlabel("Channel c")
plt.ylabel("Zählrate n")
plt.legend()
plt.savefig("../pics/Promptkurve_Dominic.pdf")
plt.show()

plt.figure(dpi=100, figsize=(16,9))

popt_lin_dominic,perr_lin_dominic=mylib.anpassung_double_err(mylib.func_lin,x_pos_dominic,x_pos_error_dominic,t_dominic, t_error_dominic,[1.,1.],False, "")
m_lin_dominic = unp.uarray(popt_lin_dominic[0], perr_lin_dominic[0])
b_lin_dominic = unp.uarray(popt_lin_dominic[1], perr_lin_dominic[1])
x_lin_dominic = np.linspace(0, np.max(x_dominic), 1000)
y_lin_dominic = m_lin_dominic * x_lin_dominic + b_lin_dominic +t_zero_delay_2
plt.plot(unp.nominal_values(x_lin_dominic), unp.nominal_values(y_lin_dominic), label="$t(c)=(%.2f\pm%.2f)$ps$\cdot c+(%.2f\pm%.2f)$ns"%(popt_lin_dominic[0]*10**3, perr_lin_dominic[0]*10**3,popt_lin_dominic[1], perr_lin_dominic[1]))
plt.legend()
plt.xlabel("Channel c")
plt.ylabel("Zeit / ns")
plt.errorbar(x=x_pos_dominic, y=t_dominic+np.ones(len(a_dominic))*t_zero_delay_2, yerr=t_error_dominic, xerr=x_pos_error_dominic,linestyle = 'None')
plt.savefig("../pics/Zeitkalibration_Dominic.pdf")
plt.show()

####
# Lebensdauermessung
####


lebensdauer_nikolas = "./Nikolas/messdaten/lebensdauerkurve.txt"
lebensdauer_dominic = "./Dominic/datensätze/Lebensdauerkurve/Lebensdauerkurve.txt"
lebensdauer_nikolas = genfromtxt(lebensdauer_nikolas, delimiter=';')
lebensdauer_dominic = genfromtxt(lebensdauer_dominic, delimiter=';')
x_leben_nikolas = unp.uarray(lebensdauer_nikolas[0:, 0],np.ones(len(lebensdauer_nikolas[0:, 0]))*10)
x_leben_nikolas = m_lin_nikolas * x_leben_nikolas + b_lin_nikolas +t_zero_delay
y_leben_nikolas = unp.uarray(lebensdauer_nikolas[0:, 1],np.sqrt(lebensdauer_nikolas[0:, 1])+1)
x_leben_dominic = unp.uarray(lebensdauer_dominic[0:, 0],np.ones(len(lebensdauer_dominic[0:, 0]))*10)
x_leben_dominic = m_lin_dominic * x_leben_dominic + b_lin_dominic +t_zero_delay_2
y_leben_dominic = unp.uarray(lebensdauer_dominic[0:, 1],np.sqrt(lebensdauer_dominic[0:, 1])+1)

def exp(p, x):
    a, b, t = p
    return a*np.exp(-x/t)+b

def lebensdauer(p,x, x_error, y, y_error):
    model = odr.Model(exp)
    data = odr.RealData(x, y, sx=x_error, sy=y_error)
    out = odr.ODR(data, model, beta0=p).run()
    popt = out.beta
    perr = out.sd_beta
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = exp(popt, x_fit)
    print("Exponential Funciton\n",popt, perr)
    plt.plot(x_fit, y_fit, label="$N(t)=N_0\cdot \exp(-\\frac{t}{\\tau})+c$\n$N_0=(%.3f\pm%.3f)$\n$\\tau=(%.3f\pm%.3f)ns$\n$c=(%.3f\pm%.3f)$"%(popt[0],perr[0],popt[2],perr[2],popt[1],perr[1]))
plt.figure(dpi=100, figsize=(16,9))
plt.errorbar(x=unp.nominal_values(x_leben_nikolas),y=unp.nominal_values(y_leben_nikolas),xerr=unp.std_devs(x_leben_nikolas),yerr=unp.std_devs(y_leben_nikolas),linestyle = 'None')
p_leben_nikolas = [120,10,10]
a_leben_nikolas = 2150
b_leben_nikolas = 8000
lebensdauer(p_leben_nikolas, unp.nominal_values(x_leben_nikolas[a_leben_nikolas:b_leben_nikolas]),unp.std_devs(x_leben_nikolas[a_leben_nikolas:b_leben_nikolas]), unp.nominal_values(y_leben_nikolas[a_leben_nikolas:b_leben_nikolas]),unp.std_devs(y_leben_nikolas[a_leben_nikolas:b_leben_nikolas]))
plt.legend()
plt.xlabel("Zeit t / ns")
plt.ylabel("Zählrate n")
plt.savefig("../pics/Lebensdauerkurve_Nikolas.pdf")
plt.show()

plt.figure(dpi=100, figsize=(16,9))
plt.errorbar(x=unp.nominal_values(x_leben_dominic),y=unp.nominal_values(y_leben_dominic),xerr=unp.std_devs(x_leben_dominic),yerr=unp.std_devs(y_leben_dominic),linestyle = 'None')
p_leben_dominic = [120,10,10]
a_leben_dominic = 2300
b_leben_dominic = 8000
lebensdauer(p_leben_dominic, unp.nominal_values(x_leben_dominic[a_leben_dominic:b_leben_dominic]),unp.std_devs(x_leben_dominic[a_leben_dominic:b_leben_dominic]), unp.nominal_values(y_leben_dominic[a_leben_dominic:b_leben_dominic]),unp.std_devs(y_leben_dominic[a_leben_dominic:b_leben_dominic]))
plt.legend()
plt.xlabel("Zeit t / ns")
plt.ylabel("Zählrate n")
plt.savefig("../pics/Lebensdauerkurve_Dominic.pdf")
plt.show()
