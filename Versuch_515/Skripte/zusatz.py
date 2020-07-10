#!/usr/bin/env python3

import os
import sys
import argparse

import sympy as sy

import numpy as np
from numpy import genfromtxt

import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

import matplotlib._color_data as mcd

#erstelle Array mit vielen Farbnamen zum späteren Plotten und auseinanderhalten der Ergeignisse
colorarr=[name for name in mcd.CSS4_COLORS
           if "xkcd:" + name in mcd.XKCD_COLORS]

#Bestimme den Abstand vom Draht aus der Dirftzeit
def odb(time):
    a =3.263803146996604e-24
    b =-1.1405849140216569e-20
    c =1.713052649211002e-17
    d =-1.4424668447880543e-14
    e =7.445575231571847e-12
    f =-2.4179091169613853e-09
    g =4.872805579474163e-07
    h =-5.7647773107223793e-05
    i =0.0034629936913756054
    j =-0.034869184588697595
    k =0.00066032444743083
    return a*time**10+b*time**9+c*time**8+d*time**7+e*time**6+f*time**5+g*time**4+h*time**3+i*time**2+j*time**1+k

#löse Das gleichungsystem zur bestimmung der möglichen teilchenflugbahnen
#r_{a}^2=(a-a')^2, r_b=...
#g(x1,x2)=m*x+c, m1=1, c2=0
#a1'=c1+m1*la, a2'=c2+m2*la
#b1'=..., b2'=...
def solve_for_winkel(a1,a2,b1,b2,ra,rb):
    c1,m2,la,lb=sy.symbols("c1 m2 la lb")
    ra=odb(ra)
    rb=odb(rb)
    equations=[
        sy.Eq((a1-(c1+1*la))**2+(a2-(0+m2*la))**2,ra**2),
        sy.Eq((a1-(c1+1*la))*1+(a2-(0+m2*la))*m2,0),
        sy.Eq((b1-(c1+1*lb))**2+(b2-(0+m2*lb))**2,rb**2),
        sy.Eq((b1-(c1+1*lb))*1+(b2-(0+m2*lb))*m2,0)]
    return sy.solve(equations)

#Gerade, die die Flugbahn der teilchen beschreibt.
def g(x,m2,c1):
    return m2*(x-c1)

#Umkehrfunktion von g. Zur Kontrolle des Szintillatordurchgangs
def g_umkehr(g, m, c):
    return g/m+c

c1,m2,la,lb=sy.symbols("c1 m2 la lb")

#Koordinaten der 48 Drähte
x_wire=np.arange(1,49,1)*8.5
y_wire=[0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,15]

#Szintillatorposition bestimmt in nWire.py
szintipos = 300
def calc_winkel(min_file,max_file):
    for filenr in range(int(min_file),int(max_file)):
        try:
            #mache das folgende für die nicht leeren Entries
            if os.stat('root/entries/entry'+str(filenr)).st_size > 0:
                print("entry"+str(filenr)+" is "+str(os.stat('root/entries/entry'+str(filenr)).st_size)+" bytes gross")
                #importiere datensätze
                data = genfromtxt("root/entries/entry"+str(filenr), delimiter=';')
                wire = data[:, 0]
                time = data[:, 1]

                i_before=0
                j_before=0
                i_line=-1
                for i in wire:
                    i_line+=1
                    if i_before==i:
                        continue
                    i_before=i
                    j_line=-1
                    for j in wire:
                        j_line+=1
                        if j_before==j:
                            continue
                        j_before=j
                        if abs(i-j)%2==1 and abs(i-j) < 6:
                            #löse das gleichungssystem für drähte aus der kombination (gerade, ungerade) welche nicht weiter als 4 auseinander sind. Dies lässt sich geometisch ausschließen
                            out=str(solve_for_winkel(x_wire[int(i)],y_wire[int(i)],x_wire[int(j)],y_wire[int(j)],time[i_line],time[j_line]))
                            #modifiziere die Ausgabe von solve_for_winkel, so dass das diese im folgenden sinnvoll verwendet werden kann
                            out=out.replace("[{c1: ","")
                            out=out.replace(" la: ","")
                            out=out.replace(" lb: ","")
                            out=out.replace(" m2: ","")
                            out=out.replace("}, {c1: ","\n")
                            out=out.replace("}]","")
                            out=out.replace(",",";")
                            # print(out)

                            #ermittel Koordinaten und Radius von den beiden Drähten
                            a1=x_wire[int(i)]
                            a2=y_wire[int(i)]
                            b1=x_wire[int(j)]
                            b2=y_wire[int(j)]
                            ra=odb(time[i_line])
                            rb=odb(time[j_line])

                            color=colorarr[(int(filenr)+int(j+i+1))%49]
                            # print(color)
                            circlea=plt.Circle((a1,a2), ra, color=color, alpha=.33, fill=False)
                            circleb=plt.Circle((b1,b2), rb, color=color, alpha=.33, fill=False)
                            ax = plt.gca()
                            plt.plot(a1,a2,'x',color=color, alpha=.33)
                            plt.plot(b1,b2,'x',color=color, alpha=.33)
                            plt.gcf().gca().add_artist(circlea)
                            plt.gcf().gca().add_artist(circleb)

                            newdata=genfromtxt(StringIO(out), delimiter=';')
                            c1 = newdata[:,0]
                            m2 = newdata[:,3]

                            # print(len(c1))
                            # for i in [0,1,2,3]:
                            #     # print("i="+str(i))
                            #     y = g(x,m2[i],c1[i])
                            #     plt.plot(x,y,color=color, alpha=.33)

                            x = np.linspace(0, 408, 1000)
                            n=0 #anzahl der möglichen winkel für die gleichen parameter
                            #n_err=0
                            stringarr=['a','b','c','d']
                            #stringarr_err=['a','b','c','d']
                            for i in range(0,len(c1)):
                                szintillator = g_umkehr(132.5, m2[i],c1[i])
                                if szintipos-20 < szintillator < szintipos+20:
                                    stringarr[n]=str(filenr)+" m2=;"+str(m2[i])+";c1=;"+str(c1[i])
                                    y = g(x,m2[i],c1[i])
                                    plt.plot(x,y,color=color, alpha=.66)
                                    n+=1
                                # elif szintipos-40 < szintillator < szintipos-20 or szintipos+20 < szintillator < szintipos+40:
                                #     stringarr_err[n_err]=str(filenr)+" m2=;"+str(m2[i])+";c1=;"+str(c1[i])
                                #     y = g(x,m2[i],c1[i])
                                #     plt.plot(x,y,color=color, alpha=.33)
                                #     n_err+=1
                                else:
                                    y = g(x,m2[i],c1[i])
                                    plt.plot(x,y,color=color, alpha=.05)

                            m=";"+str(n)+"\n"
                            #m_err=";"+str(n_err+n)+"\n"
                            #Speichere die ermittelten parameter
                            while n != 0:
                                text_file = open(min_file+"-"+max_file+".txt", "a")
                                text_file.write(stringarr[n-1]+m)
                                text_file.close()
                                n=n-1

                            # while n_err != 0:
                            #     text_file = open(min_file+"-"+max_file+"_error.txt", "a")
                            #     text_file.write(stringarr_err[n_err-1]+m_err)
                            #     text_file.close()
                            #     n_err=n_err-1

            else:
                print("entry"+str(filenr)+" is empty")
        except:
            a=1

plt.figure(dpi=100, figsize=(10.2,4))
#führe calc_winkel (die funktion bestimmt keine winkel) für die ereignisse sys.argv[1] bis sys.argv[2] aus
calc_winkel(sys.argv[1],sys.argv[2])


#plotte den ganzen spaß
plt.plot([szintipos-20,szintipos+20],[132.5,132.5])
plt.xlim(0,408)
plt.ylim(-10,150)
plt.savefig("../pics/zusatz_geraden.pdf")
plt.show()
