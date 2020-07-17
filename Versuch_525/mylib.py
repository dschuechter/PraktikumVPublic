#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import odr

def func_gauss(p, x):
    a, b, c, d = p
    return a*np.exp(-(x-b)**2/(2*c))+np.sqrt(d*d)

def func_gauss_lin(p, x):
    a, b, c, d, e = p
    return a*np.exp(-(x-b)**2/(2*c))+d*x+np.sqrt(e*e)

def func_func(p,x):
    a, b, c = p
    # return b/(x-a)+c
    return np.exp(1)**(a*(x-b))+c

def func_lin(p,x):
    a,b=p
    return a*x+b

def anpassung_double_err(function, x, x_error, y, y_error, presets, plot,customlabel):
    model = odr.Model(function)
    data = odr.RealData(x, y, sx=x_error, sy=y_error)
    out = odr.ODR(data, model, beta0=presets).run()

    popt = out.beta
    perr = out.sd_beta

    if plot == True:
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = function(popt, x_fit)
        if customlabel == False:
            plt.plot(x_fit, y_fit)
        else:
            plt.plot(x_fit, y_fit,label=customlabel)

    return popt,perr

def anpassung_xerr(function, x, y, x_error, presets, plot,customlabel):
    model = odr.Model(function)
    data = odr.RealData(x, y, sx=x_error)
    out = odr.ODR(data, model, beta0=presets).run()

    popt = out.beta
    perr = out.sd_beta

    if plot == True:
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = function(popt, x_fit)
        if customlabel == False:
            plt.plot(x_fit, y_fit)
        else:
            plt.plot(x_fit, y_fit,label=customlabel)

    return popt,perr

def anpassung_yerr(function, x, y, y_error, presets, plot,customlabel):
    model = odr.Model(function)
    data = odr.RealData(x, y, sy=y_error)
    out = odr.ODR(data, model, beta0=presets).run()

    popt = out.beta
    perr = out.sd_beta

    if plot == True:
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = function(popt, x_fit)
        if customlabel == False:
            plt.plot(x_fit, y_fit)
        else:
            plt.plot(x_fit, y_fit,label=customlabel)

    return popt,perr

def anpassung_no_err(function, x, y, presets, plot,customlabel):
    model = odr.Model(function)
    data = odr.RealData(x, y)
    out = odr.ODR(data, model, beta0=presets).run()

    popt = out.beta
    perr = out.sd_beta

    if plot == True:
        x_fit = np.linspace(min(x), max(x), 10000)
        y_fit = function(popt, x_fit)

        if customlabel == False:
            plt.plot(x_fit, y_fit)
        else:
            plt.plot(x_fit, y_fit,label=customlabel)

    return popt,perr
