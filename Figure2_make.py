# This program investigates the bias in dust temperature produced by stacking.
# The program is based on the assumption that every galaxy has dust at a single
# temperature at one of two possible temperatures. The program then investigates
# how the difference between the temperature from stacking and the mass-weighted
# dust temperature depends on the fraction of galaxies at the two temperatures.

# Last edited: 14th October 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
# ***************************************************************************
# ***************************************************************************
# 
# Parameters of the dust model
# ***************************************************************************

Tcold = 20.0
Thot = 100.0
hot_proportion = np.arange(0.0,1.0,0.01)
beta = 2.0

# Redshift array

z = np.array([0.0,1.0,2.0,3.0])

# Parameters for the observations
# ****************************************************************************

# Wavelengths of observations

wave = np.array([70.0,160.0,250.0,350.0,500.0,850.0])

# Number of galaxies in stack

nstack = 10000

# signal-to-noise in stacks

ston = 50.0

# Parameters for the Monte Carlo simulation
# ****************************************************************************

nmonte = 10

# Other constants
# ****************************************************************************

# Constant in modified blackbody

con_bb = 48.01

# Function to predict flux densities
# ****************************************************************************
# ****************************************************************************

def create_model(wavelength,T):

# calculate frequency in units o 10^12 Hz

    freq = 3.0e2/wavelength
    tp1 = np.exp((con_bb * freq) / T)
    
    tp3 = 1.0 / (tp1 - 1.0)
    
    flux = tp3 * freq**(beta + 3.0)
    return flux

# Function used in curve_fit
# ****************************************************************************
# ****************************************************************************

def func(xdata,A,T):
    A1 = np.power(10.0,A)
    tp = np.exp( (con_bb * xdata) / T)
    tp1 = 1.0 / (tp - 1.0)
    flux_predict = A1 * xdata**(3.0+beta) * tp1
    return flux_predict

# Calculate the difference between the two temperatures for each value of
# the hot proportion
# ****************************************************************************
# ****************************************************************************

nlength = len(hot_proportion)
results1 = np.empty((nlength,11))
results2 = np.empty((nlength,11))
work = np.empty(nmonte)

for j in range(0,4):
    redshift = z[j]
    
    for k in range(0,nlength):
        print("model version: ",k)
        new_wavelengths = wave / (1.0 + redshift)
        
# calculate number of hot galaxies and number of cold galaxies
# ****************************************************************************

        Nhot = int(hot_proportion[k] * nstack)
        Ncold = nstack - Nhot
        
# calculate the errors given the signal-to-noise in the stack
# ****************************************************************************
        
# calculate mean flux at each wavelength

        tp1 = create_model(new_wavelengths,Tcold)
        tp2 = create_model(new_wavelengths,Thot)
        
        flux_tot = float(Nhot) * tp2 + float(Ncold) * tp1
        flux_tot = flux_tot/float(nstack)
        errors_stack = flux_tot / ston
        errors = errors_stack * np.sqrt(float(nstack))
        
# Start of Monte Carlo generation of samples of galaxies
# ****************************************************************************
# ****************************************************************************
        
        for m in range(0,nmonte):
            
# create fluxes for the hot and cold galaxies

            flux_hot = np.empty((Nhot,6))
            flux_cold = np.empty((Ncold,6))
        
            for i in range(0,Nhot):
                flux_hot[i,:] = create_model(new_wavelengths,Thot) + \
                errors * np.random.normal(loc=0.0,scale=1.0,size=6)
                
            for i in range(0,Ncold):
                flux_cold[i,:] = create_model(new_wavelengths,Tcold) + \
                errors * np.random.normal(loc=0.0,scale=1.0,size=6)

# Carry out analysis of the simulated galaxies
# ****************************************************************************

# First, calculate mass-weighted dust temperature

            T_mw = Thot * hot_proportion[k] + Tcold * (1.0 - hot_proportion[k])
            results1[k,0] = T_mw
        
# Now calculate the mean flux at each wavelength and fit an SED
# ****************************************************************************       
        
            fluxes = np.empty(6)
        
            for i in range(0,6):
                tp1 = np.sum(flux_cold[:,i]) + np.sum(flux_hot[:,i])
                tp1 = tp1 / float(nstack)
                fluxes[i] = tp1

# Fitting the model to the data

# Selecting possible ranges for the parameters of the fit

# estimating a rough normalisation for the model

            freq = 3.0e2/new_wavelengths

            freq_ref = freq[3]
            tp = np.exp( (con_bb * freq_ref) / 25.0)
            tp2 = 1.0 / (tp - 1.0)
            tp3 = tp2 * freq_ref**(3.0+beta)

            Aguess = fluxes[3] / tp3

            Aguess = np.log10(Aguess)

            Rnorm_min = Aguess - 4.0
            Rnorm_max = Aguess + 4.0

            T_min = 10.0
            T_max = 150.0

            Tguess = 25.0

            popt,pcov = curve_fit(func,freq,fluxes,p0=(Aguess,Tguess),\
                                  sigma=errors_stack,method='trf',\
                          bounds=([Rnorm_min,T_min],\
                                  [Rnorm_max,T_max]),absolute_sigma=True)
            norm_pred = popt[0]
            T_pred = popt[1]
            
            work[m] = T_pred
            
            
# calculate the median of the results

        results1[k,j+1] = np.median(work)
        
# Now plot out results
# *****************************************************************************

T_mw = results1[:,0]

xline = hot_proportion
y_mw = results1[:,0]
yline = results1[:,1]
yline1 = results1[:,2]
yline2 = results1[:,3]
yline3 = results1[:,4]

fig = plt.figure(figsize=(10.0,10.0))
f1 = plt.axes([0.15,0.15,0.6,0.6])
f1.set_xlim(0.01,0.99)
f1.set_ylim(20.0,110.0)
f1.set_xlabel('$N_{hot}/N_{total}$',size=20)
f1.set_ylabel('$T_{dust}/K$',size=20)

f1.tick_params(axis='both',labelsize=20)

f1.plot(xline,yline,'g-')
f1.plot(xline,yline1,'r-')
f1.plot(xline,yline2,'b-')
f1.plot(xline,yline3,'y-')
f1.plot(xline,y_mw,'k-')

# Plot key

xl = np.empty(2)
yl = np.empty(2)

xl[0] = 0.6
xl[1] = 0.7
yl[0] = 30.0
yl[1] = 30.0

f1.plot(xl,yl,'g-')

yl[0] = 35.0
yl[1] = 35.0
f1.plot(xl,yl,'r-')

yl[0] = 40.0
yl[1] = 40.0
f1.plot(xl,yl,'b-')

yl[0] = 45.0
yl[1] = 45.0
f1.plot(xl,yl,'y-')

pos_ann = ([0.72,28.0])
f1.annotate('z=0',pos_ann,size=15)

pos_ann = ([0.72,33.0])
f1.annotate('z=1',pos_ann,size=15)

pos_ann = ([0.72,38.0])
f1.annotate('z=2',pos_ann,size=15)

pos_ann = ([0.72,43.0])
f1.annotate('z=3',pos_ann,size=15)


fig.savefig('Figure_stacking_method_Temperature_bias.pdf')

plt.show()





