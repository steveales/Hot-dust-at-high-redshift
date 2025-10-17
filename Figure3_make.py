# This program investigates the bias in dust temperature produced by stacking.
# The program is based on the assumption that every galaxy has dust at a single
# temperature at one of two possible temperatures. The program then investigates
# how the difference between the temperature from stacking and the mass-weighted
# dust temperature depends on the fraction of galaxies at the two temperatures.

# Last edited: 5th September 2025

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
hot_proportion = 0.01
beta = 2.0

redshift = 2.0

# Parameters for the observations
# ****************************************************************************

# Wavelengths of observations

wave = np.array([70.0,160.0,250.0,350.0,500.0,850.0])

flux_error = 0.1

# Other constants
# ****************************************************************************

# Constant in modified blackbody

con_bb = 48.01

# Function to predict flux densities
# ****************************************************************************
# ****************************************************************************

def create_model(wavelength,Tcold,Thot,hot_proportion):

# calculate frequency in units o 10^12 Hz

    freq = 3.0e2/wavelength
    tp1 = np.exp((con_bb * freq) / Tcold)
    tp2 = np.exp((con_bb * freq) / Thot)
    
    tp3 = (1.0 - hot_proportion) / (tp1 - 1.0) + hot_proportion /(tp2-1.0)
    
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
    
# First, calculate mass-weighted dust temperature

T_mw = Thot * hot_proportion + Tcold * (1.0 - hot_proportion)

print("Mass-weighted dust temperature: ",T_mw)

# Construct model
# ****************************************************************************

# Calculate fluxes for model with six flux measurements

fluxes = np.zeros(6)
errors = np.zeros(6)
    
new_wavelengths = wave / (1.0 + redshift)
    
for i in range(0,6):
    fluxes[i] = create_model(new_wavelengths[i]\
                                 ,Tcold,Thot,hot_proportion)
    
# normalise flux to 1 Jy at 350 microns

rnorm = 1.0 / fluxes[3]

fluxes = rnorm * fluxes

errors = flux_error * fluxes

for i in range(0,6):    
    print(new_wavelengths[i],fluxes[i])

# Calculate relationship between frequency and wavelength for model
# ****************************************************************************

wave_model = np.arange(30.0,1000.0,0.1)
new_wave_model = wave_model / (1.0 + redshift)

model = create_model(new_wave_model,Tcold,Thot,hot_proportion)
model = model*rnorm

# calculate curves for the hot galaxy and the cold galaxies

model_hot = create_model(new_wave_model,Tcold,Thot,1.0)
model_hot = hot_proportion * model_hot
model_hot = model_hot * rnorm

model_cold = create_model(new_wave_model,Tcold,Thot,0.0)
model_cold = (1.0-hot_proportion) * model_cold
model_cold = model_cold * rnorm

# Fitting the model to the data
# ****************************************************************************

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

popt,pcov = curve_fit(func,freq,fluxes,p0=(Aguess,Tguess),sigma=errors,method='trf',\
                          bounds=([Rnorm_min,T_min],\
                                  [Rnorm_max,T_max]),absolute_sigma=True)
rnorm_pred = popt[0]
T_pred = popt[1]


print("Luminosity-weighted dust temperature: ",T_pred)

# Now plot out results
# *****************************************************************************

# Calculate best-fit line

freq = 3.0e2/new_wave_model

new_model = func(freq,rnorm_pred,T_pred)

fig = plt.figure(figsize=(10.0,10.0))
f1 = plt.axes([0.15,0.15,0.65,0.65])
f1.set_xlim(30.0,1000.0)
f1.set_ylim(0.01,50.0)
f1.set_xlabel('Wavelength/microns',size=20)
f1.set_ylabel('Flux/Jy',size=20)

f1.tick_params(axis='both',labelsize=20)

f1.plot(wave,fluxes,'ko')
f1.errorbar(wave,fluxes,yerr=errors,fmt='none',ecolor='k')

f1.plot(wave_model,model,'k-')
f1.plot(wave_model,model_hot,'r-')
f1.plot(wave_model,model_cold,'b-')

f1.plot(wave_model,new_model,'k--')
f1.set_yscale('log')
f1.set_xscale('log')

fig.savefig('Figure_Example_SED_z=2_prop=0.01.pdf')

plt.show()




