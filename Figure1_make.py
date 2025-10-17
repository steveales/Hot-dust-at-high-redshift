# This program examines how the difference between the peak temperature 
# and mass-weighted temperatures depend on the mass-ratio for a two-component
# dust model. 

# Last edited: 3rd October 2025

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
Thot = 50.0
mass_ratio = np.arange(0.0,1.0,0.01)
beta = 2.0

# Details of Monte Carlo
# ****************************************************************************

nmonte = 1000

# Parameters for the observations
# ****************************************************************************

# Wavelengths of observations

wave = np.array([70.0,160.0,250.0,350.0,500.0,850.0])
wave2 = np.array([160.0,250.0,350.0,500.0,850.0])
wave3 = np.array([250.0,350.0,500.0,850.0])

# Fractional error in flux densities

flux_error = 0.1

# Other constants
# ****************************************************************************

# Constant in modified blackbody

con_bb = 48.01

# Function to predict flux densities
# ****************************************************************************
# ****************************************************************************

def create_model(wavelength,Tcold,Thot,mass_ratio):

# calculate frequency in units of 10^12 Hz

    freq = 3.0e2/wavelength
    tp1 = np.exp((con_bb * freq) / Tcold)
    tp2 = np.exp((con_bb * freq) / Thot)
    
    tp3 = mass_ratio / (tp2 - 1.0) + (1.0 - mass_ratio) /(tp1-1.0)
    
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
# the mass ratio
# ****************************************************************************
# ****************************************************************************

nlength = len(mass_ratio)
results = np.empty((nlength,10))
work = np.empty(nmonte)
flux_monte = np.empty(6)

for k in range(0,nlength):
    
# First, calculate mass-weighted dust temperature

    T_mw = (Thot * mass_ratio[k] + (1.0 - mass_ratio[k]) * Tcold)
    results[k,0] = T_mw
    print(T_mw)

# Construct model
# ****************************************************************************
# ****************************************************************************

# Create six flux measurements
# ****************************************************************************

    fluxes = np.zeros(6)
    errors = np.zeros(6)
    

    for i in range(0,6):
        fluxes[i] = create_model(wave[i],Tcold,Thot,mass_ratio[k])
    
# normalise flux to 1 Jy at 350 microns

    rnorm = 1.0 / fluxes[3]

    fluxes = rnorm * fluxes

    errors = flux_error * fluxes
    
# Calculate relationship between frequency and wavelength for model
# ****************************************************************************

    wave_model = np.arange(30.0,1000.0,0.1)

    model = create_model(wave_model,Tcold,Thot,mass_ratio[k])
    model = model*rnorm
    
# Now carry out Monte Carlo
# ****************************************************************************

    work = np.empty(nmonte)
    
    for i in range(0,nmonte):
        flux_monte = fluxes + errors * np.random.normal(loc=0.0,scale=1.0,size=6)
        
# Fitting the model to the data
# ****************************************************************************

# Selecting possible ranges for the parameters of the fit

# estimating a rough normalisation for the model

        freq = 3.0e2/wave

        freq_ref = freq[3]
        tp = np.exp( (con_bb * freq_ref) / 25.0)
        tp2 = 1.0 / (tp - 1.0)
        tp3 = tp2 * freq_ref**(3.0+beta)

        Aguess = flux_monte[3] / tp3

        Aguess = np.log10(Aguess)

        Rnorm_min = Aguess - 4.0
        Rnorm_max = Aguess + 4.0

        T_min = 10.0
        T_max = 80.0

        Tguess = 25.0

        popt,pcov = curve_fit(func,freq,flux_monte,p0=(Aguess,Tguess),sigma=errors,method='trf',\
                          bounds=([Rnorm_min,T_min],\
                                  [Rnorm_max,T_max]),absolute_sigma=True)
        rnorm_pred = popt[0]
        T_pred = popt[1]
        
        work[i] = T_pred

    results[k,1] = np.median(work)
    
# Now calculate Tpeak, using definition from Liang et al. (2019)
# *****************************************************************************

    T_pred = results[k,1]
    
    freq_for_peak = 3.0e2/wave_model
    A = 1.0
    
    model_peak = func(freq_for_peak,A,T_pred)
    
    select = np.where(model_peak==np.max(model_peak))[0]
    
    wave_peak = wave_model[select]
    
    Tpeak = 2.9e3/wave_peak[0]
    results[k,4] = Tpeak
    
# Now do the same for when there are only five flux measurements
# ****************************************************************************

    fluxes2 = np.empty(5)
    errors2 = np.empty(5)
    freq2 = 3.0e2/wave2
    
    for i in range(0,5):
        fluxes2[i] = fluxes[i+1]
    
    errors2 = fluxes2 * flux_error
    
# Now carry out Monte Carlo
# ****************************************************************************

    work = np.empty(nmonte)
    
    for i in range(0,nmonte):
        flux_monte = fluxes2\
            + errors2 * np.random.normal(loc=0.0,scale=1.0,size=5)
        
# Fitting the model to the data

# Selecting possible ranges for the parameters of the fit

# estimating a rough normalisation for the model

        freq = 3.0e2/wave

        freq_ref = freq[3]
        tp = np.exp( (con_bb * freq_ref) / 25.0)
        tp2 = 1.0 / (tp - 1.0)
        tp3 = tp2 * freq_ref**(3.0+beta)

        Aguess = flux_monte[2] / tp3

        Aguess = np.log10(Aguess)

        Rnorm_min = Aguess - 4.0
        Rnorm_max = Aguess + 4.0

        T_min = 10.0
        T_max = 80.0

        Tguess = 25.0

        popt,pcov = curve_fit(func,freq2,flux_monte,p0=(Aguess,Tguess),sigma=errors2,method='trf',\
                          bounds=([Rnorm_min,T_min],\
                                  [Rnorm_max,T_max]),absolute_sigma=True)
        rnorm_pred = popt[0]
        T_pred = popt[1]
        
        work[i] = T_pred

    results[k,2] = np.median(work)  
    
# Now calculate Tpeak, using definition from Liang et al. (2019)
# *****************************************************************************
    
    T_pred = results[k,2]
    
    model_peak = func(freq_for_peak,A,T_pred)
    
    select = np.where(model_peak==np.max(model_peak))[0]
    
    wave_peak = wave_model[select]
    
    Tpeak = 2.9e3/wave_peak[0]
    results[k,5] = Tpeak

# Now do the same for when there are only four flux measurements
# ****************************************************************************

    fluxes3 = np.empty(4)
    freq3 = 3.0e2/wave3
    errors3 = np.empty(4)
    
    for i in range(0,4):
        fluxes3[i] = fluxes[i+2]
        
    errors3 = fluxes3 * flux_error
    
# Now carry out Monte Carlo
# ****************************************************************************

    work = np.empty(nmonte)
    
    for i in range(0,nmonte):
        flux_monte = fluxes3\
            + errors3 * np.random.normal(loc=0.0,scale=1.0,size=4)    
    
# Fitting the model to the data

# Selecting possible ranges for the parameters of the fit

# estimating a rough normalisation for the model

        freq = 3.0e2/wave

        freq_ref = freq[3]
        tp = np.exp( (con_bb * freq_ref) / 25.0)
        tp2 = 1.0 / (tp - 1.0)
        tp3 = tp2 * freq_ref**(3.0+beta)

        Aguess = flux_monte[1] / tp3

        Aguess = np.log10(Aguess)

        Rnorm_min = Aguess - 4.0
        Rnorm_max = Aguess + 4.0

        T_min = 10.0
        T_max = 80.0

        Tguess = 25.0

        popt,pcov = curve_fit(func,freq3,flux_monte,p0=(Aguess,Tguess),sigma=errors3,method='trf',\
                          bounds=([Rnorm_min,T_min],\
                                  [Rnorm_max,T_max]),absolute_sigma=True)
        rnorm_pred = popt[0]
        T_pred = popt[1]
        
        work[i] = T_pred

    results[k,3] = np.median(work)

    
# Now calculate Tpeak, using definition from Liang et al. (2019)
# *****************************************************************************

    freq_for_peak = 3.0e2/wave_model
    A = 1.0
    
    T_pred = results[k,3]
    
    model_peak = func(freq_for_peak,A,T_pred)
    
    select = np.where(model_peak==np.max(model_peak))[0]
    
    wave_peak = wave_model[select]
    
    Tpeak = 2.9e3/wave_peak
    results[k,6] = Tpeak[0]
       
for i in range(0,nlength):
    print(mass_ratio[i],results[i,0],results[i,4],results[i,5],results[i,6])
    
# Now plot out results
# *****************************************************************************

T_mw = results[:,0]
T_lw = results[:,4]
T_lw_2 = results[:,5]
T_lw_3 = results[:,6]

xline = mass_ratio
yline = T_lw
yline2 = T_lw_2
yline3 = T_lw_3

fig = plt.figure(figsize=(10.0,10.0))
f1 = plt.axes([0.15,0.15,0.6,0.6])
f1.set_xlim(0.0,1.0)
f1.set_ylim(20.0,50.0)
f1.tick_params(axis='both',labelsize=20)
f1.set_xlabel('$M_{hot}/M_{total}$',size=20)
f1.set_ylabel('$T_{dust}$/K',size=20)

f1.plot(xline,yline,'r-')
f1.plot(xline,yline2,'b-.')
f1.plot(xline,yline3,'--')
f1.plot(xline,T_mw,'k-')


fig.savefig('Figure_Temperature_bias.pdf')
plt.show()





