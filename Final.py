import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import Rheosys as rhs
import pandas as pd



"""Set transfer function parameters"""
Lambda_constant=1.5                        # Value for lambda n/g
g_constant=2.5                             # Value for spring constant g

f_s = 20                                   # Sample frequency
N = 2000                                   # Number of points
P = 10                                      # Number of repeated periods
P_tf = 0                                   # Number of removed transient periods

M = 10

plot_signal='Multisine'                    # Select 'Chirp' or 'Multsine' to change plots

multisine_param     = {'phase'  : 'Random',     # Select phase for the multisine, (Schroeder,Random,Linear,Zero,Rudin,Newman)
                        'j1'    : 1,            # Starting frequency multisine
                        'j2'    : N/2,}         # Stopping freqency multisine

chirp_param         = { 'j1'    : 1,            # Starting frequency chirp
                        'j2'    : N/2,}         # Stopping frequency chirp

interpolation_param = { 'set'   : True,         # Set to True to activate interpolation of raw created signal
                        'kind'  : 0}            # Set kind to 0 for Zero Order Hold, optional every odd power number including 2

normalization_param = { 'set'   : True,         # Set to True to activate normalization
                        'method': 'STD',      # Select normalization methode (None, STDev, Amplitude, RMS)
                        'max'   : 1,            # Optional to multiply signal with a value to set amplitude hight, default =1
                        'x_norm': 0,}           # Equal normalization of previous dignal, x_norm is 'method' value of previous normalizaiton, 0 is default set 'method'

noise_param     = { 'inputset'  : False,         # Set the value for noise on input to true or false and include SNR in decibels
                    'inputdb'   : 40,
                    'outputset'  : True,         # Set the value for noise on output to true or false and include SNR in decibels
                    'outputdb'   : 40}

window_param        = { 'set'   : False,        # Set to True to activate windwo
                        'r'     : 0.15}         # Set value of R between 0-1, 0 is rectangular, 1 is Hann window

optimizecrest_param  = {'set'   : False,       # Set to True to optimize the signal the crest clipping algorithm
                       'clip'   : 0.9,          # Set clipping value which clips the crest value
                       'R'      : 10,          # Amount of realization to test the optimized signal u
                      'variable': False,}       # Varible clipping value, True to linearly change 'clip' value from set value to 1

multisine_log_param = {'set'    : False,        # Optional Logarithmic amplitude vector A in multisin
                       'decade' : 40}           # Amount of frequencies per decade, j1 and j2 are defined in 'multisine_params'

LPM_param           = {'set'    : True,        # Set to True To calculate the LocalPolynomialMeasurement set it to True
                   'order'      : 2,            # method.order[2] order of approximation
                   'dof'        : 1,            # method.dof degrees of freedom, independent experiments default = 1
                   'transient'  : 1}            # method.transient, include transien estimation = 1, exclude transient estimation = 0


""" Calculation of Time window"""
check = rhs.check_variables(f_s,N,P,P_tf,str(window_param['set']))

f_0   = f_s/N                                         # Excitation frequency
T     = N/f_s                                         # Time length
t     = np.linspace(0, T,N,endpoint=False)            # Time vector
f     = np.linspace(0,f_s,N,endpoint=False)           # Frequency range
N_tf  = int(P_tf*N)                                   # Number of transient points
t_transient= np.linspace(0, T*(P+P_tf),(N*P+N_tf),endpoint=False)   # Transient-time vector

### Transfer function of the maxwell system ###
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])

# Convert transfer function to magnitude and phase of FRF
w, mag, phase = signal.bode(s1,f*2*np.pi)
w_tf=w/(2*np.pi)                            # Frequency of transfer function in Hz
phase_tf= np.deg2rad(phase)                 # Fase of transfer function in radians
mag_tf=10**(mag/20)                         # Magnitude of transfer function [-]
G_0=mag_tf*np.exp(1j*(phase_tf))            # FRF from defined transfer function

# analytical transient curve
transient_analytic = np.exp(-t_transient[:(N)]/Lambda_constant)


"""
CALCULATION FOR G_0
The first steady state periode of the total signal is used to calculate G_0
"""


# Creation of the multisine signal, included frequency vector logarithmic or standard
if multisine_log_param['set']:
    A_multi,f_log=rhs.log_amplitude(N,multisine_param['j1'],multisine_param['j2'],multisine_log_param['decade'])
    u_mutlisine, phase_multisine=rhs.multisine(f_s,N, [multisine_param['j1'],multisine_param['j2']],phase_response=multisine_param['phase'],A_vect=A_multi)
    t=t[f_log]
else:
    A_multi=np.ones(N)
    u_mutlisine, phase_multisine=rhs.multisine(f_s,N, [multisine_param['j1'],multisine_param['j2']],phase_response=multisine_param['phase'],A_vect=A_multi)

# Calculation of the input signal for the multisine signal over the entire transient time period
u_chirp=rhs.chirp_exponential(f_s,N,[chirp_param['j1'],chirp_param['j2']])

#Select u_chirp or u_multi as tranient signal for figure generation
if plot_signal=='Multisine':
    u_select   = u_mutlisine
    band_range = [multisine_param['j1'],multisine_param['j2']]
elif plot_signal=='Chirp':
    u_select = u_chirp
    band_range = [chirp_param['j1'],chirp_param['j2']]

# Crest optimization algorithm is activated if optimizecrest_param is set to true, use of clipping
if optimizecrest_param['set']:
    u_select=rhs.crest_optimization(u_select,optimizecrest_param['clip'],optimizecrest_param['R'],1,optimizecrest_param['variable'])


excitedharm_range = np.linspace(band_range[0],band_range[1],int(band_range[1]-band_range[0])+1,endpoint=True)
f_range = excitedharm_range*f_0
u_transient= np.tile(u_select, P)
if P_tf!=0:
    u_transient = np.insert(u_transient,0,u_transient[-N_tf:])

# Save the set transient basic signal and apply further normalization an interpolation
u_0 = u_select
u_0_transient=u_transient

# Calculations of the multiphase crest factor
fact_crest = rhs.crest_fac(u_0)
fact_effic = rhs.power_efficiency(fft(u_0),band_range)
fact_loss  = rhs.power_loss(fft(u_0),band_range)
fact_quali = rhs.quality_factor(fact_crest,fact_effic,fact_loss)

print('Crest factor: {0:.2f} '.format(fact_crest))
print('Power efficiency: {0:.2f} '.format(fact_effic))
print('Power loss: {0:.2f} '.format(fact_loss))
print('Signal quality: {0:.2f} '.format(fact_quali))

# Apply the window function individually to each period,
if window_param['set']:
    window = rhs.window_function(t, T, window_param['r'])
    window_transient = np.tile(window, P+P_tf)
    u_0_transient = u_transient * window_transient
else:
    u_0_transient = u_transient

# Sampled transient signal as a function of t
if interpolation_param['set']:
    u_0_transient_t=interp1d(t_transient,u_0_transient,kind=interpolation_param['kind'])
    u_0_transient=u_0_transient_t(t_transient)

# Normalize the input signal to None, STDev, Amplitude or RMS
if normalization_param['set']:
    u_0_transient=rhs.normalization(u_0_transient,normalization_param['method'],normalization_param['max'])


# Initialize lists to collect data
U_M = []
Y_M = []
G_etfe_p=[]
G_etfe_m=[]
G_ML_m=[]
G_LPM_m =[]

for m in range(M):
    """
    CALCULATIONS OF NOISE
    Starting with the addition of measurement noise on either inpot or output
    Noise on input is not transfer  trough the transfer function in calculation of the output
"""
    # Apply measurement noise to the input
    if noise_param['inputset']:
        n_u=rhs.add_noise_with_snr(u_0_transient,noise_param['inputdb'])
        u_n_transient=u_0_transient + n_u
    else:
        u_n_transient=u_0_transient


    # Compute the response output y by
    t_out,y_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)

    # Application of measurement noise on the output
    if noise_param['outputset']:
        n_y = rhs.add_noise_with_snr(y_out,noise_param['outputdb'])
        y_n_transient= y_out + n_y
        y_0_transient=np.squeeze(y_out)          # y output in time domain entire transient signal
    else:
        y_0_transient=np.squeeze(y_out)          # y output in time domain entire transient signal
        y_n_transient=y_0_transient

    """
    CALCULATION OF G^ IN MULTIPLE PERIODS
    Depending on the amount of selected transfer free periods
    The overall Frequency responce is calculated

    """
    u_p=[]
    y_p=[]
    U_p=[]
    Y_p=[]


    for p in range(P):

        # Calculate indices for extracting one period
        start_idx = (N *   p)+ N_tf
        end_idx = start_idx + N

        if p==0:
            y_n_steady = y_n_transient[start_idx:end_idx]
        else:
            y_n_steady = y_n_transient[(N *   p)+ N_tf:]
        # Extract one period from the transient signals
        y_p_period = y_n_transient[start_idx:end_idx]
        u_p_period = u_n_transient[start_idx:end_idx]

        u_p.append(u_p_period)
        y_p.append(y_p_period)

        # Calculate FFT of the extracted period
        Y_p_period = fft(y_p_period)
        U_p_period = fft(u_p_period)

        # Append FFT results to the lists
        Y_p.append(Y_p_period[:int(band_range[1])])
        U_p.append(U_p_period[:int(band_range[1])])

        # Calculate FRF for each period and append to G_n
        G_etfe_p.append(Y_p_period/U_p_period)
    # Defined the steady state measurement
    y_p_steady = y_n_transient[N_tf:]
    u_p_steady = u_n_transient[N_tf:]

    # Calculate FFT of the steady state
    Y_p_steady = fft(y_p_steady)
    U_p_steady = fft(u_p_steady)

    U_M.append(U_p)
    Y_M.append(Y_p)
    G_ML_m.append(np.sum(Y_p,axis=0)/np.sum(U_p,axis=0))
    G_etfe_m.append(np.mean(G_etfe_p,axis=0))
    if LPM_param['set']:
        G_LPM_m.append(rhs.matpy_LPM(u_p_steady,y_p_steady,u_select,N,f_s,excitedharm_range,LPM_param['order'],LPM_param['dof'],LPM_param['transient']))

U_M=np.array(U_M)
Y_M=np.array(Y_M)
G_etfe=np.array(np.mean(G_etfe_m,axis=0))
G_ML=np.array(np.mean(G_ML_m,axis=0))
G_LPM=np.array(np.squeeze(np.mean(G_LPM_m,axis=0)))


"""
Calculations have been made, to export the right
"""
f_range=f_range[1:]
G_0=G_0[1:int(band_range[1])]

if LPM_param['set']:
    G_export=G_LPM[:int(band_range[1])-1]
    G_export_m=G_LPM_m
else:
    G_export=G_ML[1:int(band_range[1])]
    G_export_m=G_ML_m


# Save specified signal
#rhs.export_data_for_visualization('data_visualization_test.csv',G_export, G_export_m,  f_range, G_0, band_range, N, P, u_p_steady,t, noise_param)

# Example usage:
data_path = 'multi_random_lpm'
plot_title=rhs.create_plot_title(f_s, N, P, M, plot_signal, multisine_param, chirp_param, interpolation_param, normalization_param, noise_param, window_param, optimizecrest_param, multisine_log_param, LPM_param)
rhs.export_data_for_visualization_pickle(data_path, G_export, G_export_m,  f_range, G_0, band_range, N, P, M, u_p_steady,t, noise_param, plot_title)

break
# FRF calculation by Maximum Likelihood
var_G_ML=np.var(G_ML_m,axis=0)
var_G_etfe=np.var(G_etfe_p,axis=0)

#matlab_transfer=rhs.matlab_input_file('Matlab_input_randmulti_nonoise_20.mat',u_0_transient,y_0_transient,u_select,N,1/f_s,excitedharm_range)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_randmulti_noise_20.mat',u_n_transient,y_n_transient,u_select,N,1/f_s,excitedharm_range)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_randmulti_test.mat',u_p_steady,y_p_steady,u_select,N,1/f_s,excitedharm_range)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_multischroederlpm_test.mat',u_p_steady,y_p_steady,u_select,N,1/f_s,excitedharm_range,G_LPM)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_multischroeder_test.mat',u_p_steady,y_p_steady,u_select,N,1/f_s,excitedharm_range,G_ML)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_multisineYU.mat',U_M,Y_M,f,N,1/f_s,f_range)




#f_range=f_range[1:]
G_0=G_0[:int(band_range[1])]
G_1_multi_trans=G_ML
#G_1_multi_trans=G_ML[1:int(band_range[1])]
#G_2_multi_window=G_ML[:int(band_range[1])]

G_3_multi_LPM=G_LPM
#G_3_multi_LPM=G_LPM[:int(band_range[1])-1]
#G_4_chirp_window=G_ML
#G_5_chirp_LPM=G_LPM[:int(band_range[1])]
#G_6_rand_LPM=G_LPM
#G_7_randclip_LPM=G_LPM[:int(band_range[1])]


var_G_1_multi_trans=np.var(G_ML_m[:int(band_range[1])],axis=0)
#var_G_2_multi_window=np.var(G_ML_m[:int(band_range[1])],axis=0)
var_G_3_multi_LPM=np.var(G_LPM_m[:int(band_range[1])],axis=0)
#var_G_4_chirp_window=np.var(G_ML_m,axis=0)
#var_G_5_chirp_LPM=np.var(G_LPM_m[:int(band_range[1])],axis=0)
#var_G_6_rand_LPM=np.var(G_LPM_m,axis=0)
#var_G_7_randclip_LPM=np.var(G_LPM_m[:int(band_range[1])],axis=0)

dif_G_1_multi_trans=G_0-G_1_multi_trans
#dif_G_2_multi_window=G_ML[:int(band_range[1])]-G_0[:int(band_range[1])]
dif_G_3_multi_LPM=G_0[1:]-G_3_multi_LPM[:-1]
#dif_G_4_chirp_window=G_ML-G_0[:int(band_range[1])]
#dif_G_5_chirp_LPM=G_LPM[:int(band_range[1])]-G_0[:int(band_range[1])]
#dif_G_6_rand_LPM=G_LPM-G_0[:int(band_range[1])]
#dif_G_7_randclip_LPM=G_LPM[:int(band_range[1])]-G_0[:int(band_range[1])]


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(G_0),'-',label='$\hat{G}_{0}$')
plt.plot(f_range,rhs.DB(G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
plt.plot(f_range,rhs.DB(var_G_1_multi_trans),'-',label='$\hat{G}_{1} Var Multisine schroeder transient$')
plt.plot(f_range,rhs.DB(dif_G_1_multi_trans),'o',label='$\hat{G}_{1} Bias Multisine schroeder transient$')
plt.plot(f_range,rhs.DB(G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range[1:],rhs.DB(dif_G_3_multi_LPM),'o',label='$\hat{G}_{3}Bias Multisine schroeder LPM$')
plt.plot(f_range,np.log10(abs(var_G_3_multi_LPM)),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range[1:],rhs.DB(dif_G_3_multi_LPM),'o',label='$\hat{G}_{3}Bias Multisine schroeder LPM$')
plt.plot(f_range,rhs.DB(G_ML[:int(band_range[1])]-G_etfe[:int(band_range[1])]),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend(loc='best')
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

break

# Phase plot of the FRF transferfunction with multisine
plt.plot(f_log,rhs.DB(G_0[f_log-1]),'-',label='$\hat{G}_{0}$')
plt.plot(f_log,rhs.DB(dif_G_3_multi_LPM[f_log-1]),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_log,rhs.DB(dif_G_1_multi_trans[f_log-1]),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
plt.plot(f_log,rhs.DB(G_3_multi_LPM[f_log-1]),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_log,rhs.DB(G_1_multi_trans[f_log-1]),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend(loc='best')
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

break





"""
PLOTTING FOR VISUALIZATION OF THE FRF RESULTS

"""


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(G_0[:int(band_range[1])]),'-',label='$\hat{G}_{0}$')
#plt.plot(f_range,rhs.DB(G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
#plt.plot(f_range,rhs.DB(G_2_multi_window),'o',label='$\hat{G}_{2}Multisine schroeder windowing$')
#plt.plot(f_range,rhs.DB(G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range,rhs.DB(G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
#plt.plot(f_range,rhs.DB(G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')
plt.plot(f_range,rhs.DB(G_6_rand_LPM),'o',label='$\hat{G}_{6}Multisine random LPM$')
plt.plot(f_range,rhs.DB(G_7_randclip_LPM),'o',label='$\hat{G}_{7}Multisine random clip LPM$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

"""
PLOTTING THE BIAS OF EACH SIGNAL

"""

# Phase plot of the FRF transferfunction with multisine
#plt.plot(f_range,rhs.DB(dif_G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
#plt.plot(f_range,rhs.DB(dif_G_2_multi_window),'o',label='$\hat{G}_{2}Multisine windowing$')
#plt.plot(f_range,rhs.DB(dif_G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range,rhs.DB(dif_G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
#plt.plot(f_range,rhs.DB(dif_G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')
plt.plot(f_range,rhs.DB(dif_G_6_rand_LPM),'o',label='$\hat{G}_{6}Multisine random LPM$')
plt.plot(f_range,rhs.DB(dif_G_7_randclip_LPM),'o',label='$\hat{G}_{7}Multisine random clip LPM$')
plt.title(f'Bias of the FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()


"""
PLOTTING THE VARIANCE OF EACHT SIGNAL

"""
break
# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,np.log10(abs(var_G_1_multi_trans)),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
plt.plot(f_range,np.log10(abs(var_G_2_multi_window)),'o',label='$\hat{G}_{2}Multisine schroeder windowing$')
plt.plot(f_range,np.log10(abs(var_G_3_multi_LPM)),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range,np.log10(abs(var_G_4_chirp_window)),'o',label='$\hat{G}_{4}Chirp windowing$')
plt.plot(f_range,np.log10(abs(var_G_5_chirp_LPM)),'o',label='$\hat{G}_{5}Chirp LPM$')
plt.plot(f_range,np.log10(abs(var_G_6_rand_LPM)),'o',label='$\hat{G}_{6}Multisine random LPM$')
plt.plot(f_range,np.log10(abs(var_G_7_randclip_LPM)),'o',label='$\hat{G}_{7}Multisine random clip LPM$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(dif_G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine schroeder transient$')
#plt.plot(f_range,rhs.DB(dif_G_2_multi_window),'o',label='$\hat{G}_{2}Multisine windowing$')
plt.plot(f_range,rhs.DB(dif_G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine schroeder LPM$')
plt.plot(f_range,rhs.DB(dif_G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
#plt.plot(f_range,rhs.DB(dif_G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')
plt.plot(f_range,rhs.DB(dif_G_6_rand_LPM),'o',label='$\hat{G}_{6}Multisine random LPM$')
plt.plot(f_range,rhs.DB(dif_G_7_randclip_LPM),'o',label='$\hat{G}_{7}Multisine random clip LPM$')
plt.title(f'Bias of the FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

break
"""
PLOTTING THE BIAS AND VARIANCE TRADE OF IN ONE PLOT
"""

# Assuming you've defined the necessary variables and functions elsewhere
# f_range, M, P, N, noise_input_u, noise_output_y, f_s, etc.

# Placeholder for your actual data
signals = [
    {'name': 'Multisine Transient', 'var': var_G_1_multi_trans, 'dif': dif_G_1_multi_trans},
    # Add similar dictionaries for your other signals (e.g., G_2_multi_window, G_3_multi_LPM, etc.)
]

for signal in signals:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot variance on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Variance [dB^2]', color=color)
    ax1.plot(f_range, rhs.DB(signal['var']), 'o-', label=f"Variance of {signal['name']}", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for bias
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Bias [dB]', color=color)  # we already handled the x-label with ax1
    ax2.plot(f_range, rhs.DB(signal['dif']), 's-', label=f"Bias of {signal['name']}", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    plt.title(f"Bias-Variance Plot for {signal['name']}\n{M} measurements, P={P} periods, N={N}.\nSNR$_u$={noise_input_u[1]} dB, SNR$_y$={noise_output_y[1]} dB")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.show()




"""
PLOTTING THE DIFFERENCE BETWEEN ETFE AND ML
"""

# Phase plot of the FRF transferfunction with multisine
#plt.plot(f_range,rhs.DB(G_ML),'o',label='$\hat{G}_{ML}$')
#plt.plot(f,rhs.DB((G_etfe)),'-',label='$\hat{G}_{etfe}$')
#plt.plot(f,rhs.DB(G_0),'-',label='$\hat{G}_{0}$')
#plt.plot(f_range,rhs.DB(G_LPM),'-',label='$\hat{G}_{LPM}$')

#plt.plot(f,rhs.DB(G_ML-G_etfe),'o',label='G_ML-G$_etfe$')
#plt.plot(f,rhs.DB(G_0-G_etfe),'-',label='|$\hat{G}_{0}$-$\hat{G}_{etfe}$|')
#plt.plot(f,rhs.DB(G_0-G_ML),'o',label='|$\hat{G}_{0}$-$\hat{G}_{ML}$|')

#plt.plot(f,rhs.DB(var_G_ML),'o',label='Variance ML')
#plt.plot(f,rhs.DB(var_G_etfe),'o',label='Variance etfe')
#plt.plot(f,rhs.DB(var_G_etfe-var_G_ML),'-',label='Variance etfe')

plt.title(f'{plot_signal} input in {M} measurments with P={P} and P_TF={P_tf}, windowing={window_set[0]}.\n Noise values SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()


#Figure of the created signal transient state
plt.plot(t_transient[:N_tf],y_n_transient[:N_tf]-y_n_steady[-N_tf:],label='y-y$_p$')
plt.plot(t_transient[:N_tf],y_n_steady[-N_tf:]-y_n_transient[:N_tf],label='y-y$_p$')
plt.plot(t_transient[:N_tf],transient_analytic[:N_tf],label='analytic')
plt.yscale('log')
plt.ylabel('y-y$_p$')
plt.legend()
plt.xlabel('Time[s]')
plt.title(f'Transient detection of output y over steady output y_p, where y_p is Period number {P_tf+1} ')
plt.show()



