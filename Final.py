import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import Rheosys as rhs
from scipy.io import loadmat
from scipy.io import savemat
import matlab.engine


"""Set transfer function parameters"""
Lambda_constant=1.5                        # Value for lambda n/g
g_constant=2.5                             # Value for spring constant g

f_s = 20                                   # Sample frequency
N = 2000                                   # Number of points
P = 1                                     # Number of repeated periods
P_tf = 0                                   # Number of removed transient periods
interpolation_kind=[True,0]                # Interpolation by Zero Order Hold
normalize_value   =[True,'STDev']           # Select normalization methode (None, STDev, Amplitude, RMS)

M = 1

j_1=1                                      # Starting frequency multisine
j_2=N/2                                    # Stopping freqency multisine
phase_selection = 'Schroeder'              # Select phase for the multisine, (Schroeder,Random,Linear,Zero,Rudin,Newman)

k_1 = 1                                    # Starting frequency chirp
k_2 = N/2                                  # Stopping frequency chirp

noise_input_u=[True,20,]                   # Set the value for noise on input to true or false and include SNR in decibels
noise_output_y=[True,20,]                  # Set the value for noise on output to true or false and include SNR in decibels

window_set=[False,0.15]                    # Set window to true if needed, set value between 0-1, 0 is rectangular, 1 is Hann window

plot_signal='Multisine'                    # Select 'Chirp' or 'Multsine' to change plots

# To calculate the LocalPolynomialMeasurement set it to True
# Values indicate method.order[2] order of approximation
# method.dof [1] degrees of freedom, independent experiments
# method.transient [1], include transien estimation = , exclude transient estimation = 0
LPM=[True,2,1,0]

""" Calculation of Time window"""
check = rhs.check_variables(f_s,N,P,P_tf,window_set)

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
G_0=mag_tf*np.exp(1j*(phase_tf))           # FRF from defined transfer function

# analytical transient curve
transient_analytic = np.exp(-t_transient[:(N)]/Lambda_constant)


"""
CALCULATION FOR G_0
The first steady state periode of the total signal is used to calculate G_0
"""

if P<=P_tf:
    raise ValueError("The number of periods P must be larger then the removed amount of transient free prediods P_tf")

# Calculation of the input signal for the multisine signal over the entire transient time period
u_mutlisine, phase_multisine=rhs.multisine(f_s,N,P, [j_1,j_2],phase_response=phase_selection,time_domain=True)

# Calculation of the input signal for the multisine signal over the entire transient time period
u_chirp=rhs.chirp_exponential(f_s,N,P,[k_1,k_2])

#Select u_chirp or u_multi as tranient signal for figure generation
if plot_signal=='Multisine':
    #u_transient=u_mutlisine
    u_select   = u_mutlisine
    band_range = [j_1,j_2]
elif plot_signal=='Chirp':
    #u_transient=u_chirp
    u_select = u_chirp
    band_range = [k_1,k_2]
excitedharm_range = np.linspace(band_range[0],band_range[1],int(band_range[1]-band_range[0])+1,endpoint=True)
f_range = excitedharm_range*f_0
u_transient= np.tile(u_select, P)
if P_tf!=0:
    u_transient = np.insert(u_transient,0,u_transient[-N_tf:])


# Save the set transient basic signal and apply further normalization an interpolation
u_0= u_select
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


if window_set[0]==True:
    # Calculate the window function again
    window = rhs.window_function(t, T, window_set[1])

    # Apply the window function individually to each period,
    window_transient = np.tile(window, P+P_tf)

    # Applying the extended window function to the extended signal
    u_0_transient = u_transient * window_transient
else:
    u_0_transient = u_transient
# Sampled transient signal as a function of t
if interpolation_kind[0]==True:
    u_0_transient_t=interp1d(t_transient,u_0_transient,kind=interpolation_kind[1])
    u_0_transient=u_0_transient_t(t_transient)

# Normalize the input signal to None, STDev, Amplitude or RMS
if normalize_value[0]==True:
    u_0_transient=rhs.normalization(u_0_transient,normalize=str(normalize_value[1]))


# Initialize lists to collect data
U_M = []
Y_M = []
G_etfe_p=[]
G_ML_m=[]
G_LPM_m =[]

for m in range(M):
    """
    CALCULATIONS OF NOISE
    Starting with the addition of measurement noise on either inpot or output
    Noise on input is not transfer  trough the transfer function in calculation of the output
"""
    # Apply measurement noise to the input
    if noise_input_u[0]==True:
        n_u=rhs.add_noise_with_snr(u_0_transient,noise_input_u[1])
        u_n_transient=u_0_transient + n_u
    else:
        u_n_transient=u_0_transient


    # Compute the response output y by
    t_out,y_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)

    # Application of measurement noise on the ouput
    if noise_output_y[0]==True:
        n_y = rhs.add_noise_with_snr(y_out,noise_output_y[1])
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

        # Extract one period from the transient signals
        y_p_period = y_n_transient[start_idx:end_idx]
        u_p_period = u_n_transient[start_idx:end_idx]

        u_p.append(u_p_period)
        y_p.append(y_p_period)

        # Calculate FFT of the extracted period
        Y_p_period = fft(y_p_period)
        U_p_period = fft(u_p_period)

        # Append FFT results to the lists
        Y_p.append(Y_p_period)
        U_p.append(U_p_period)

        # Calculate FRF for each period and append to G_n
        G_etfe_p.append(Y_p_period/U_p_period)

    G_LPM_m.append(rhs.matpy_LPM(u_p,y_p,u_select,N,f_s,excitedharm_range))
    U_M.append(U_p)
    Y_M.append(Y_p)
    G_ML_m.append(np.sum(Y_p,axis=0)/np.sum(U_p,axis=0))

U_M=np.array(U_M)
Y_M=np.array(Y_M)
G_etfe=np.array(np.mean(G_etfe_p,axis=0))
G_ML=np.array(np.mean(G_ML_m,axis=0))
G_LPM=np.array(np.squeeze(G_LPM_m))

# FRF calculation by Maximum Likelihood
var_G_ML=np.var(G_ML_m,axis=0)
var_G_etfe=np.var(G_etfe_p,axis=0)

# Save specified signal
G_0_final=np.save('G_0_multisine.npy',G_0)


"""
PLOTTING FOR VISUALIZATION OF THE RESULTS

"""

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(G_ML),'o',label='$\hat{G}_{ML}$')
plt.plot(f,rhs.DB((G_etfe)),'-',label='$\hat{G}_{etfe}$')
plt.plot(f,rhs.DB(G_0),'-',label='$\hat{G}_{0}$')
plt.plot(f_range,rhs.DB(G_LPM),'-',label='$\hat{G}_{LPM}$')

#plt.plot(f,rhs.DB(G_ML-G_etfe),'o',label='G_ML-G$_etfe$')
plt.plot(f,rhs.DB(G_0-G_etfe),'-',label='|$\hat{G}_{0}$-$\hat{G}_{etfe}$|')
plt.plot(f,rhs.DB(G_0-G_ML),'o',label='|$\hat{G}_{0}$-$\hat{G}_{ML}$|')

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





