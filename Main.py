import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import Rheosys as rhs
from scipy.io import loadmat


"""Set transfer function parameters"""
Lambda_constant=1.5                        # Value for lambda n/g
g_constant=2.5                             # Value for spring constant g

f_s = 20                                   # Sample frequency
N = 2000                                   # Number of points
P = 1                                     # Number of repeated periods
P_tf = 0                                   # Number of removed transient periods
interpolation_kind=[True,0]                # Interpolation by Zero Order Hold
normalize_value   =[True,'STDev']           # Select normalization methode (None, STDev, Amplitude, RMS)

M =1

j_1=1                                      # Starting frequency multisine
j_2=N/2                                    # Stopping freqency multisine
phase_selection = 'Schroeder'              # Select phase for the multisine, (Schroeder,Random,Linear,Zero,Rudin,Newman)

k_1 = 1                                    # Starting frequency chirp
k_2 = N/2                                  # Stopping frequency chirp

noise_input_u=[True,40,]                   # Set the value for noise on input to true or false and include SNR in decibels
noise_output_y=[True,40,]                  # Set the value for noise on output to true or false and include SNR in decibels

window_set=[False,0.15]                    # Set window to true if needed, set value between 0-1, 0 is rectangular, 1 is Hann window

plot_signal='Chirp'                    # Select 'Chirp' or 'Multsine' to change plots

""" Calculation of Time window"""
check = rhs.check_variables(f_s,N,P,P_tf,window_set)

f_0   = f_s/N                                         # Excitation frequency
T     = N/f_s                                         # Time length
t     = np.linspace(0, T,N,endpoint=False)            # Time vector
f     = np.linspace(0,f_s,N,endpoint=False)           # Frequency range
f_range = np.linspace(j_1,j_2,int(j_2-j_1)+1,endpoint=True)
N_tf  = int(P_tf*N)                                   # Number of transient points
t_transient= np.linspace(0, T*(P+P_tf),(N*P+N_tf),endpoint=False)   # Transient-time vector

### Transfer function of the maxwell system ###
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])
s2 = signal.lti([1,2], [1, 2, 3, 4])

# Convert transfer function to magnitude and phase of FRF
w, mag, phase = signal.bode(s1,f*2*np.pi)
w_tf=w/(2*np.pi)                            # Frequency of transfer function in Hz
phase_tf= np.deg2rad(phase)                 # Fase of transfer function in radians
mag_tf=10**(mag/20)                         # Magnitude of transfer function [-]
G_tf=mag_tf*np.exp(1j*(phase_tf))           # FRF from defined transfer function

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

u_transient= np.tile(u_select, P)
if P_tf!=0:
    u_transient = np.insert(u_transient,0,u_transient[-N_tf:])


# Save the set transient basic signal and apply further normalization an interpolation
u_0_transient=u_transient

if window_set[0]==True:
    # Calculate the window function again
    window = rhs.window_function(t, T, window_set[1])

    # Apply the window function individually to each period,
    window_transient = np.tile(window, P+P_tf)

    # Applying the extended window function to the extended signal
    u_0_transient = u_0_transient * window_transient

# Normalize the input signal to None, STDev, Amplitude or RMS
if normalize_value[0]==True:
    u_0_transient=rhs.normalization(u_0_transient,normalize=str(normalize_value[1]))

# Sampled transient signal as a function of t
if interpolation_kind[0]==True:
    u_0_transient_t=interp1d(t_transient,u_0_transient,kind=interpolation_kind[1])
    u_0_transient=u_0_transient_t(t_transient)

# Compute the response
t_out,y_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)

# Calculate G_0 without the disturbtion of any noise
y_0_transient=np.squeeze(y_out)                # y output in time domain entire transient signal
#y_0=y_0_transient[(N*P_tf):(N*(P_tf+1))]       # Copy of first steady period y output
#u_0=u_0_transient[(N*P_tf):(N*(P_tf+1))]       # Copy of fire steady period u input
y_0=y_0_transient[N_tf:(N+N_tf)]                # Copy of first steady period y output
u_0=u_0_transient[N_tf:(N+N_tf)]                # Copy of fire steady period u input
Y_0=fft(y_0)                                   # Y output to frequency domain
U_0=fft(u_0)                                   # U input to frequency domain or reconstructed signal
G_0=Y_0/U_0                                    # FRF of the multiphase

# Calculations of the multiphase crest factor
fact_crest = rhs.crest_fac(u_0)
fact_effic = rhs.power_efficiency(U_0,band_range)
fact_loss  = rhs.power_loss(U_0,band_range)
fact_quali = rhs.quality_factor(fact_crest,fact_effic,fact_loss)

print('Crest factor: {0:.2f} '.format(fact_crest))
print('Power efficiency: {0:.2f} '.format(fact_effic))
print('Power loss: {0:.2f} '.format(fact_loss))
print('Signal quality: {0:.2f} '.format(fact_quali))


# Initialize lists to collect data
u_M = []
y_M = []
U_M = []
Y_M = []
G_M = []

for m in range(M):
    """
    CALCULATIONS OF NOISE
    Starting with the addition of measurement noise on either inpot or output
    Noise on input is not transfer trough the transfer function in calculation of the output
    """
    # Sampled transient signal as a function of t
    if interpolation_kind[0]==True:
        u_n_transient_t=interp1d(t_transient,u_transient,kind=interpolation_kind[1])
        u_n_transient=u_n_transient_t(t_transient)

    # Apply measurement noise to the input
    if noise_input_u[0]==True:
        n_u=rhs.add_noise_with_snr(u_n_transient,noise_input_u[1])
        u_n_transient=u_n_transient + n_u
    else:
        u_n_transient=u_n_transient

    if window_set[0]==True:
        # Calculate the window function again
        window = rhs.window_function(t, T, window_set[1])

        # Apply the window function individually to each period,
        window_transient = np.tile(window, P+P_tf)

        # Applying the extended window function to the extended signal
        u_n_transient = u_n_transient * window_transient


    # Normalize the input signal to None, STDev, Amplitude or RMS
    if normalize_value[0]==True:
        u_n_transient=rhs.normalization(u_n_transient,normalize=str(normalize_value[1]))

    # Compute the response output y by, input noise is only on measurement not on generator, so the model is inserted with u_0
    t_n_out,y_n_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)

    # Application of measurement noise on the ouput
    if noise_output_y[0]==True:
        n_y = rhs.add_noise_with_snr(y_n_out,noise_output_y[1])
        y_n_transient= y_out + n_y
    else:
        y_n_transient=np.squeeze(y_out)          # y output in time domain entire transient signal


    """
    CALCULATION OF G^ IN MULTIPLE PERIODS
    Depending on the amount of selected transfer free periods
    The overall Frequency responce is calculated

    """

    U_n=[]
    Y_n=[]
    G_n=[]

    u_steady=u_n_transient[N_tf:]
    y_steady=y_n_transient[N_tf:]

    for p in range(P):

        # Calculate indices for extracting one period
        start_idx = (N *   p)
        end_idx = start_idx + N

        # Extract one period from the transient signals
        y_n_period = y_n_transient[start_idx:end_idx]
        u_n_period = u_n_transient[start_idx:end_idx]

        # Calculate FFT of the extracted period
        Y_n_period = fft(y_n_period)
        U_n_period = fft(u_n_period)

        # Append FFT results to the lists
        Y_n.append(Y_n_period)
        U_n.append(U_n_period)

        # Calculate FRF for each period and append to G_n
    U_n=np.array(U_n)
    Y_n=np.array(Y_n)
    G_n.append(np.sum(Y_n,axis=0)/np.sum(U_n,axis=0))

    u_M.append(u_n_transient)
    y_M.append(y_n_transient)
    G_M.append(G_n)
    U_M.append(U_n)
    Y_M.append(Y_n)

u_M=np.array(u_M)
y_M=np.array(y_M)
U_M=np.array(U_M)
Y_M=np.array(Y_M)
G_M=np.array(G_M)


data_u = u_M.reshape(1, 1, M, -1)  # Reshape assuming u_n_transient is a 1D array for each experiment
data_y = y_M.reshape(1, 1, M, -1)  # Reshape assuming y_n_transient is a 1D array for each experiment


# FRF calculation by emperical transfer function estimate
G_etfe=(np.mean(G_M,axis=0))

# FRF calculation by Maximum Likelihood
G_ML=np.mean(np.sum(Y_M,axis=0)/np.sum(U_M,axis=0),axis=0)
G_MLn=np.sum(Y_n,axis=0)/np.sum(U_n,axis=0)
var_G_ML=np.var(np.sum(Y_M,axis=0)/np.sum(U_M,axis=0),axis=0)


#matlab_transfer=rhs.matlab_input_file('Matlab_input_multisine.mat',u_n_transient,y_n_transient,u_select,N,1/f_s,f_range)
#matlab_transfer=rhs.matlab_input_file('Matlab_input_multisineYU.mat',U_n,Y_n,f,N,1/f_s,f_range)


#np.save('G_0_multisine.npy',G_0)

"""
PLOTTING FOR VISUALIZATION OF THE RESULTS

"""

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(abs(G_ML)),'-',label='$\hat{G}_{ML}$')
#plt.plot(f,rhs.DB(abs(np.squeeze(G_LPM))),'-',label='$\hat{G}_{LPM}$')
#plt.plot(f,rhs.DB(abs(np.squeeze(G_ML))),'-',label='Multisine $\hat{G}$ ML response')
plt.plot(f,rhs.DB(abs(G_0)),'-',label='G$_0$')
#plt.plot(f,rhs.DB((abs(G_etfe)-abs(G_ML))),'o',label='|$\hat{G}_{etfe}$-$\hat{G}_{ML}$|')
#plt.plot(f,rhs.DB((abs(mag_tf)-abs(G_0))),'o',label='|G$_{tf}$-G$_0$|')
plt.plot(f,rhs.DB((abs(G_ML)-abs(G_0))),'o',label='|$\hat{G}_{ML}$-G$_0$|')
plt.plot(f,rhs.DB((abs(G_MLn)-abs(G_0))),'o',label='|$\hat{G}_{MLn}$-G$_0$|')
plt.plot(f,rhs.DB(var_G_ML),'o',label='Variance')
plt.title(f'{plot_signal} input ETFE example P={P} with P_TF={P_tf}, SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t,u_0)
plt.title(f'{plot_signal} of first steady state periode, periode number {P_tf+1}')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
plt.plot(f,rhs.DB(U_0))
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.xlim([f_s/N,f_s/2])
plt.title(f'Frequency range of the {plot_signal} input signal U')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t_transient,u_0_transient)
plt.title('Input signal u of the ODE Maxwell transient signal')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t,y_0)
plt.title('Output signal y of the ODE Maxwell the first steady state period ')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

if P_tf!=0:

    #Figure of the created signal transient state
    plt.plot(t_transient[:N_tf],y_0_transient[:N_tf]-y_0[-N_tf:],label='y-y$_p$')
    plt.plot(t_transient[:N_tf],y_0[-N_tf:]-y_0_transient[:N_tf],label='y-y$_p$')
    plt.plot(t_transient[:N_tf],transient_analytic[:N_tf],label='analytic')
    plt.yscale('log')
    plt.ylabel('y-y$_p$')
    plt.legend()
    plt.xlabel('Time[s]')
    plt.title(f'Transient detection of ouput y over steady output y_p, where y_p is Period number {P_tf+1} ')
    plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(abs(G_0)),'-',label=f'G_0 {plot_signal}')
plt.plot(w_tf,rhs.DB(mag_tf),'-',label="Transfer Function")
plt.plot(f,rhs.DB((abs(G_0)-abs(mag_tf))),'o',label=f'|G_0(j$\omega$){plot_signal}-G(j$\omega$)tf|')
plt.title('FRF magnitude response of the maxwell model G$_0$')
plt.legend()
plt.xlim([f_s/N,f_s/2])
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f,(np.angle(G_0,True)) ,'-',label=f"G_0 {plot_signal} respone")
plt.plot(w_tf,np.rad2deg(phase_tf),'-',label="Transfer Function")
plt.title('FRF phase response of the maxwell model G$_0$')
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlim([f_s/N,f_s/2])
plt.xlabel('Frequency (Hz)')
plt.show()

# Bode plot in rad/s
plt.figure()
plt.subplot(2, 1, 1) # magnitude plot
plt.title('Bode plot of the FRF Maxwell model G$_0$')
plt.plot(f*2*np.pi, rhs.DB(abs(G_0)), '-', label=f'{plot_signal} response')
plt.plot(w_tf*2*np.pi, rhs.DB(mag_tf), '-', label="Transfer Function")
plt.xlim([(f_s*2*np.pi)/N,f_s*np.pi])
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')

plt.subplot(2, 1, 2) # phase plot
plt.plot(f*2*np.pi, np.angle(G_0,True), '-', label=f"{plot_signal} response")
plt.plot(w_tf*2*np.pi, np.rad2deg(phase_tf), '-', label="Transfer Function")
plt.xlim([(f_s*2*np.pi)/N,f_s*np.pi])
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (rad/s)')

# Show the figure
plt.tight_layout()
plt.show()


# Load the MATLAB file
#data = loadmat('G_output.mat')

# Access the 'output_data' structure
#output_data = data['output_data']

#G_data = np.squeeze(output_data['G'][0,0])

#plt.plot(f_range,rhs.DB(G_data))
#plt.xscale('log')
#plt.show()
