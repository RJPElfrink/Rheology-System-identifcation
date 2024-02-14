import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import Rheosys as rhs


"""Set transfer function parameters"""
Lambda_constant=1.5                             # Value for lambda n/g
g_constant=2.5                                  # Value for spring constant g

f_s = 20                                        # Sample frequency
N = 2000                                        # Number of points
P = 4                                           # Number of repeated periods
P_Tf = 3                                        # Number of removed transient periods
interpolation_kind=[True,0]                     # Interpolation by Zero Order Hold
normalize_value   =[True,'STDev']               # Select normalization methode (None, STDev, Amplitude, RMS)

J1=1                                            # Starting frequency multisine
J2=N/2                                          # Stopping freqency multisine
phase_selection = 'Schroeder'                   # Select phase for the multisine, (Schroeder,Random,Linear,Zero,Rudin,Newman)

k_1 = 1                                         # Starting frequency chirp
k_2 = N/2                                       # Stopping frequency chirp

plot_signal='Chirp'                             # Select 'Chirp' or 'Multsine' to change plots

noise_input_u=[True,20,]                         # Set the value for noise on input to true or false and include SNR in decibels
noise_output_y=[True,20,]                       # Set the value for noise on output to true or false and include SNR in decibels

M_experiments=10                                 # Define te number amount of experiments to average the signal

smoothening='ETFE'

""" Calculation of Time window"""

f_0 =f_s/N                                      # Excitation frequency
T = N/f_s                                       # Time length
t= np.linspace(0, T,N*(P-P_Tf),endpoint=False)           # Time vector
f=np.linspace(0,f_s,N*(P-P_Tf),endpoint=False)           # Frequency range
t_transient= np.linspace(0, T*P,N*P,endpoint=False)        # Transient-time vector


### Transfer function of the maxwell system ###
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])
s2 = signal.lti([1,2], [1, 2, 3, 4])

# Convert transfer function to magnitude and phase of FRF
w, mag, phase = signal.bode(s1,f*2*np.pi)
w_tf=w/(2*np.pi)                        # Frequency of transfer function in Hz
phase_tf= np.deg2rad(phase)             # Fase of transfer function in radians
mag_tf=10**(mag/20)                     # Magnitude of transfer function [-]
G_tf=mag_tf*np.exp(1j*(phase_tf))

"""
TRANSIENT STATE CALCULATIONS
Calculations of the extended signal to remove the transient part
and repeat the signal to the full time window of N*P
clip the last period of Ntotal to select the steady state periode N
"""

if P<=P_Tf:
    raise ValueError("The number of periods must be larger then the removed amount of transfer free prediods")

# Calculation of the input signal for the multisine signal over the entire transient time period
u_mutlisine, phase_multisine=rhs.multisine(N,f_s, f_0, [J1,J2],P=P,phase_response=phase_selection,time_domain=True)

# Calculation of the input signal for the multisine signal over the entire transient time period
u_chirp=rhs.chirp_exponential(t,[k_1,k_2],f_s,N,P)

#Select u_chirp or u_multi as tranient signal for figure generation
if plot_signal=='Multisine':
    u_transient=u_mutlisine
elif plot_signal=='Chirp':
    u_transient=u_chirp

# Save the set transient basic signal and apply further normalization an interpolation
u_0_transient=u_transient

# Normalize the input signal to None, STDev, Amplitude or RMS
if normalize_value[0]==True:
    u_0_transient=rhs.normalization(u_0_transient,normalize=str(normalize_value[1]))

# Sampled transient signal as a function of t
if interpolation_kind[0]==True:
    u_0_transient_t=interp1d(t_transient,u_0_transient,kind=interpolation_kind[1])
    u_0_transient=u_0_transient_t(t_transient)

#u_transient=rhs.add_noise_with_snr(u_transient,2)
# Compute the response
t_out,y_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)


# Calculate G_0 without the disturbtion of any noise
y_0_transient=np.squeeze(y_out)                # y output in time domain entire transient signal
#y_0=y_transient[((N*P-N)):]                 # Copy of last steady period y output
#u_0=u_transient[((N*P-N)):]                 # Copy of last steady period u input
y_0=y_0_transient[((N*P_Tf)):]                 # Copy of last steady period y output
u_0=u_0_transient[((N*P_Tf)):]                 # Copy of last steady period u input
Y_0=fft(y_0)                                 # Y output to frequency domain
U_0=fft(u_0)                                 # U input to frequency domain or reconstructed signal
G_0=Y_0/U_0                                  # FRF of the multiphase

# Calculations of t he multiphase crest factor
print('Crest factor: {0:.2f} dB'.format(rhs.DB(rhs.crest_fac(u_0))))
print('Crest factor: {0:.2f} '.format(rhs.crest_fac(u_0)))

""" #Calculate of G^ by addition of noise"""
U_M=[]
Y_M=[]
G_M=[]

#Select u_chirp or u_multi as tranient signal for figure generation
if smoothening=='ML':
    mp=P-P_Tf
elif smoothening=='ETFE':
    mp=M_experiments

for m in range(mp):

    # Sampled transient signal as a function of t
    if interpolation_kind[0]==True:
        u_n_transient_t=interp1d(t_transient,u_transient,kind=interpolation_kind[1])
        u_n_transient=u_n_transient_t(t_transient)

    if noise_input_u[0]==True:
        n_u=rhs.add_noise_with_snr(u_n_transient,noise_input_u[1])
        u_n_transient=u_n_transient + n_u
    else:
        u_n_transient=u_n_transient

    # Normalize the input signal to None, STDev, Amplitude or RMS
    if normalize_value[0]==True:
        u_n_transient=rhs.normalization(u_n_transient,normalize=str(normalize_value[1]))


    #u_transient=rhs.add_noise_with_snr(u_transient,2)
    # Compute the response
    t_n_out,y_n_out, _ = signal.lsim(s1, U=u_n_transient, T=t_transient)

    if noise_output_y[0]==True:
        n_y = rhs.add_noise_with_snr(y_n_out,noise_output_y[1])
        y_n_transient= y_out + n_y
    else:
        y_n_transient=np.squeeze(y_out)          # y output in time domain entire transient signal

    y_n=y_n_transient[((N*P_Tf)):]               # Copy of last steady period y output
    u_n=u_n_transient[((N*P_Tf)):]               # Copy of last steady period u input
    Y_n=fft(y_n)                                 # Y output to frequency domain
    U_n=fft(u_n)                                 # U input to frequency domain or reconstructed signal
    G_n=Y_n/U_n                                  # FRF of the multiphase

    # Append the results for this experiment
    U_M.append(U_n)
    Y_M.append(Y_n)
    G_M.append(G_n)

# If you need to convert these lists to NumPy arrays after the loop
U_M = np.array(U_M)
Y_M = np.array(Y_M)
G_M = np.array(G_M)

G_etfe=np.mean(G_M,axis=0)

G_ML=np.sum(Y_M,axis=0)/np.sum(U_M,axis=0)

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(abs(np.squeeze(G_etfe))),'-',label='$\hat{G}$')
#plt.plot(f,rhs.DB(abs(np.squeeze(G_ML))),'-',label='Multisine $\hat{G}$ ML response')
plt.plot(f,rhs.DB(abs(G_0)),'-',label='G$_0$')
#plt.plot(w_tf,rhs.DB(mag_tf),'-',label="Transfer Function")
#plt.plot(f,rhs.DB((abs(G_etfe)-abs(G_ML))),'o',label='|G$_etfe$-G$_ML$|')
#plt.plot(f,rhs.DB((abs(mag_tf)-abs(G_0))),'o',label='|G$_{tf}$-G$_0$|')
#plt.plot(f,rhs.DB((abs(G_etfe)-abs(G_0))),'o',label='|$\hat{G}$-G$_0$|')
plt.title(f'{plot_signal} input ETFE example M={M_experiments}, SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

break

transient_analytic = np.exp(-t_transient/Lambda_constant) # analytical transient curve

# Figure of the created multisine signal in time domain
plt.plot(t,u_0)
plt.title('Multisine input signal of last period')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
plt.plot(f,rhs.DB(U_0))
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.xlim([f_s/N,f_s/2])
plt.title('Frequency range of input signal U')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t_transient,u_0_transient)
plt.title('Input signal u of the ODE Maxwell transient signal')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t_transient,y_0_transient)
plt.title('Output signal y of the ODE Maxwell transient signal')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t,y_0)
plt.title('Output signal y of the ODE Maxwell steady state ')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
#plt.plot(t_transient,np.tile(y_0,P)-y_transient,label='y-y$_p$')
#plt.plot(t_transient,transient_analytic,label='analytical transient')
#plt.ylabel('y-y$_p$')
#plt.ylim(-0.001,1.5*np.max(np.tile(y_0,P)-y_transient))
#plt.legend()
#plt.xlabel('Time[s]')
#plt.title('Transient detection of the transient ouput y over steady output y')
#plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(abs(G_0)),'-',label='Multisine response')
plt.plot(w_tf,rhs.DB(mag_tf),'-',label="Transfer Function")
plt.plot(f,rhs.DB((abs(G_0)-abs(mag_tf))),'o',label='|G(j$\omega$)multisine-G(j$\omega$)tf|')
plt.title('FRF magnitude response of the maxwell model G$_0$')
plt.legend()
plt.xlim([f_s/N,f_s/2])
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f,(np.angle(G_0,True)) ,'-',label="Multisine respone")
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
plt.plot(f*2*np.pi, rhs.DB(abs(G_0)), '-', label='Multisine response')
plt.plot(w_tf*2*np.pi, rhs.DB(mag_tf), '-', label="Transfer Function")
plt.xlim([(f_s*2*np.pi)/N,f_s*np.pi])
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')

plt.subplot(2, 1, 2) # phase plot
plt.plot(f*2*np.pi, np.angle(G_0,True), '-', label="Multisine response")
plt.plot(w_tf*2*np.pi, np.rad2deg(phase_tf), '-', label="Transfer Function")
plt.xlim([(f_s*2*np.pi)/N,f_s*np.pi])
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (rad/s)')

# Show the figure
plt.tight_layout()
plt.show()


