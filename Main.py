import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import Rheosys as rhs

# Script for calculating the FRF of the Maxwell model
# CHANGE VALUES ONLY IN LINES 13-24

f_s = 2                                         # Sample frequency
N = 200                                         # Number of points

J1=1                                             # Starting frequency multisine
J2=80                                           # Stopping freqency multisine

P = 4                                            # Number of repeated periods
N_transient=N*P                                  # Number of transient points

Lambda_constant=1.5                              # Value for lambda n/g
g_constant=2.5                                   # Value for spring constant g
kind=0                                           # interpolation kind

# Calculation of time window
f_0 =f_s/N                                       # Excitation frequency
T = N/f_s                                        # Time length
t= np.linspace(0, T,N,endpoint=False)            # Time vector
#f=fftfreq(N,1/f_s)                              # Frequency range
f=np.linspace(0,f_s,N,endpoint=False)

# Transfer function of the maxwell system
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])

# Transfer function plots of the maxwel system
w_tf, mag_tf, phase_tf = signal.bode(s1,f*2*np.pi)

### TRANSIENT STATE CALCULATIONS ###
# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*P
# clip the last period of Ntotal to select the steady state periode N

T_transient=N_transient/f_s                 # Transinet Time length
t_transient= np.linspace(0, T_transient,N_transient,endpoint=False)        # Transient-time vector

# Calculation of the input signal over the entire transient time period
u_transient, phase_u=rhs.multisine(N,f_s, f_0, [J1,J2],P=P,phase_response='Schroeder',normalize='STDev',time_domain=True)

# Calculation of the input signal over the entire transient time period as a function of t
u_transient_t, phase_u=rhs.multisine(N,f_s, f_0, [J1,J2],P=P,phase_response='Schroeder',normalize='STDev',time_domain=False)


# Sampled transient signal as a function of t
u_transient_int_t=interp1d(t_transient,u_transient,kind=kind)
u_transient=u_transient_t(t_transient)
# Strain rate function, input signal dependent on t
def gamma_dot(t):
    return u_transient_t(t)

# Maxwell function
# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,g):
    return g*gamma_dot(t) - tau/L

# Solve function for the Maxwel ODE
sol_transient = solve_ivp(ODE_maxwell, [0, t_transient[-1]], [0], args=(Lambda_constant, g_constant), t_eval=t_transient)

# Solutions of the ODE
y_transient=np.squeeze(sol_transient.y)          # y output in time domain entire transient signal
y_steady=y_transient[((N_transient-N)):]         # Copy of last steady period y output
u_steady=u_transient[((N_transient-N)):]         # Copy of last steady period u input
Y_steady=fft(y_steady)                           # Y output to frequency domain
U_steady=fft(u_steady)                           # U input to frequency domain or reconstructed signal
G_steady=Y_steady/U_steady                       # FRF of the multiphase
G_mag=abs(G_steady)
G_phase=np.angle(G_steady)

# Calculations of t he multiphase crest factor
print('Crest factor: {0:.2f} dB'.format(rhs.DB(rhs.crest_fac(u_steady))))
print('Crest factor: {0:.2f} '.format(rhs.crest_fac(u_steady)))

# Figure of the created multisine signal in time domain
plt.plot(t,u_steady)
plt.title('Multisine input signal of last period')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
plt.plot(f,rhs.DB(U_steady))
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.title('Frequency range of input signal U')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
plt.plot(t_transient,y_transient-np.tile(y_steady,P))
plt.ylabel('y-y$_p$')
plt.xlabel('Time[s]')
plt.title('Transient detection of the ouput y')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t,y_steady)
plt.title('Output signal y of the ODE Maxwell steady state')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()


# Figure of the created multisine signal in time domain
plt.plot(t_transient,u_transient)
plt.title('Output signal y of the ODE Maxwell steady state')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signal in time domain
plt.plot(t_transient,y_transient)
plt.title('Output signal y of the ODE Maxwell steady state')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f,np.angle(G_steady,deg=True) ,'-',label="Multisine respone")
plt.plot(w_tf/(2*np.pi),phase_tf,'-',label="Transfer Function")
plt.title('FRF phase response of the maxwell model G$_0$')
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (Hz)')
plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,20*np.log10(G_steady),'-',label='Multisine response')
plt.plot(w_tf/(2*np.pi),mag_tf,'-',label="Transfer Function")
plt.title('FRF magnitude response of the maxwell model G$_0$')
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

# Create a figure with two subplots
plt.figure()

# Bode plot in rad/s
plt.subplot(2, 1, 2) # magnitude plot
plt.plot(f*2*np.pi, 20*np.log10(G_steady), '-', label='Multisine response')
plt.plot(w_tf, mag_tf, '-', label="Transfer Function")
plt.title('FRF magnitude response of the Maxwell model G$_0$')
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (rad/s)')

plt.subplot(2, 1, 1) # phase plot
plt.plot(f*2*np.pi, np.angle(G_steady, deg=True), '-', label="Multisine response")
plt.plot(w_tf, phase_tf, '-', label="Transfer Function")
plt.title('FRF phase response of the Maxwell model G$_0$')
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (rad/s)')

# Show the figure
plt.tight_layout()  # Adjusts the plots to fit into the figure area.
plt.show()