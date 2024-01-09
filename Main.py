import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import rfft,rfftfreq
from scipy import signal
import Rheosys as rhs
import Visualplots

# Script for calculating the FRF of the Maxwell model
# The FRF of generated multisine is

f_s = 2000                                        # Sample frequency
N = 2000                                          # Number of points

J1=1                                             # Starting frequency multisine
J2=N/2                                            # Stopping freqency multisine

NBp = 3                                          # Number of repeated periods
N_transient=N*NBp                                # Number of transient points

Lambda_constant=1.5                              # Value for lambda n/g
g_constant=2.5                                   # Value for spring constant g
kind=0                                           # interpolation kind

# Calculation of time window
f_0 =f_s/N                                       # Excitation frequency
T = N/f_s                                        # Time length
t= np.linspace(0, T,N,endpoint=False)            # Time vector
f=rfftfreq(N,1/f_s)                              # Frequency range

# Calculate the multisine signal with selected phase and normalization
# Phase options: Schroeder, Random, Zero, Linear, Newman, Rudin
u_multi=rhs.multisine(N,f_s, f_0, [J1,J2],phase_response='Schroeder',normalize='Amplitude',time_domain=True,Tau=900)

# Reconstruction of signal between sample points, use of scipy interpolation, kind is 0,2 and all odd numbers
u_multi_t_int=interp1d(t,u_multi,kind=kind)
u_multi_int = u_multi_t_int(t)

# Solving of the maxwell differentail equation
def gamma_dot_multi(t):
    return u_multi_t_int(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    return G*gamma_dot_multi(t) - tau/L

# Solve function for the Maxwel ODE
sol_multi = solve_ivp(ODE_maxwell, [0, t[-1]], [0], args=(Lambda_constant,g_constant), t_eval=t)

# Solutions of the ODE
y_multi=np.squeeze(sol_multi.y)         # y output in time domain
Y_multi=rfft(y_multi)                    # Y output to frequency domain
U_multi=rfft(u_multi_int)                # U input to frequency domain or reconstructed signal
G_multi=Y_multi/U_multi                 # FRF of the multiphase


# Calculations of t he multiphase crest factor
print('Crest factor: {0:.2f} dB'.format(rhs.DB(rhs.crest_fac(u_multi))))
print('Crest factor: {0:.2f} '.format(rhs.crest_fac(u_multi)))

# Figure of the created multisine signal in time domain
plt.plot(t,u_multi)
plt.title('Multisine input signal')
plt.ylabel('Amplitude (dB)')
plt.xlabel('Time (s)')
plt.show()

# Figure of the created multisine signals frequency domain in Decibel
plt.plot(f,rhs.DB(U_multi))
#plt.xlim([0, f_s / 2])
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.xscale('log')
plt.show()

# plot the reconstructed zero-order-hold signal over the original signal
Visualplots.input_reconstructer_sampling(t,u_multi,t,u_multi_int)

# plot the maxwel solution over time
Visualplots.maxwel_plot(t,np.squeeze(sol_multi.y),title='ODE Maxwel of multisine Schroeder phase')

# Transfer function of the maxwell system
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])

# Transfer function plots of the maxwel system
w, mag, phase = signal.bode(s1,f)

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f, np.angle(G_multi,deg=True),label="Multisine respone")
plt.plot(w,phase,label="Transfer Function")
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (Hz) test')
plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f, rhs.DB(abs(G_multi)),label='Multisine response')
plt.plot(w,mag,label="Transfer Function")
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz) test')
plt.show()


### TRANSIENT STATE CALCULATIONS ### moeten nog verder worden beschreven
# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*NBp
# clip the last period of Ntotal to select the steady state periode N

T_transient=N_transient/f_s                 # Transinet Time length
T_s = 1/f_s                                 # Sampling time
t_transient= np.linspace(0, T,N,endpoint=False)       # Transient Time vector

t_transient= np.linspace(0, T_transient,N_transient,endpoint=False)        # Transient-time vector
u_transient=rhs.multisine(N_transient,f_s, f_0, [J1,J2],phase_response='Schroeder',normalize='RMS',time_domain=True)
u_transient_t=rhs.multisine(N_transient,f_s, f_0, [J1,J2],phase_response='Schroeder',normalize='RMS',time_domain=False)

u_transient_int_t=interp1d(t_transient,u_transient,kind=kind)

def gamma_dot(t):
    return u_transient_t(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,g):
    #L,G=args
    return g*gamma_dot(t) - tau/L

# Solve function for the Maxwel ODE
sol_transient = solve_ivp(ODE_maxwell, [0, t_transient[-1]], [0], args=(Lambda_constant, g_constant), t_eval=t_transient)

# Plot of the total length of the transient signal output
Visualplots.maxwel_plot(t_transient,np.squeeze(sol_transient.y),title='ODE Maxwell transient signal')

# Solutions of the ODE
y_transient=np.squeeze(sol_transient.y)          # y output in time domain entire transient signal
y_steady=y_transient[((N_transient-N)):]         # Copy of last steady period y output
u_steady=u_transient[((N_transient-N)):]         # Copy of last steady period u input
Y_steady=rfft(y_steady)                           # Y output to frequency domain
U_steady=rfft(u_steady)                           # U input to frequency domain or reconstructed signal
G_steady=Y_steady/U_steady                       # FRF of the multiphase

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f,np.angle(G_steady,deg=True) ,'o',label="Multisine respone")
plt.plot(w,phase,'o',label="Transfer Function")
plt.title('FRF response of the maxwell model')
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (Hz) test')
plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,20*np.log10(G_steady),'o',label='Multisine response')
plt.plot(w,mag,'o',label="Transfer Function")
plt.title('FRF response of the maxwell model')
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz) test')
plt.show()

