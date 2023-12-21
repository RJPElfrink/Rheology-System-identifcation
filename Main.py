import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft,fftshift,ifft,ifftshift,rfft,rfftfreq,irfft,fftfreq
from scipy import signal
import Rheosys as rhs
import Visualplots

#random.seed(10)

f_s = 480                                        # Sample frequency
N = 480                                      # Number of points

Lambda_constant=1.5                              # Value for lambda n/g
g_constant=2.5                                   # Value for spring constant g

J1=1                                             # Starting frequency multisine
J2=200                                           # Stopping freqency multisine

kind=0                                           # interpolation kind
Up=100                                           # Upsampling for plots

NBp = 5                                         # Number of block points
Ntotal=N*NBp                                     # Number of transient points

# Calculation of time window
f_0 =f_s/N                                   # Excitation frequency
T = N/f_s                                        # Time length
t= np.linspace(0, T,N,endpoint=False)            # Time vector
t_CT= np.linspace(0, t[-1], N*Up,endpoint=True)  # Over sampling continuous-time vector
f=np.linspace(0,f_s,N,endpoint=False)            # Frequency range

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

sol_multi = solve_ivp(ODE_maxwell, [0, t[-1]], [0], args=(Lambda_constant,g_constant), t_eval=t)

# Solutions of the ODE
y_multi=np.squeeze(sol_multi.y)         # y output in time domain
Y_multi=fft(y_multi)                    # Y output to frequency domain
U_multi=fft(u_multi_int)                # U input to frequency domain or reconstructed signal
G_multi=Y_multi/U_multi                 # FRF of the multiphase


# Calculations of the multiphase crest factor
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
plt.xlim([0, f_s / 2])
plt.ylabel('Amplitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.show()

# plot the reconstructed zero-order-hold signal over the original signal
Visualplots.input_reconstructer_sampling(t,u_multi,t,u_multi_int)

# plot the maxwel solution over time
Visualplots.maxwel_plot(t,np.squeeze(sol_multi.y),title='ODE Maxwel of multisine Schroeder phase')

# Transfer function of the maxwell system
# G=g/((1j*f)+1/L)
s1 = signal.lti([g_constant], [1, 1/Lambda_constant])

# Transfer function plots of the maxwel system
w, mag, phase = s1.bode()

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f, rhs.DB(abs(G_multi)),label="Multisine respone")
plt.plot(w,phase,label="Transfer Function")  # magnitude plot
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (Hz) test')
plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f, np.angle(G_multi,deg=True),label='Multisine response')
plt.plot(w,mag,label="Transfer Function")
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz) test')
plt.show()


### TRANSIENT STATE CALCULATIONS ### moeten nog verder worden beschreven
# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*NBp
Ttotal=Ntotal/f_s                 # Time length
T_s = 1/f_s                        # Sampling time

t_Trans= np.linspace(0, Ttotal,Ntotal,endpoint=False)        # Transient-time vector
u_Trans=rhs.multisine(Ntotal,f_s, f_0, [J1,J2],phase_response='Schroeder',normalize='Amplitude',time_domain=True,Tau=900)
u_multi_t=rhs.multisine(Ntotal,f_s, f_0, [J1,J2],phase_response='Schroeder',normalize='Amplitude',time_domain=False,Tau=900)


def gamma_dot(t):

    return u_multi_t(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,g):
    #L,G=args
    return g*gamma_dot(t) - tau/L

sol_Trans = solve_ivp(ODE_maxwell, [0, t_Trans[-1]], [0], args=(Lambda_constant, g_constant), t_eval=t_Trans)

y_Trans=sol_Trans.y
y_Bp=np.squeeze(y_Trans)
y_Bp=y_Bp[((Ntotal-N)):]
y_new=np.tile(y_Bp,NBp)
y_dif=np.squeeze(y_Trans)-y_new

### Creation of the
Ud_new=((fft(y_new)))                         # DFT input signal
Udsplit_new=fftshift(Ud_new)                             # DFT input signal zero split
fd_new=np.linspace(0,f_s,Ntotal,endpoint=False)                             # DFT frequency
fdsplit_new=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,Ntotal,endpoint=True)    # DFT frequency zero split

Visualplots.simple_plot(t_Trans,u_Trans,title=str(f'Transient signal of N = {N} times {NBp} with N total of {Ntotal}'),x_label=str('Time [s]'),y_label=str('Amplitude'))
Visualplots.maxwel_plot(t_Trans,np.squeeze(sol_Trans.y),title='ODE Maxwell transient signal')
Visualplots.maxwel_plot(t_Trans,y_new,title='ODE Maxwell repeated signal')
Visualplots.maxwel_plot(t_Trans,y_dif,title='ODE Maxwell difference between transient and repeated signal')

# Solutions of the ODE
y_multi_new=np.squeeze(sol_Trans.y)             # y output in time domain entire signal
y_multi_new=y_multi_new[((Ntotal-N)):]          # Copy of last transient period y output
u_multi_new=u_Trans[((Ntotal-N)):]              # Copy of last transient period u input
Y_multi_new=fft(y_multi_new)                    # Y output to frequency domain
U_multi_new=fft(u_multi_new)                    # U input to frequency domain or reconstructed signal
G_multi_new=Y_multi_new/U_multi_new             # FRF of the multiphase

# Magnitude plot of FRF transferfunction with multisine
plt.plot(f, rhs.DB((G_multi_new)),label="Multisine respone")
plt.plot(w,phase,label="Transfer Function")  # magnitude plot
plt.legend()
plt.xscale('log')
plt.ylabel('Phase')
plt.xlabel('Frequency (Hz) test')
plt.show()

# Phase plot of the FRF transferfunction with multisine
plt.plot(f, np.angle(G_multi_new,deg=True),label='Multisine response')
plt.plot(w,mag,label="Transfer Function")
plt.legend()
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz) test')
plt.show()