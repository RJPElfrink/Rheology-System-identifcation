import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft,fftshift
import rheosys as rhs
import Visualplots


# To avoid aliasing and leakage, it is advised to determine the N (number of points in the window) and f_s (sampling frequency)
# From that calculate
# f_0=f_s/N = 1/T = 1/T_s*N
# line number k=N*f/f_s
# AVOID leakage: f_s/f_0 must be an integer

#define excitation varibales
f_s =1000                   # Sample frequency
N= 1000                    # Number of points


phi= 0                # signal phase
kind=0              #interpolation kind

NBp = 6                  # Number of block points
Ntotal=N*NBp             # Number of transient points

Up=100                  # Upsampling for plots

# Calculation of time window
f_0 = 25*f_s/N                 # Excitation frequency
k_value= (N*f_0)/f_s
T = N/f_s                 # Time length
T_s = 1/f_s             # Sampling time
t_DT= np.linspace(0, T,N,endpoint=False)        # Discrete-time vector
t_CT= np.linspace(0, t_DT[-1], N*Up,endpoint=True)  # Continuous-time vector

u_DT=rhs.sine(2*np.pi*f_0*t_DT+phi)
u_CT=rhs.sine(2*np.pi*f_0*t_CT+phi)

crest=rhs.crest_fac(u_DT)
### Calculation of fourier transform in frequency distribution
Ud=(np.abs(fft(u_DT)))/N                         # DFT input signal
Udsplit=fftshift(Ud)                             # DFT input signal zero split
fd=np.linspace(0,f_s,N,endpoint=False)                             # DFT frequency
fdsplit=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,N,endpoint=False)    # DFT frequency zero split

# Reconstruction of signal between sample points, use of scipy interpolation, kind is 0,2 and all odd numbers
u_interp=interp1d(t_DT,u_DT,kind=kind)
U_DT_int = u_interp(t_CT)

# Solving of the maxwell differentail equation
def gamma_dot(t):
    return u_interp(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    #L,G=args
    return G*gamma_dot(t) - tau/L

sol_Block = solve_ivp(ODE_maxwell, [0, t_DT[-1]], [0], args=(1.5,2.5), t_eval=t_DT)

Visualplots.input_signal_sample(t_DT,u_DT,t_CT,u_CT)
Visualplots.input_stem_split_freqency(f_0,Udsplit,fdsplit)
Visualplots.input_line_frequency(fd,Ud)
Visualplots.input_reconstructer_sampling(t_DT,u_DT,t_CT,U_DT_int)
Visualplots.maxwel_plot(t_DT,np.squeeze(sol_Block.y))

# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*NBp
Ttotal=Ntotal/f_s                 # Time length
T_s = 1/f_s                 # Sampling time

t_Trans= np.linspace(0, Ttotal,Ntotal,endpoint=False)        # Transient-time vector
u_Trans=rhs.sine(2*np.pi*f_0*t_Trans + phi)


def gamma_dot(t):
    return rhs.sine(2*np.pi*f_0*t+phi)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    #L,G=args
    return G*gamma_dot(t) - tau/L

sol_Trans = solve_ivp(ODE_maxwell, [0, t_Trans[-1]], [0], args=(1.5, 2.5), t_eval=t_Trans)

y_Trans=sol_Trans.y
y_Bp=np.squeeze(y_Trans)
y_Bp=y_Bp[((Ntotal-N)):]
y_new=np.tile(y_Bp,NBp)
y_dif=np.squeeze(y_Trans)-y_new

### Creation of the
Ud_new=(np.abs(fft(y_new)))/Ntotal                         # DFT input signal
Udsplit_new=fftshift(Ud_new)                             # DFT input signal zero split
fd_new=np.linspace(0,f_s,Ntotal,endpoint=False)                             # DFT frequency
fdsplit_new=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,Ntotal,endpoint=True)    # DFT frequency zero split

Visualplots.simple_plot(t_Trans,u_Trans,title=str(f'Transient signal of N = {N} times {NBp} with N total of {Ntotal}'),x_label=str('Time [s]'),y_label=str('Amplitude'))
Visualplots.maxwel_plot(t_Trans,np.squeeze(sol_Trans.y),title='ODE Maxwell transient signal')
Visualplots.maxwel_plot(t_Trans,y_new,title='ODE Maxwell repeated signal')
Visualplots.maxwel_plot(t_Trans,y_dif,title='ODE Maxwell difference between transient and repeated signal')
#Visualplots.input_line_frequency(fd_new,Ud_new)