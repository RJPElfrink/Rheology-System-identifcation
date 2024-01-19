import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import Rheosys
import Rheosys as rhs
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft,fftshift,ifft,ifftshift,rfft,rfftfreq,irfft,fftfreq
from scipy import signal
import Rheosys as rhs
import Visualplots

#random.seed(10)

f_s =480                                         # Sample frequency
N= 480                                           # Number of points

Lambda_constant=1.5                              # Value for lambda n/g
g_constant=2.5                                   # Value for spring constant g

phi= 0                                           # starting phase
kind=0                                           # interpolation kind

NBp = 6                                          # Number of block points
Ntotal=N*NBp                                     # Number of transient points

kind=0                                           # interpolation kind
Up=100                                           # Upsampling for plots

# Calculation of time window
f_0 = 25*f_s/N                                  # Excitation frequency
T = N/f_s                                        # Time length
t= np.linspace(0, T,N,endpoint=False)            # Time vector
t_CT= np.linspace(0, t[-1], N*Up,endpoint=True)  # Over sampling continuous-time vector
f=np.linspace(0,f_s,N,endpoint=False)            # Frequency range


u_single=np.sin(2*np.pi*f_0*t+phi)
u_CT=np.sin(2*np.pi*f_0*t_CT+phi)

### Calculation of fourier transform in frequency distribution
Ud=(np.abs(fft(u_single)))                        # DFT input signal
Udsplit=fftshift(Ud)                             # DFT input signal zero split
fd=np.linspace(0,f_s,N,endpoint=False)                             # DFT frequency
fdsplit=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,N,endpoint=False)    # DFT frequency zero split

# Reconstruction of signal between sample points, use of scipy interpolation, kind is 0,2 and all odd numbers
u_single_t=interp1d(t,u_single,kind=kind)
u_single_int = u_single_t(t)

# Solving of the maxwell differentail equation
def gamma_dot(t):
    return u_single_t(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    #L,G=args
    return G*gamma_dot(t) - tau/L

sol_Block = solve_ivp(ODE_maxwell, [0, t[-1]], [0], args=(Lambda_constant,g_constant), t_eval=t)

y_block=np.squeeze(sol_Block.y)
Y_block=fft(y_block)
U_block=fft(u_single_int)
G_block=Y_block/U_block

#plt.plot(fd,G_block)
#plt.show()

Visualplots.input_signal_sample(t,u_single,t_CT,u_CT)
Visualplots.input_stem_split_freqency(f_0,Udsplit,fdsplit)
Visualplots.input_line_frequency(fd,Ud)
Visualplots.input_reconstructer_sampling(t,u_single,t,u_single_int)
Visualplots.maxwel_plot(t,np.squeeze(sol_Block.y))

# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*NBp
Ttotal=Ntotal/f_s                 # Time length
T_s = 1/f_s                 # Sampling time

t_Trans= np.linspace(0, Ttotal,Ntotal,endpoint=False)        # Transient-time vector
u_Trans=np.sin(2*np.pi*f_0*t_Trans + phi)


def gamma_dot(t):
    return np.sin(2*np.pi*f_0*t+phi)

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
Ud_new=(np.abs(fft(y_new)))/Ntotal                         # DFT input signal
Udsplit_new=fftshift(Ud_new)                             # DFT input signal zero split
fd_new=np.linspace(0,f_s,Ntotal,endpoint=False)                             # DFT frequency
fdsplit_new=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,Ntotal,endpoint=True)    # DFT frequency zero split

Visualplots.simple_plot(t_Trans,u_Trans,title=str(f'Transient signal of N = {N} times {NBp} with N total of {Ntotal}'),x_label=str('Time [s]'),y_label=str('Amplitude'))
Visualplots.maxwel_plot(t_Trans,np.squeeze(sol_Trans.y),title='ODE Maxwell transient signal')
Visualplots.maxwel_plot(t_Trans,y_new,title='ODE Maxwell repeated signal')
Visualplots.maxwel_plot(t_Trans,y_dif,title='ODE Maxwell difference between transient and repeated signal')

