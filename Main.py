import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft,fftshift

def sine (period):
    return np.sin(period)

# To avoid aliasing and leakage, it is advised to determine the N (number of points in the window) and f_s (sampling frequency)
# From that calculate
# f_0=f_s/N = 1/T = 1/T_s*N
# line number k=N*f/f_s
# AVOID leakage: f_s/f_0 must be an integer

#define excitation varibales
f_s = 100                 # Sample frequency
N= 100                    # Number of points
f_0 = 20                 # Excitation frequency

phi= np.pi/3            # signal phase
Up=100                  # Upsampling for plots

# Calculation of time window
T=N/f_s                 # Time length
T_s = 1/f_s             # Sampling time
t_DT= np.arange(0, T,1/f_s)        # Discrete-time vector
t_CT= np.arange(0, T, 1/(f_s*Up))  # Continuous-time vector


u_DT=sine(2*np.pi*f_0*t_DT+phi)
u_CT=sine(2*np.pi*f_0*t_CT+phi)

plt.plot(t_DT, u_DT, "o", t_CT, u_CT, "-")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()

### Creation of the
Ud=(np.abs(fft(u_DT)))/N                         # DFT input signal
Udsplit=fftshift(Ud)                             # DFT input signal zero split
dB=-20*np.log(Udsplit)
fd=np.linspace(0,f_s,N,endpoint=False)                             # DFT frequency
fdsplit=np.linspace(-np.floor(N/2),-np.floor(N/2)+N,N,endpoint=False)    # DFT frequency zero split
Lines=np.arange(0,N,1)                      # Line numbers after DFT

fig0, (ax0) = plt.subplots(1, 1, layout='constrained')
ax0.stem([-f_0,f_0],[0.5,0.5],linefmt='blue', markerfmt='D',label='Sample frequency $kf_0=kf_s/N$')
ax0.stem(fdsplit,Udsplit,linefmt='red', markerfmt='D',label='DFT input frequency')
fig0.suptitle('Amplitude spectrum of DFT and FT should coincide with the $f_0$ frequency')
fig0.supxlabel('f[Hz]')
fig0.supylabel('$|U_{DFT}|$')
ax0.legend()
plt.show()

# Calculation of u_t, by use of scipy interpolation, kind is 0,2 and all odd numbers
u_interp=interp1d(t_DT,u_DT,kind=0)
u_t_interp = u_interp(t_CT)

plt.plot(t_DT, u_DT, "o", t_CT, u_t_interp, "-")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()

def gamma_dot(t):
    return u_interp(t)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    #L,G=args
    return G*gamma_dot(t) - tau/L

sol = solve_ivp(ODE_maxwell, [0, T], [0], args=(1.5, 2.5), t_eval=t_DT)

plt.plot(np.squeeze(t_DT), np.squeeze(sol.y))
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell')
plt.show()