import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft,fftshift

def sine (period):
    return np.sin(period)

def DB (arr):
    ref = 1
    decibel=[]
    for i in arr:
        if i!=0:
            decibel.append(20 * np.log10(abs(i) / ref))
        else:
            decibel.append(-60)

    return decibel

# To avoid aliasing and leakage, it is advised to determine the N (number of points in the window) and f_s (sampling frequency)
# From that calculate
# f_0=f_s/N = 1/T = 1/T_s*N
# line number k=N*f/f_s
# AVOID leakage: f_s/f_0 must be an integer

#define excitation varibales
f_s =1000                   # Sample frequency
N= 1000                    # Number of points

phi= 0            # signal phase
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

u_DT=sine(2*np.pi*f_0*t_DT+phi)
u_CT=sine(2*np.pi*f_0*t_CT+phi)


### Calculation of fourier transform in frequency distribution
Ud=(np.abs(fft(u_DT)))/N                         # DFT input signal
Udsplit=fftshift(Ud)                             # DFT input signal zero split
fd=np.linspace(0,f_s,N,endpoint=False)                             # DFT frequency
fdsplit=np.linspace(-np.floor(f_s/2),-np.floor(f_s/2)+f_s,N,endpoint=False)    # DFT frequency zero split

#Plot with sample points and continous signal
plt.plot(t_DT, u_DT, ".", t_CT, u_CT, "-")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()

#Stemplot of split frequency disribution
plt.stem([-f_0,f_0],[2*max(Udsplit),2*max(Udsplit)],linefmt='blue', markerfmt='D',label='Sample frequency $kf_0=kf_s/N$')
plt.stem(fdsplit,Udsplit,linefmt='red', markerfmt='d',label='DFT input frequency')
plt.title('Amplitude spectrum of DFT and FT should coincide with the $f_0$ frequency')
plt.xlabel('f[Hz]')
plt.ylabel('$|U_{DFT}|$')
plt.yscale('log')
plt.legend()
plt.show()

#Lineplot over the frequency distribuation in decibels
plt.plot(fd,DB(Ud),'-')
plt.title('The position of the FFT components on the frequency axis in Hz ')
plt.xlabel('f[Hz]')
plt.ylabel('$dB$')
plt.legend()
plt.show()

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

sol_Block = solve_ivp(ODE_maxwell, [0, t_DT[-1]], [0], args=(1.5, 2.5), t_eval=t_DT)

# Plot of the reconstructed signal between sample points
plt.plot(t_DT, u_DT, ".")
plt.plot(t_CT, U_DT_int, "-")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.title(f'Reconstructed signal between sample points with interpolation kind of {kind}')
plt.axis('tight')
plt.show()

# Plot of the solution of the maxwell differential
plt.plot(np.squeeze(t_DT), np.squeeze(sol_Block.y))
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell')
plt.show()

# Calculations of the extended signal to remove the transient part
# and repeat the signal to the full time window of N*NBp
Ttotal=Ntotal/f_s                 # Time length
T_s = 1/f_s                 # Sampling time

t_Trans= np.linspace(0, Ttotal,Ntotal,endpoint=False)        # Transient-time vector
u_Trans=sine(2*np.pi*f_0*t_Trans + phi)


def gamma_dot(t):
    return sine(2*np.pi*f_0*t+phi)

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

# Plot of total time signal
plt.plot(t_Trans, u_Trans)
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.title(f'Transient signal of N = {N} times {NBp} with N total of {Ntotal}')
plt.show()

plt.plot(np.squeeze(t_Trans), np.squeeze(sol_Trans.y))
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell transient transient signal')
plt.show()

plt.plot(t_Trans, y_new)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title(f'ODE Maxwell repeated signal ')
plt.show()

plt.plot(t_Trans, y_dif)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell difference beteween transient and repeated signal')
plt.show()

plt.plot(fd_new,DB(Ud_new),'d')
plt.title('The position of the FFT components on the frequency axis in Hz ')
plt.xlabel('f[Hz]')
plt.ylabel('$|U_{DFT}|$')
plt.legend()
plt.show()