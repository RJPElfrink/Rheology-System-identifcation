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
N= 10                    # Number of points
f_0 = 25                 # Excitation frequency
phi= np.pi/3            # signal phase

NBp = 100                  # Number of block points
Ntrans=1000                # Number of transient points

Up=100                  # Upsampling for plots

# Calculation of time window
T=Ntrans/f_s                 # Time length
T_s = 1/f_s                 # Sampling time
t_Trans= np.linspace(0, T,Ntrans*Up,endpoint=False)        # Discrete-time vector

u_Trans=sine(2*np.pi*f_0*t_Trans+phi)

plt.plot(t_Trans, u_Trans)
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()

def gamma_dot(t):
    return sine(2*np.pi*f_0*t+phi)

# tau_dot=G*gamma_dot(t) -1/lambda *tau
def ODE_maxwell(t, tau, L,G):
    #L,G=args
    return G*gamma_dot(t) - tau/L

sol = solve_ivp(ODE_maxwell, [0, t_Trans[-1]], [0], args=(1.5, 2.5), t_eval=t_Trans)


plt.plot(np.squeeze(t_Trans), np.squeeze(sol.y))
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell')
plt.show()

y_Trans=sol.y
y_Bp=np.squeeze(y_Trans)
y_Bp=y_Bp[((Ntrans-NBp))*Up:]
y_new=np.tile(y_Bp,N)

plt.plot(t_Trans, y_new)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell')
plt.show()

y_dif=np.squeeze(y_Trans)-y_new

plt.plot(t_Trans, y_dif)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=False)
plt.title('ODE Maxwell')
plt.show()
