import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
import rheosys as rhs


def u_continuous(time):
    u=np.sin(time)
    return u



def u_sampling (excitation_frequency,sample_frequency,sample_time):
    f0=excitation_frequency                     # f0 = excitation wave frequency [Hz]
    Fs=sample_frequency                         # Fs = sampling frequency [Hz]
    T=sample_time                               # Record window duration [s]

    dt = 1/Fs                                   # Sampling period [s]
    t = np.arange(0,T+dt,dt)   # Time vector for sampling [s]

    # generate samples at the specified times
    u_sampled = np.sin(2*np.pi*f0*t); # [signal units]

    return (t,u_sampled)

#run the function
u,time= u_sampling(10,8,2)

plt.plot(u,time)
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()
