import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from math import floor


def u_continuous(time):
    u = np.sin(time)
    return u

def u_sampling(excitation_frequency, sample_frequency, sample_time):
    # f0 = excitation wave frequency [Hz]
    f0 = excitation_frequency
    # Fs = sampling frequency [Hz]
    Fs = sample_frequency
    T = sample_time                               # Record window duration [s]

    dt = 1/Fs                                   # Sampling period [s]
    t = np.arange(0, T+dt, dt)   # Time vector for sampling [s]

    # generate samples at the specified times
    u_sampled = np.sin(2*np.pi*f0*t)  # [signal units]

    return (u_sampled, t)

def DAC_0(u_discrete, sample_time, time_range):
    u_test = []
    for i in range(len(time_range)):
        x = int(time_range[i]//sample_time)
        u_test.append(u_discrete[x])
    return np.array(u_test)


#define excitation varibales
f_excitation = 15
f_sample = 8
T_sample = 1/f_sample
T_total = 2
t2 = np.arange(0, T_total+(T_sample/4), T_sample/4)

# run the function
u_d, time = u_sampling(f_excitation, f_sample, T_total)
u_t = DAC_0(u_d, T_sample, t2)

# Alternative calculation of u_t, by use of scipy interpolation, kind is 0,2 and all odd numbers
u_t_interp=interp1d(time,u_d,kind=0)


plt.plot(time, u_d, "o", t2, u_t, "-")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')
plt.axis('tight')
plt.show()
