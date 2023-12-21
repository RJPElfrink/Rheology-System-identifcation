import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from random import random
from scipy.fft import fft,fftshift,ifft,ifftshift,irfft,rfft,rfftfreq


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...
def sine (period):
    return np.sin(period)

def cos (period):
    return np.cos(period)

def DB(arr):
    return 20 * np.log10(np.abs(arr))

# Return the root mean square of all the elements in the array, flattened out
def rms(arr):
    rms = np.sqrt(np.mean(np.square(abs(arr))))
    return rms

#Calculation of the crest factor of the signal
def crest_fac(frequency):
    return (max(abs(frequency)))/rms(frequency)

# Normalization of the signals amlitude, between -1 and 1
def normalize_amp(y):
    y = y / np.max(np.abs(y))
    return y

# Multisine phase generation of phi=-j(j-1)*pi/F
# Where F is the total amount of frequencies
def schroeder_phase(j_range,J):
    phase=[]
    for i in range(len(j_range)):
        x=-1*(j_range[i])*(j_range[i-1])
        phase = np.append(phase, (x*np.pi)/J)

    phase[0]=0

    return phase

# Multisine phase generation of phi=-Tau*2*pi*f_s*(j/N)
def linear_phase(tau,j_range,f_0,N):
    phase=np.zeros(len(j_range))

    for i in range(len(j_range)):
        phase[i]=-j_range[i]/N*tau*2*np.pi*f_0
    return phase

# Multisine phase generation Noise-like, with crest factor under 6 dB when N is a power of 2.
def rudin_phase(J):

    def rudinshapiro(J):
        #Return first N terms of Rudin-Shapiro sequence.
        def hamming(x):

            return bin(x).count('1')

        out = np.empty(J, dtype=int)
        for n in range(J):
            b = hamming(n << 1 & n)
            a = (-1)**b
            out[n] = a

        return out


    phase = -np.pi*rudinshapiro(J)
    phase[phase == -np.pi] = 0

    return phase

def newman_phase(J,j1):
    #Sweep-like phase,

    k = np.arange(J) + j1
    phase = (np.pi*(k-1)**2)/J
    return phase

# Generation of the multisine signal with input N,f_s, [j1,j2] and phase_response='schroeder'
def multisine(N, f_s,f_0, frequency_limits, A_vect=None, Tau=None, **kwargs):

    #f_0 = f_s/N                                      # Excitation frequency
    T = N/f_s                                        # Time length
    u = np.zeros(N)                                  # Signal vector

    j1=np.floor(frequency_limits[0])                 # Starting frequency multisine
    j2=np.ceil(frequency_limits[1])                  # Ending frequency multisine
    J=int(j2-j1)                                     # Total frequencies multisine
    j_range=np.ndarray.astype(np.linspace(j1,j2-1,J,endpoint=False),int)                            # Range of j1 to j2
    j_range_split =np.ndarray.astype(np.concatenate((np.arange(-j2+1, 0), np.arange(j1, j2))),int)

    #Kwargs defenitions
    phase_response = kwargs.get('phase_response', 'Random')     # Default phase is Random
    normalize = kwargs.get('normalize', 'None')                 # Default normalization is None
    time_domain = kwargs.get('time_domain', True)               # Default output without t function


    # If args is empty, set A_vect to np.ones(N), A is an amplification vector for the multisine
    if A_vect is None:
        A = np.ones(N)
    elif np.size(A_vect) < J:
        raise ValueError(f'The size of A must be at least equal to {J} (j2-j1)')
    else:
        A = A_vect[:J]

    # If args is empty, set A_vect to np.ones(N), A is an amplification vector for the multisine
    if Tau is None:
        tau = 1
    else:
        tau = Tau


    # Select phase response
    if isinstance(phase_response, str):
            if phase_response == 'Schroeder':
                phase = schroeder_phase(j_range,J)
            elif phase_response == 'Random':
                phase = np.random.uniform(0,2*np.pi,J)
            elif phase_response == 'Linear':
                phase = linear_phase(tau,j_range,f_s,N)
            elif phase_response == 'Newman':
                phase = newman_phase(J,j1)
            elif phase_response == 'Rudin':
                phase = rudin_phase(J)
            elif phase_response == 'Zero':
                phase = np.zeros(N)

            else:
                raise ValueError('Phase Response must be Random, Schroeder, Zero, Rudin, Newman or Linear.')
    else:
        phase = phase_response

    # Output u is not a function but a signal array
    if time_domain:
        t= np.linspace(0, T,N,endpoint=False)            # Time vector
        for j in j_range:

            u+= A[j]*np.sin(j*t*2*np.pi*f_0 + phase[j])

    # Output of u is a function dependend on t, u(t)
    else:
        u = lambda t: sum(A[j] * np.sin(j * t * 2 * np.pi * f_0 + phase[j]) for j in j_range)
        return u

    # Normalize the signal spectrum
    if isinstance(normalize, str):

        if normalize == 'Amplitude':        #Amplitude normalization between -1 & 1
            u = normalize_amp(u)
        elif normalize == 'None':           # No normalization, output of pure amplification
            return u

        #elif normalize == 'Power':
            #normalize = rhs.normalize_power(u)
    else:
                raise ValueError('Normalize must be Amplitude or None')

    return u



