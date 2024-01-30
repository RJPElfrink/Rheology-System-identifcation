import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from random import random
from scipy.fft import fft


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...



def chirp_exponential(time,frequencie_limits,f_s,N,P):

    f_0  = f_s/N
    T = N/f_s                                        # Time length

    k_1  = frequencie_limits[0]
    k_2  = frequencie_limits[1]

    L    = T/np.log(k_2/k_1)
    signal= np.zeros_like(time)

    for ii,t in enumerate(time):
        signal[ii]=np.sin(2*np.pi*f_0*k_1*L*(np.exp((t/L))-1))

    signal=np.tile(signal,P)

    return signal


def fft_normalization(signal, ouput, **kwargs):
    input=fft(signal)
    output=fft(ouput)

    normalize = kwargs.get('normalize', 'None')                 # Default normalization is None
    # Normalize the signal spectrum
    if isinstance(normalize, str):

        if normalize == 'Amplitude':        #Amplitude normalization between -1 & 1
            U = normalize_amp(input)
            Y = normalize_amp(output)
            print('check amp')
        elif normalize == 'RMS':
            U = normalize_rms(input)
            Y = normalize_rms(output)
        elif normalize == 'STDev':
            U = normalize_stdev(input)
            Y = normalize_stdev(output)
        elif normalize == 'None':           # No normalization, output of pure amplification
            print('check none')
            U=input
            Y=output
            G=Y/U
            return U,Y,G
    else:
            raise ValueError('Normalize must be Amplitude, RMS, STDev or None')
    G=Y/U
    print('check done')
    return U, Y, G

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
def normalize_amp(signal):
    signal = signal / np.max(np.abs(signal))
    return signal

# Normalization of the signals amlitude by RMS devision
def normalize_rms(signal):
    current_rms = rms(signal)
    if current_rms > 0:
        signal = signal / current_rms
    return signal

# Normalization of the signals amplitude by standard deviation devision
def normalize_stdev(signal):
    current_stdev = np.std(signal)
    if current_stdev > 0:
        signal = signal / current_stdev
    return signal

# Multisine phase generation of phi=-j(j-1)*pi/F
# Where F is the total amount of frequencies crest of 1.89
def schroeder_phase(j_range,J):
    phase=np.zeros(J)
    for i in range(len(j_range)):
        phase[i]= -1*(j_range[i])*(j_range[i]-1)*np.pi/J
    return phase

# Multisine linear phase generation crest of 28.0
def linear_phase(tau,j_range,f_s,N):
    phase=np.zeros(len(j_range))
    #phi=-Tau*2*pi*f_s*(j/N)
    for i in range(len(j_range)):
        phase[i]=-j_range[i]*tau*2*np.pi*f_s/N
    return phase

# Phase generation which looks like an inverse schroeder with crest factor of 1.90
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

#Sweep-like phase generation, crest of 2.65
def newman_phase(J,j1):

    k = np.arange(J) + j1
    phase = (np.pi*(k-1)**2)/J
    return phase

# Generation of the multisine signal with input N,f_s, [j1,j2] and phase_response='schroeder'
def multisine(N, f_s,f_0, frequency_limits, P=1 , A_vect=None, Tau=1, **kwargs):

    j1=int(np.floor(frequency_limits[0]))                 # Starting frequency multisine
    j2=int(np.ceil(frequency_limits[1]))                  # Ending frequency multisine
    J=int(j2-j1)                                     # Total frequencies multisine
    j_range=np.ndarray.astype(np.linspace(j1,j2-1,J,endpoint=True),int)             # Range of j1 to j2

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
    #if Tau is None:
    #    tau = 1
    #else:
    #    tau = Tau


    # Select phase response
    if isinstance(phase_response, str):
            if phase_response == 'Schroeder':
                phase = schroeder_phase(j_range,J)
            elif phase_response == 'Random':
                phase = np.random.uniform(0,2*np.pi,J)
            elif phase_response == 'Linear':
                phase = linear_phase(Tau,j_range,f_s,N)
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

    # Calculation of the multisine with corresponding phase
    # Output u is not a function but a signal array
    if time_domain:

        u = np.zeros(N*P)                                    # Signal vector
        T = (P*N)/f_s                                        # Time length
        t= np.linspace(0, T,N*P,endpoint=False)            # Time vector

        for j in range(len(j_range)):
            u+= A[j]*np.cos(j_range[j]*t*2*np.pi*f_0 + phase[j])

    # Output of u is a function dependend on t, u(t)
    else:
        #def u_of_t(x):
           # signal = np.zeros_like(x)
          #  for j in range(J):
         #       signal += A[j] * np.sin(j_range[j] * x * 2 * np.pi * f_0 + phase[j])
        #    return signal

        u = lambda k: sum(A[j] * np.cos(j_range[j] * k * 2 * np.pi * f_0 + phase[j]) for j in range(len(j_range)))
        return u,phase

    # Normalize the signal spectrum
    if isinstance(normalize, str):

        if normalize == 'Amplitude':        #Amplitude normalization between -1 & 1
            u = normalize_amp(u)
        elif normalize == 'RMS':
            u = normalize_rms(u)
        elif normalize == 'STDev':
            u = normalize_stdev(u)
        elif normalize == 'None':           # No normalization, output of pure amplification
            return u,phase

        #elif normalize == 'Power':
            #normalize = rhs.normalize_power(u)
    else:
                raise ValueError('Normalize must be Amplitude, RMS, STDev or None')

    return u,phase



