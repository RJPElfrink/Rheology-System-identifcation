import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from random import random
from scipy.fft import fft,fftshift,ifft,ifftshift


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...
def sine (period):
    return np.sin(period)

def cos (period):
    return np.cos(period)


def DB(arr):
    return 20 * np.log10(np.abs(arr))

def rms(arr):
    # Return the root mean square of all the elements of arr, flattened out.
    rms = np.sqrt(np.mean(np.square(abs(arr))))
    return rms

def crest_fac(frequency):
    return (max(abs(frequency)))/rms(frequency)


def multisine_advanced(frequency_limits, f_s, N, **kwargs):

    def schroeder_phases(N,index_vector, NN,  magnitude, p1,lower_lim,upper_lim):
        phase = np.zeros(N)
        phase[1] = p1

        for nn in range(2, NN + 1):
            ll = np.arange(1, nn)
            phase[index_vector[nn - 1] - 1] = phase[1] - 2 * np.pi * np.sum((nn - ll) * magnitude[index_vector[ll] - 1])


        phase = np.zeros(N)
        phase[1] = p1

        #zero array of the total amount of points, the first values equal the phase frequencies
        #design of the phase: sigma= -j(j-1)*pi/F
        # can not yet figure out to implement the p1 phase and not exceed the amount of frequncies analyzed, cheated this by settin [NN]=0
        phase[1:NN+1] = -np.arange(lower_lim, upper_lim+1) * np.arange(lower_lim-1, upper_lim) * np.pi / NN
        phase[NN]=0

        return phase

    def force_fft_symmetry(X):
        Y = X.copy()
        X_start_flipped = np.flipud(X[1:np.floor_divide(len(X), 2) + 1])
        Y[np.ceil(len(X) / 2).astype(int):] = np.real(X_start_flipped) - 1j * np.imag(X_start_flipped)
        return Y

    f_0 = f_s / N
    T=N/f_s
    lower_lim=round(frequency_limits[0] / f_0) + 1
    upper_lim=round(frequency_limits[1] / f_0) + 1
    f_index = np.arange(1, lower_lim), np.arange(1, upper_lim)

    if np.any(f_index[1] > N / 2):
        raise ValueError('Frequency limits must be beneath Nyquist.')

    # Freqency list of the defined amount of frequencies, with length NN
    index_vector = np.arange(f_index[0][0], f_index[1][-1] + 1)
    NN = len(index_vector)


    mag = kwargs.get('MagnitudeResponse', np.zeros(N))
    phase_response = kwargs.get('PhaseResponse', 'Schroeder')
    start_at_zero = kwargs.get('StartAtZero', True)
    normalise = kwargs.get('Normalise', True)
    time_domain = kwargs.get('TimeDomain', False)
    initial_phase = kwargs.get('InitialPhase', 0)

    # Magnitude vector of the amount of points (N) in one decade, where the first points are non zero
    if not np.any(mag):
        mag = np.zeros(N)
        mag[index_vector - 1] = 1.0 / len(index_vector)

    else:
        if len(mag) != N:
            raise ValueError('Magnitude response must be the same length as the desired signal.')

        mag = mag ** 2

        full_indices = np.arange(1, N + 1)
        zero_value_indices = np.setdiff1d(full_indices, index_vector)

        if np.any(mag[zero_value_indices]):
            print('Non-zero magnitude values present outside of frequency limits.')
            mag[zero_value_indices] = np.zeros_like(zero_value_indices)

        if np.sum(mag[index_vector - 1]) != 1:
            mag[index_vector - 1] = mag[index_vector - 1] / np.sum(mag[index_vector - 1])

    if isinstance(phase_response, str):
        if phase_response == 'Schroeder':
            phase = schroeder_phases(N, index_vector, NN, mag, initial_phase,lower_lim,upper_lim)
        elif phase_response == 'ZeroPhase':
            phase = np.zeros(N)
        elif phase_response == 'Random':
            phase = np.random.randn(N)
        else:
            raise ValueError('Phase Response string must be Schroeder, ZeroPhase, or NormalDistribution.')
    else:
        phase = phase_response
        if np.any(phase.shape != (1, N)):
            raise ValueError('Phase response must be 1 x N.')

    # switch between time domain or frequencie domain
    if time_domain:
        # if calculating in time domain
        y = np.zeros(N)
        t=np.linspace(0,T,N,endpoint=False)
        for nn in range(NN):
            mm = index_vector[nn]
            y += np.sqrt(mag[mm - 1] / 2) * np.sin(2 * np.pi * f_0 * (mm - 1) * t + phase[mm - 1])
            #y += (mag[mm - 1]) * np.sin(2 * np.pi * f_0 * (mm - 1) * t + phase[mm - 1])
        print('max', np.max(y), 'min', np.min(y),'average',np.average(y))
    else:
        #Y = np.sqrt(mag / 2) * np.exp(1j * phase)
        Y = (mag) * np.exp(1j * phase)
        #y = ifft(force_fft_symmetry(Y)) * (N / 2)
        #print('y',np.size(y),y,'y')
        y = 2*np.real(ifft((Y)))

        #print('y',np.size(y),y,'y')
        #plt.plot(y)
        #plt.plot(Y)
        #plt.show()


    if normalise:
        y = y / np.max(np.abs(y))

    if start_at_zero:
        y_sign = y > 0
        zero_inds = np.where(y_sign[:-1] != y_sign[1:])[0]

        zero_grad = np.abs(y[zero_inds] - y[zero_inds + 1])
        min_ind = np.argmin(zero_grad)

        y_wrapped = np.concatenate((y[zero_inds[min_ind]:], y[:zero_inds[min_ind]]))
        y = y_wrapped

    return y

def run_multi_advanced():
    # Define the parameters
    f_s = 4800              # Sampling frequency
    N = 4800                 # Number of samples (for 1 second)

    # Generate multisine between 1 Hz and 2 kHz
    y = multisine_advanced([1, 500], f_s, N,PhaseResponse='Schroeder',TimeDomain=False,Normalise=True,InitialPhase=0,StartAtZero=True)


    T=N/f_s
    # Plot
    t = np.linspace(0,T,N,endpoint=False)
    f = np.linspace(0,N,f_s)
    print(np.max(f),'f', np.max(t) , 't ')

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')

    plt.subplot(212)
    plt.semilogx(f, 20 * np.log10(np.abs(fft(y))))
    plt.xlim([1, f_s / 2])
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


    print('Crest factor: {0:.2f} dB'.format(20*np.log10(crest_fac(y))))
    print('Crest factor: {0:.2f} '.format(crest_fac(y)))


    return plt.show()

#run_multi_advanced()