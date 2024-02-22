import numpy as np

# Functions for generating different types of signals

def chirp_exponential(f_s, N, P, frequency_limits):
    """
    Generates an exponential chirp signal.
    Parameters:
    - time: Time vector.
    - frequency_limits: Tuple of start and end frequencies.
    - f_s: Sampling frequency.
    - N: Number of samples.
    - P: Number of periods.
    """
    f_0         = f_s / N
    T           = N / f_s
    time        = np.linspace(0, T,N,endpoint=False)
    k_1, k_2    = frequency_limits
    L           = T / np.log(k_2 / k_1)
    signal      = np.sin(2 * np.pi * f_0 * k_1 * L * (np.exp(time / L) - 1))
    signal      = np.tile(signal, P)
    return signal


def chirp_linear(f_s, N, P, frequency_limits):
    """
    Generates an exponential chirp signal.
    Parameters:
    - time: Time vector.
    - frequency_limits: Tuple of start and end frequencies.
    - f_s: Sampling frequency.
    - N: Number of samples.
    - P: Number of periods.
    """
    f_0         = f_s / N
    T           = N / f_s
    time        = np.linspace(0, T,N,endpoint=False)
    k_1, k_2    = frequency_limits
    L           = np.pi*(k_2-k_1)*f_0**2
    signal      = np.sin((L*time+2*np.pi*k_1*f_0)*time)
    signal      = np.tile(signal, P)
    return signal

# Functions for normalization of signals

def normalize_amp(signal):
    """
    Normalizes the amplitude of the signal between -1 and 1.
    """
    return signal / np.max(np.abs(signal))

def normalize_rms(signal):
    """
    Normalizes the signal by its root mean square value.
    """
    rms_val = np.sqrt(np.mean(np.square(abs(signal))))
    return signal / rms_val if rms_val > 0 else signal

def normalize_stdev(signal):
    """
    Normalizes the signal by its standard deviation.
    """
    std_val = np.std(signal)
    return signal / std_val if std_val > 0 else signal

def normalization(signal, normalize='None'):
    """
    Normalizes the signal based on the specified method.
    """
    if normalize == 'Amplitude':
        return normalize_amp(signal)
    elif normalize == 'RMS':
        return normalize_rms(signal)
    elif normalize == 'STDev':
        return normalize_stdev(signal)
    elif normalize == 'None':
        return signal
    else:
        raise ValueError('Normalize must be "Amplitude", "RMS", "STDev", or "None"')

# Functions for phase generation in multisine signals

def schroeder_phase(j_range, J):
    """
    Generates Schroeder phase for multisine signals.
    """
    return np.array([-j * (j - 1) * np.pi / J for j in j_range])

def linear_phase(tau, j_range, f_s, N):
    """
    Generates linear phase for multisine signals.
    """
    return np.array([-j * tau * 2 * np.pi * f_s / N for j in j_range])

def rudin_phase(J):
    """
    Generates Rudin phase for multisine signals.
    """
    rudin_shapiro = lambda J: np.array([(-1) ** bin(n << 1 & n).count('1') for n in range(J)])
    phase         = -np.pi * rudin_shapiro(J)
    phase[phase == -np.pi] = 0
    return phase

def newman_phase(J, j1):
    """
    Generates Newman phase for multisine signals.
    """
    k     = np.arange(J) + j1
    return (np.pi * (k - 1) ** 2) / J

# Main function for generating multisine signals

def multisine(f_s, N, P, frequency_limits, A_vect=None,phase_response='Schroeder', Tau=1 , time_domain=True):
    """
    Generates a multisine signal.
    Parameters:
    - N: Number of samples.
    - f_s: Sampling frequency.
    - f_0: Base frequency.
    - frequency_limits: Tuple of start and end frequencies.
    - P: Number of periods.
    - A_vect: Amplitude vector.
    - Tau: Time constant for linear phase generation.
    - phase_response: Type of phase response ('Random', 'Schroeder', etc.).
    - time_domain: Boolean indicating if output is a function of time u(t) with lambda t.
    """
    f_0           = f_s / N
    j1, j2        = map(int, (np.floor(frequency_limits[0]), np.ceil(frequency_limits[1])))
    J             = j2 - j1
    j_range       = np.linspace(j1, j2 - 1, J, endpoint=True).astype(int)

    # Set up amplitude vector
    A             = np.ones(N) if A_vect is None else A_vect[:J]

    # Select phase response
    phase         = {
        'Schroeder': schroeder_phase(j_range, J),
        'Random'   : np.random.uniform(0, 2 * np.pi, J),
        'Linear'   : linear_phase(Tau, j_range, f_s, N),
        'Newman'   : newman_phase(J, j1),
        'Rudin'    : rudin_phase(J),
        'Zero'     : np.zeros(N)
    }.get(phase_response, ValueError('Invalid phase response'))

    # Calculate multisine signal
    u = np.zeros(N * P) if time_domain else lambda k: sum(A[j] * np.cos(j_range[j] * k * 2 * np.pi * f_0 + phase[j]) for j in range(J))
    if time_domain:
        T = (P * N) / f_s
        t = np.linspace(0, T, N * P, endpoint=False)
        for j in range(J):
            u += A[j] * np.cos(j_range[j] * t * 2 * np.pi * f_0 + phase[j])

    return u, phase

# Noise generation

def add_noise_with_snr(signal, snr_db):
    # Calculate signal power and convert from dB to linear scale
    P_signal = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10)

    # Calculate desired noise power for the given SNR
    P_noise_desired = P_signal / snr_linear

    # Generate white Gaussian noise
    noise = np.random.normal(0, 1, signal.shape)

    # Calculate the power of the generated noise
    P_noise_actual = np.mean(noise**2)

    # Scale the noise to achieve the desired noise power
    noise = noise * np.sqrt(P_noise_desired / P_noise_actual)

    return  noise


def gaussian_noise(signal, mean=0, std=1):
    noise = np.random.normal(mean, std, signal.shape)
    return noise

def add_uniform_noise(signal, low=-1, high=1):
    noise = np.random.uniform(low, high, signal.shape)
    return noise

# Windowing
def window_function(t, T, r):
    # Normalizing time array
    t_normalized = t / T

    # Window function
    window = np.zeros_like(t)
    for i, tn in enumerate(t_normalized):
        if tn <= r / 2:
            window[i] = np.cos(np.pi / r * (tn - r / 2)) ** 2
        elif r / 2 < tn < 1 - r / 2:
            window[i] = 1
        elif tn >= 1 - r / 2:
            window[i] = np.cos(np.pi / r * (tn - 1 + r / 2)) ** 2

    return window


# Utility functions

def DB(signal):
    """
    Calculates the decibel value of the signal.
    """
    return 20 * np.log10(np.abs(signal))

def rms(signal):
    """
    Calculates the root mean square of the signal.
    """
    return np.sqrt(np.mean(np.square(abs(signal))))

def crest_fac(signal):
    """
    Calculates the crest factor of the signal.
    """
    return max(abs(signal)) / rms(signal)

def power_efficiency(signal,bandlimit):
    """
    total power in the total frequency band of interest
    """
    interest = signal[int(bandlimit[0]):int(bandlimit[1])]
    P        = sum(abs(signal)**2)/2
    Pint     = sum(abs(interest)**2)
    return Pint/P

def power_loss(signal,bandlimit):
    """
    power loss in the total frequency band
    """
    interest = signal[int(bandlimit[0]):int(bandlimit[1])]
    Ploss    = min(abs(interest)**2)
    PMS      = np.mean(abs(interest)**2)

    return Ploss/PMS

def quality_factor(crest,efficiency,loss):

    return np.sqrt(crest**2/(efficiency*loss))

def calculate_snr(signal, noisy_signal):


    # Calculate the power of the signal
    P_signal = np.mean(signal**2)

    # Calculate the noise (difference between noisy signal and original signal)
    noise = noisy_signal - signal

    # Calculate the power of the noise
    P_noise = np.mean(noise**2)

    # Calculate SNR in linear scale and then convert to dB
    SNR_linear = P_signal / P_noise
    SNR_dB = 20 * np.log10(SNR_linear)

    return SNR_dB
