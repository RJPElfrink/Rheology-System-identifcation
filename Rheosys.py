import numpy as np

# Functions for generating different types of signals

def chirp_exponential(time, frequency_limits, f_s, N, P):
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
    k_1, k_2    = frequency_limits
    L           = T / np.log(k_2 / k_1)
    signal      = np.sin(2 * np.pi * f_0 * k_1 * L * (np.exp(time / L) - 1))
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

def multisine(N, f_s, f_0, frequency_limits, P=1, A_vect=None,phase_response='Schroeder', Tau=1 , time_domain=True):
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


import numpy as np

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

# Example usage
# Assuming you have a 'signal' array and a 'noisy_signal' array
# signal = np.array([...])  # Your original signal
# noisy_signal = np.array([...])  # Your signal with added noise
# snr = calculate_snr(signal, noisy_signal)
# print(f"The SNR is: {snr} dB")

def gaussian_noise(signal, mean=0, std=1):
    noise = np.random.normal(mean, std, signal.shape)
    return noise

def add_uniform_noise(signal, low=-1, high=1):
    noise = np.random.uniform(low, high, signal.shape)
    return noise

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
