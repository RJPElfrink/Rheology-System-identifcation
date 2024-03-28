import numpy as np
from scipy.io import savemat
import matlab.engine
from scipy.fft import fft,ifft

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
    #signal      = np.tile(signal, P)

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

def normalize_amp(signal,maximum=1,x_norm=0):
    """
    Normalizes the amplitude of the signal between -1 and 1.
    """
    if x_norm==0:
        x_norm=np.max(np.abs(signal))
    normalsignal= (signal / x_norm)*maximum

    return normalsignal,x_norm

def normalize_rms(signal,maximum=1,x_norm=0):
    """
    Normalizes the signals power by its root mean square value.
    """
    if x_norm==0:
        x_norm = np.sqrt(np.mean(np.square(abs(signal))))
    normalsignal= (signal / x_norm)*maximum

    return normalsignal,x_norm

def normalize_stdev(signal,maximum=1,x_norm=0):
    """
    Normalizes the signal power by its standard deviation.
    """
    if x_norm==0:
        x_norm = np.std(signal)

    print("x_norm", x_norm)
    normalsignal= (signal / x_norm)*maximum

    return normalsignal,x_norm

def normalization(signal, normalize ,maximum ,x_norm):
    """
    Normalizes the signal based on the specified method.
    """
    print("test")
    if normalize == 'Amplitude':
        return normalize_amp(signal,maximum,x_norm)
    elif normalize == 'RMS':
        return normalize_rms(signal,maximum,x_norm)
    elif normalize == 'STDev':
        return normalize_stdev(signal,maximum,x_norm)
    elif normalize == 'None':
        return signal,x_norm
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

def multisine(f_s, N, frequency_limits, A_vect=None,phase_response='Schroeder', Tau=1 , time_domain=True):
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
    #u = np.zeros(N * P) if time_domain else lambda k: sum(A[j] * np.cos(j_range[j] * k * 2 * np.pi * f_0 + phase[j]) for j in range(J))
    u = np.zeros(N) if time_domain else lambda k: sum(A[j] * np.cos(j_range[j] * k * 2 * np.pi * f_0 + phase[j]) for j in range(J))
    if time_domain:
        T = (N) / f_s
        #T = (P * N) / f_s
        t = np.linspace(0, T, N , endpoint=False)
        #t = np.linspace(0, T, N * P, endpoint=False)
        for j in range(J):
            u += A[j] * np.cos(j_range[j] * t * 2 * np.pi * f_0 + phase[j])

    return u, phase

def crest_optimization(signal,Clip_value=0.9,R=100,rms_value=1,variable=False):
    """
    Optimizes the signal u by applying a clipping algorithm to minimize the crest factor.

    Parameters:
    u (np.ndarray): The input signal to be optimized.
    Clip (float): The clipping level relative to the maximum absolute value of the signal.
    R (int, optional): The number of iterations for the clipping algorithm. Default is 100.

    Returns:
    np.ndarray: The optimized signal after clipping.
    list: The history of crest factors for each iteration.
    """
    U = fft(signal)
    UFixed = np.abs(U)  # amplitude to be respected

    Cr = []  # Crest factor history
    if variable==True:
        Clip=np.linspace(Clip_value,1,R)
    else:
        Clip=np.ones(R)*Clip_value

    for k in range(R):  # clipping algorithm
        u = np.real(ifft(U))
        current_cr = np.max(np.abs(u))/rms_value
        Cr.append(current_cr)
        uClip = Clip[k] * current_cr
        u[np.abs(u) > uClip] = u[np.abs(u) > uClip] / np.abs(u[np.abs(u) > uClip]) * uClip
        U = fft(u)
        U = UFixed * np.exp(1j * np.angle(U))  # restore original amplitude spectrum

    return u, Cr,uClip

def log_amplitude(N, k1, k2, kdens):
    """
    Generates a vector of length N where elements corresponding to logarithmically spaced
    frequencies between k1 and k2 are set to 1, and all other elements are 0.

    Parameters:
    N (int): The length of the output vector.
    k1 (int): The starting frequency.
    k2 (int): The stopping frequency.
    kdens (int): The number of frequencies in a decade.

    Returns:
    np.ndarray: A vector of length N with specified frequencies set to 1.
    """
    # Create logarithmic grid in [k1 k2]
    k1Log = np.log10(k1)
    k2Log = np.log10(k2)
    fLog = np.round(10**np.arange(k1Log, k2Log + 1/kdens, 1/kdens))

    # Filter out duplicates and ensure the frequencies fall within the range [1, N]
    fAll = np.unique(fLog).astype(int)
    fAll = fAll[(fAll >= 1) & (fAll <= N)]

    # Initialize vector A with zeros
    A = np.zeros(N, dtype=int)

    # Set the elements corresponding to the logarithmically spaced frequencies to 1
    A[fAll-1] = 1  # -1 for zero-based indexing in Python

    return A,fAll

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

# Value check befor running the main operation

def check_variables(samplefrequency,samplenumbers,periods,transientperiod,window):
    f_s=samplefrequency
    N  = samplenumbers
    P  = periods
    P_tf = transientperiod
    window = window


    f_0   = f_s/N                                         # Excitation frequency
    T     = N/f_s                                         # Time length
    t     = np.linspace(0, T,N,endpoint=False)            # Time vector
    f     = np.linspace(0,f_s,N,endpoint=False)           # Frequency range
    N_tf  = P_tf*N                                   # Number of transient points


    if f_s>=0.5*N:
        raise ValueError(f'To prevent leakage the number of samples N must always be more then 2 times the sample frequency f_s.')
    if N %1 !=0 :
        raise ValueError(f'The number of samples N, must always be an integer value. {N} can not be a part of a sample')
    if N_tf % 1 != 0:
        raise ValueError(f"The product of P_tf ({P_tf}) and N ({N}) must be an integer, decimal number of periods can be processed. Received {N_tf} instead.")
    if window[0]==True  and P_tf %1 !=0 :
        raise ValueError(f'When applying the window on the input signal, one can not select a percentage valued transient period P_tf. Select P_tf {P_tf} either as {int(np.floor(P_tf))} or {int(np.ceil(P_tf))}')

def matlab_input_file(filename,input,output,reference,samplenumbers,sampletime,frequencyrange):
        # Organizing the data into the specified structure
    # Path for the MATLAB .m file with .m extension
    m_filename = filename #if filename.endswith('.mat') else f"{filename}.mat"

    input_array = np.asarray(input)
    output_array = np.asarray(output)
    reference_array = np.asarray(reference)
    frequencyrange_array = np.asarray(frequencyrange)

    # Organizing the data into the specified structure
    data = {
        'u': input_array,
        'y': output_array,
        'r': reference_array,
        'N': np.array([samplenumbers]),
        'Ts': np.array([sampletime]),
        'ExcitedHarm': frequencyrange_array
    }

    # Save the dictionary to a .mat file structured as a 1x1 struct
    #savemat(m_filename, {'data': data})
    # Save the dictionary to a .mat file
    return savemat(m_filename,  data)

def matlab_input_data(input,output,reference,samplenumbers,sampletime,frequencyrange):

    input_array = np.asarray(input)
    output_array = np.asarray(output)
    reference_array = np.asarray(reference)
    frequencyrange_array = np.asarray(frequencyrange)

    # Organizing the data into the specified structure
    data = {
        'u': input_array,
        'y': output_array,
        'r': reference_array,
        'N': np.array([samplenumbers]),
        'Ts': np.array([sampletime]),
        'ExcitedHarm': frequencyrange_array
    }

    # Save the dictionary to a .mat file structured as a 1x1 struct
    #savemat(m_filename, {'data': data})
    # Save the dictionary to a .mat file
    return data


def matlab_method(filename,order=2,dof=2,transient_on=1):
        # Organizing the data into the specified structure
    # Path for the MATLAB .m file with .m extension
    m_filename = filename if filename.endswith('.mat') else f"{filename}.mat"

    # Organizing the data into the specified structure
    method = {
        'order': np.array([order]),             #order of the local polynomial approximation (default 2)
        'dof': np.array([dof]),                 #degrees of freedom of the (co-)variance estimates = equivalent number of independent experiments - 1 (default ny)
        'transient': np.array([transient_on]),  #determines the estimation of the transient term (optional; default 1) 1: transient term is estimated 0: no transient term is estimated
    }

    # Save the dictionary to a .mat file structured as a 1x1 struct
    #savemat(m_filename, {'data': data})
    # Save the dictionary to a .mat file
    return savemat(m_filename,  method)

def matpy_LPM(u,y,reference_signal,N,fs,excitedharmonics,order=2,dof=1,transient_on=1):

    eng = matlab.engine.start_matlab()

    eng.cd(r'C:\Users\R.J.P. Elfrink\OneDrive - TU Eindhoven\Graduation\Git\Rheology-System-identifcation\EngineMatpy', nargout=0)
    # test connection with the right folder, triarea is a different script in the same folder
    #ret = eng.triarea(1.0,5.0)



    # Convert Python data to MATLAB compatible types
    u_matlab = matlab.double(u)  # Convert list to MATLAB double
    y_matlab = matlab.double(y)  # Convert list to MATLAB double
    r_matlab = matlab.double(reference_signal)  # Convert list to MATLAB double
    ExcitedHarm_matlab = matlab.double(excitedharmonics)  # Convert list to MATLAB double
    order_matlab=matlab.double(order)
    dof_matlab=matlab.double(dof)
    transient_matlab=matlab.double(transient_on)


    # Call the Matpy function with data from Python
    output_data_matlab = eng.Matpy(u_matlab, y_matlab, r_matlab, N, fs, ExcitedHarm_matlab,order_matlab,dof_matlab,transient_matlab)


    # Accessing the encapsulated data in Python
    #for i in range(0, M ):
    #G_i = output_data_matlab['G'][0]

    # Stop MATLAB engine
    eng.quit()
    G_LPM=np.squeeze(np.array(output_data_matlab['G']))

    return G_LPM

