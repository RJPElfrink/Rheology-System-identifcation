import numpy as np

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

def multisine(f_s, N,  frequency_limits, A_vect=None,phase_response='Schroeder', Tau=1 , time_domain=True):
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
    j1            = int(np.floor(frequency_limits[0]))
    j2            = int(np.ceil(frequency_limits[1]))
    J             = int(j2 - j1)
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
     # only needed for output with t as Lambda undifined parameter
    u = np.zeros(N) if time_domain else lambda k: sum(A[j] * np.cos(j_range[j] * k * 2 * np.pi * f_0 + phase[j]) for j in range(J))


    if time_domain:
        T = (N) / f_s
        t = np.linspace(0, T, N , endpoint=False)
        for j in range(J):
            u += A[j] * np.cos(j_range[j] * t * 2 * np.pi * f_0 + phase[j])

    return u, phase

#Parameter waardes
f_s = 20                                   # Sample frequency
N = 2000                                   # Number of points
j1=1                                       # multisine starting frequency
j2=N/4                                     # Multisine

u_mutlisine, phase_multisine=multisine(f_s,N, [j1,j2],phase_response='Schroeder')