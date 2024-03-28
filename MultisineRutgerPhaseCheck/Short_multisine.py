import numpy as np
import matplotlib.pyplot as plt
#Parameter waardes

f_s = 20                                   # Sample frequency
N = 2000                                   # Number of points
j1=1                                       # multisine starting frequency
j2=N/4                                     # Multisine

f_0           = f_s / N
j1            = int(np.floor(j1))
j2            = int(np.ceil(j2))
J             = int(j2 - j1)
j_range       = np.linspace(j1, j2 - 1, J, endpoint=True).astype(int)

def schroeder_phase(j_range, J):
    """
    Generates Schroeder phase for multisine signals.
    """
    return np.array([-j * (j - 1) * np.pi / J for j in j_range])

phase=schroeder_phase(j_range,J)

# Set up amplitude vector
A             = np.ones(N)

T = (N) / f_s
t = np.linspace(0, T, N , endpoint=False)
u = np.zeros(N)
for j in range(J):
    u += A[j] * np.cos(j_range[j] * t * 2 * np.pi * f_0 + phase[j])




