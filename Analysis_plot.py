import numpy as np
import Rheosys as rhs
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd


# Example usage:

multisine_schroeder_lpm = rhs.load_data_for_visualization_pickle('visualization_data.pickle')
multisine_schroeder_transient=1
multisine_schroeder_window=1
multisine_random_lpm = rhs.load_data_for_visualization_pickle('multi_random_lpm.pickle')
multisine_random_transient=1
multisine_random_window=1
chirp_window=1


M   = multisine_schroeder_lpm['M']
P   = multisine_schroeder_lpm['P']
N   = multisine_schroeder_lpm['N']
nu  = multisine_schroeder_lpm['nu']
ny  = multisine_schroeder_lpm['ny']
f_range=multisine_schroeder_lpm['f_range']


# Create figure and a single subplot (ax)
fig, ax = plt.subplots(figsize=(12, 8))  # Wider figure to clearly differentiate multiple lines

# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(multisine_schroeder_lpm['G_0']), '-', label='$G_{0}$ Schroeder', color='blue')
ax.plot(f_range, rhs.DB(multisine_schroeder_lpm['G']), 'o-', label='$\hat{G}_{1}$ Schroeder', color='red')
ax.plot(f_range, rhs.DB(multisine_random_lpm['bias']), 'x-', label='Bias Schroeder', color='green')
ax.plot(f_range, rhs.DB(multisine_random_lpm['var']), 'v-', label='Variance Schroeder', color='purple')

# Plot lines for 'multisine_random_lpm' (assuming it has similar structure)
ax.plot(f_range, rhs.DB(multisine_random_lpm['G_0']), '--', label='$G_{0}$ Random', color='cyan')
ax.plot(f_range, rhs.DB(multisine_random_lpm['G']), 's-', label='$\hat{G}_{1}$ Random', color='magenta')
ax.plot(f_range, rhs.DB(multisine_random_lpm['bias']), '+-', label='Bias Random', color='lime')
ax.plot(f_range, rhs.DB(multisine_random_lpm['var']), '^-', label='Variance Random', color='orange')

# Set plot parameters
ax.set_title(f'Frequency Response Function (FRF) Analysis\n{M} Measurements, P={P} Periods, N={N}', fontsize=14)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Magnitude [dB]', fontsize=12)
ax.set_xscale('log')
ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend(loc='best', fontsize=10)  # Changed to 'best' to automatically adjust legend placement
ax.set_xlim([f_range[0], f_range[-1]])
ax.set_ylim([-140, 20])  # Adjusted upper limit and lowered bottom limit for better visibility

# Fine-tuning
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels etc
plt.show()
"""
PLOTTING FOR VISUALIZATION OF THE FRF RESULTS

"""


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(multisine_schroeder_lpm['G_0']),'-',label='$\hat{G}_{0}$')
plt.plot(f_range,rhs.DB(multisine_schroeder_lpm['G']),'o',label='$\hat{G}_{1} Multisine Schroeder$')
plt.plot(f_range,rhs.DB(multisine_schroeder_lpm['bias']),'o',label='$\hat{G}_{1} Bias Multisine windowing$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {nu} dB, SNR$_y$ {ny} , ')
plt.legend(loc='best')
#plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

"""
PLOTTING THE VARIANCE OF EACHT SIGNAL

"""
break

"""
PLOTTING THE BIAS AND VARIANCE TRADE OF IN ONE PLOT
"""

# Assuming you've defined the necessary variables and functions elsewhere
# f_range, M, P, N, noise_input_u, noise_output_y, f_s, etc.

# Placeholder for your actual data
signals = [
    {'name': 'Multisine Transient', 'var': var_G_1_multi_trans, 'dif': dif_G_1_multi_trans},
    # Add similar dictionaries for your other signals (e.g., G_2_multi_window, G_3_multi_LPM, etc.)
]

for signal in signals:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot variance on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Variance [dB^2]', color=color)
    ax1.plot(f_range, rhs.DB(signal['var']), 'o-', label=f"Variance of {signal['name']}", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for bias
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Bias [dB]', color=color)  # we already handled the x-label with ax1
    ax2.plot(f_range, rhs.DB(signal['dif']), 's-', label=f"Bias of {signal['name']}", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    plt.title(f"Bias-Variance Plot for {signal['name']}\n{M} measurements, P={P} periods, N={N}.\nSNR$_u$={noise_input_u[1]} dB, SNR$_y$={noise_output_y[1]} dB")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    plt.show()



