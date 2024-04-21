import numpy as np
import Rheosys as rhs
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd


# Example usage:
chirp_window=rhs.load_data_for_visualization_pickle('chirp_window.pickle')
chirp_transient=rhs.load_data_for_visualization_pickle('chirp_transient.pickle')
chirp_lpm=rhs.load_data_for_visualization_pickle('chirp_lpm.pickle')

multisine_schroeder = rhs.load_data_for_visualization_pickle('multisine_schroeder.pickle')
multisine_random_crest_window=rhs.load_data_for_visualization_pickle('multisine_random_crest_window.pickle')
multisine_random_crest_transient = rhs.load_data_for_visualization_pickle('multisine_random_crest_transient.pickle')
multisine_random_crest_lpm = rhs.load_data_for_visualization_pickle('multisine_random_crest_lpm.pickle')

M   = chirp_window['M']
P   = chirp_window['P']
N   = chirp_window['N']
nu  = chirp_window['nu']
ny  = chirp_window['ny']
f_range=chirp_window['f_range']
f_range_trans=chirp_transient['f_range']
G_0= chirp_window['G_0']


"""
PLOTTING FOR VISUALIZATION OF THE FRF RESULTS

"""
# Create figure and a single subplot (ax)
fig, ax = plt.subplots(figsize=(12, 8))  # Wider figure to clearly differentiate multiple lines

# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(chirp_window['G_0']), '-', label='$G_{0}$', color='blue')
ax.plot(f_range, rhs.DB(chirp_window['bias']), 'x-', label='Bias $\hat{G}_{1}$ Chirp windowing', color='green')
ax.plot(f_range_trans, rhs.DB(chirp_transient['bias']), 'x-', label='Bias $\hat{G}_{2}$ Chirp transient', color='red',alpha=1)
ax.plot(f_range, rhs.DB(chirp_lpm['bias']), 'x-', label='Bias $\hat{G}_{3}$ Chirp LPM', color='magenta',alpha=1)
ax.plot(f_range_trans, rhs.DB(multisine_random_crest_transient['bias']), 'x-', label='Bias $\hat{G}_{4}$ Multisine crest transient', color='lime',alpha=1)
ax.plot(f_range, rhs.DB(multisine_random_crest_window['bias']), 'x-', label='Bias $\hat{G}_{5}$ Multisine crest windowing', color='orange',alpha=1)
ax.plot(f_range, rhs.DB(multisine_random_crest_lpm['bias']), 'x-', label='Bias $\hat{G}_{6}$ Multisine crest LPM', color='cyan',alpha=1)

# Set plot parameters
ax.set_title(f'Frequency Response Function (FRF) Analysis\n{M} Measurements, P={P} Periods, N={N}', fontsize=14)
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Magnitude [dB]', fontsize=12)
ax.set_xscale('log')
ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.legend(loc='upper left', fontsize=10)  # Changed to 'best' to automatically adjust legend placement
ax.set_xlim([f_range[0], f_range[-1]])
ax.set_ylim([-80, 20])  # Adjusted upper limit and lowered bottom limit for better visibility

# Fine-tuning
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels etc
plt.show()

"""
Storage of labels & names
"""

# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(chirp_window['G_0']), '-', label='$G_{0}$', color='blue')

ax.plot(f_range, rhs.DB(chirp_window['G']), '^-', label='$\hat{G}_{1}$ Chirp windowing', color='green')
ax.plot(f_range, rhs.DB(chirp_window['bias']), 'x-', label='Bias $\hat{G}_{1}$ Chirp windowing', color='green')
ax.plot(f_range, 10*np.log10(chirp_window['var']), 's-', label='Variance $\hat{G}_{1}$ Chirp windowing', color='green')

ax.plot(f_range_trans, rhs.DB(chirp_transient['G']), '^-', label='$\hat{G}_{2}$ Chirp transient', color='red',alpha=1)
ax.plot(f_range_trans, rhs.DB(chirp_transient['bias']), 'x-', label='Bias $\hat{G}_{2}$ Chirp transient', color='red',alpha=1)
ax.plot(f_range_trans, 10*np.log10(chirp_transient['var']), 's-', label='Variance $\hat{G}_{2}$ Chirp transient', color='red',alpha=1)

ax.plot(f_range, rhs.DB(chirp_lpm['G']), '^-', label='$\hat{G}_{3}$ Chirp LPM', color='magenta',alpha=1)
ax.plot(f_range, rhs.DB(chirp_lpm['bias']), 'x-', label='Bias $\hat{G}_{3}$ Chirp LPM', color='magenta',alpha=1)
ax.plot(f_range, 10*np.log10(chirp_lpm['var']), 's-', label='Variance $\hat{G}_{3}$ Chirp LPM', color='magenta',alpha=1)

ax.plot(f_range_trans, rhs.DB(multisine_random_crest_transient['G']), '^-', label='$\hat{G}_{4}$ Multisine crest transient', color='lime',alpha=1)
ax.plot(f_range_trans, rhs.DB(multisine_random_crest_transient['bias']), 'x-', label='Bias $\hat{G}_{4}$ Multisine crest transient', color='lime',alpha=1)
ax.plot(f_range_trans, 10*np.log10(multisine_random_crest_transient['var']), 'S-', label='Variance $\hat{G}_{4}$ Multisine crest transient', color='lime',alpha=1)

ax.plot(f_range, rhs.DB(multisine_random_crest_window['G']), '^-', label='$\hat{G}_{5}$ Multisine crest windowing', color='orange',alpha=1)
ax.plot(f_range, rhs.DB(multisine_random_crest_window['bias']), 'x-', label='Bias $\hat{G}_{5}$ Multisine crest windowing', color='orange',alpha=1)
ax.plot(f_range, 10*np.log(multisine_random_crest_window['var']), 's-', label='Variance $\hat{G}_{5}$ Multisine crest windowing', color='orange',alpha=1)

ax.plot(f_range, rhs.DB(multisine_random_crest_lpm['G']), '^-', label='$\hat{G}_{6}$ Multisine crest LPM', color='cyan',alpha=1)
ax.plot(f_range, rhs.DB(multisine_random_crest_lpm['bias']), 'x-', label='Bias $\hat{G}_{6}$ Multisine crest LPM', color='cyan',alpha=1)
ax.plot(f_range, 10*np.log10(multisine_random_crest_lpm['var']), 's-', label='Variance $\hat{G}_{6}$ Multisine crest LPM', color='cyan',alpha=1)

