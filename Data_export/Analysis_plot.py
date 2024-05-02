import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import matplotlib.lines as mlines

import sys
sys.path.append(r'C:\Users\R.J.P. Elfrink\OneDrive - TU Eindhoven\Graduation\Git\Rheology-System-identifcation')
import Rheosys as rhs
sys.path.append(r'C:\Users\R.J.P. Elfrink\OneDrive - TU Eindhoven\Graduation\Git\Rheology-System-identifcation\Data_export')
# Example usage:
chirp_window=rhs.load_data_for_visualization_pickle('Chirp_window_input_20dB_output_20dB_N_20000.pickle')
chirp_transient=rhs.load_data_for_visualization_pickle('Chirp_transient_input_20dB_output_20dB_N_16000.pickle')
chirp_lpm=rhs.load_data_for_visualization_pickle('Chirp_lpm_input_20dB_output_20dB_N_20000.pickle')

#multisine_schroeder = rhs.load_data_for_visualization_pickle('Multisine_oeder.pickle')
multisine_window=rhs.load_data_for_visualization_pickle('Multisine_crest_window_input_20dB_output_20dB_N_20000.pickle')
multisine_transient = rhs.load_data_for_visualization_pickle('Multisine_crest_transient_input_20dB_output_20dB_N_16000.pickle')
multisine_lpm = rhs.load_data_for_visualization_pickle('Multisine_crest_lpm_input_20dB_output_20dB_N_20000.pickle')

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

# Define color palette
colors = {
    "Zero": "#1f77b4",  # Blue
    "One": "#ff7f0e",  # Orange
    "Two": "#2ca02c", # Green
    "Three": "#d62728",  # Red
    "Four": "#9467bd",  # Purple
    "Five": "#8c564b",   # Brown
    "Six": "#e377c2"  # Pink
}

# colours for colourblindnes
colors = {
    'Zero': '#440154',   #dark purple
    'One': '#31688e',          #Blue
    'Two': '#35b779',         #Green
    'Three': '#00204c',     #dark blue
    'Four': '#585c56',       #grayish
    'Five': '#e69f00',        #orange
    'Six': '#56b4e9',      #sky blue
    'Seven': '#009e73'   #Bluish green
}

# Color setup
colormaps = ['viridis','plasma', 'inferno', 'magma', 'cividis', 'Spectral', 'coolwarm']
viridis = plt.cm.get_cmap('viridis',8)
color_names = ['Zero','One', 'Two', 'Three', 'Four', 'Five', 'Six','Seven']
#color_names =['Seven','Zero','One', 'Two', 'Three', 'Four', 'Five', 'Six']

colors = {name: viridis(i) for i, name in enumerate(color_names[:8])}



"""
Chirp plots
"""


# Create figure and a single subplot (ax)
fig, ax = plt.subplots(figsize=(10, 8))  # Wider figure to clearly differentiate multiple lines
# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(chirp_window['G_0']), '-', label='$G_{0}$', color=colors['Zero'])
ax.plot(f_range, rhs.DB(chirp_window['bias']), '^-', label='Bias $\hat{G}_{1}$ Chirp windowing', color=colors['One'])
ax.plot(f_range, 10*np.log10(chirp_window['var']), '--', label='Variance $\hat{G}_{1}$ Chirp windowing', color=colors['One'])
ax.plot(f_range_trans, rhs.DB(chirp_transient['bias']), '^-', label='Bias $\hat{G}_{2}$ Chirp transient clip', color=colors['Two'])
ax.plot(f_range_trans, 10*np.log10(chirp_transient['var']), '--', label='Variance $\hat{G}_{2}$ Chirp transien clip', color=colors['Two'])
ax.plot(f_range, rhs.DB(chirp_lpm['bias']), '^-', label='Bias $\hat{G}_{3}$ Chirp LPM', color=colors['Three'])
ax.plot(f_range, 10*np.log10(chirp_lpm['var']), '--', label='Variance $\hat{G}_{3}$ Chirp LPM', color=colors['Three'])

method_lines = [
    mlines.Line2D([], [], color=colors['Zero'], marker='', linestyle='-', label='$G_{0}$'),
    mlines.Line2D([], [], color=colors['One'], marker='', linestyle='-', label='Chirp windowing'),
    mlines.Line2D([], [], color=colors['Two'], marker='', linestyle='-', label='Chirp transient clip'),
    mlines.Line2D([], [], color=colors['Three'], marker='', linestyle='-', label='Chirp LPM')
]
# Creating custom legend entries for Bias and Variance
type_lines = [
    mlines.Line2D([], [], color='black', marker='^', linestyle='-', label='Bias'),
    mlines.Line2D([], [], color='black', marker='', linestyle='--', label='Variance')
]
# Combine the legend entries
handles = method_lines + [mlines.Line2D([], [], color='none', marker='', linestyle='', label='')] * (len(method_lines) - len(type_lines)) + type_lines
labels = [h.get_label() for h in handles]

#title_test=
# Plot parameters
ax.set_title(chirp_transient['title']+str('\n'),fontsize=16)
ax.set_xlabel('Frequency (Hz)', fontsize=16)
ax.set_ylabel('Magnitude [dB]', fontsize=16)
ax.legend(handles, labels, loc='upper center', fontsize=16,  ncol=2)
ax.set_xscale('log')
ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.set_xlim([f_range[0], f_range[-1]])
ax.set_ylim([-80, 40])

# save figure to pdf
fig.savefig('chirp_biasvariance_N20000_nu20_ny20.pdf')

plt.tight_layout()
plt.show()


"""
Multisine plots
"""


# Create figure and a single subplot (ax)
fig, ax = plt.subplots(figsize=(10, 8))  # Wider figure to clearly differentiate multiple lines
# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(chirp_window['G_0']), '-', label='$G_{0}$', color=colors['Zero'])
ax.plot(f_range, rhs.DB(multisine_window['bias']), '^-', label='Bias $\hat{G}_{4}$ Multisine windowing', color=colors['Four'])
ax.plot(f_range, 10*np.log10(multisine_window['var']), '--', label='Variance $\hat{G}_{4}$ Multisine windowing', color=colors['Four'])
ax.plot(f_range_trans, rhs.DB(multisine_transient['bias']), '^-', label='Bias $\hat{G}_{5}$ Multisine transient clip', color=colors['Five'])
ax.plot(f_range_trans, 10*np.log10(multisine_transient['var']), '--', label='Variance $\hat{G}_{5}$ Multisine transient clip', color=colors['Five'])
ax.plot(f_range, rhs.DB(multisine_lpm['bias']), '^-', label='Bias $\hat{G}_{6}$ Multisine LPM', color=colors['Six'])
ax.plot(f_range, 10*np.log10(multisine_lpm['var']), '--', label='Variance $\hat{G}_{6}$ Multisine LPM', color=colors['Six'])

method_lines = [
    mlines.Line2D([], [], color=colors['Zero'], marker='', linestyle='-', label='$G_{0}$'),
    mlines.Line2D([], [], color=colors['Four'], marker='', linestyle='-', label='Multisine windowing'),
    mlines.Line2D([], [], color=colors['Five'], marker='', linestyle='-', label='Multisine transient clip'),
    mlines.Line2D([], [], color=colors['Six'], marker='', linestyle='-', label='Multisine LPM')
]

# Creating custom legend entries for Bias and Variance
type_lines = [
    mlines.Line2D([], [], color='black', marker='^', linestyle='-', label='Bias'),
    mlines.Line2D([], [], color='black', marker='', linestyle='--', label='Variance')
]
# Combine the legend entries
handles = method_lines + [mlines.Line2D([], [], color='none', marker='', linestyle='', label='')] * (len(method_lines) - len(type_lines)) + type_lines
labels = [h.get_label() for h in handles]
# Creating a single legend with two columns, ensuring correct item placement

# Plot parameters
ax.set_title(multisine_transient['title']+str('\n'),fontsize=16)
ax.set_xlabel('Frequency (Hz)', fontsize=16)
ax.set_ylabel('Magnitude [dB]', fontsize=16)
ax.legend(handles, labels, loc='upper center', fontsize=16,  ncol=2)
ax.set_xscale('log')
ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.set_xlim([f_range[0], f_range[-1]])
ax.set_ylim([-80, 40])

# Save plot to pdf
fig.savefig('multisine_biasvariance_N20000_nu20_ny20.pdf')

plt.tight_layout()
plt.show()


"""
Storage of labels & names
"""

# Plot lines for 'multisine_schroeder_lpm'
ax.plot(f_range, rhs.DB(chirp_window['G_0']), '-', label='$G_{0}$', color=colors['Zero'])

ax.plot(f_range, rhs.DB(chirp_window['G']), '-', label='$\hat{G}_{1}$ Chirp windowing', color=colors['One'])
ax.plot(f_range, rhs.DB(chirp_window['bias']), '^-', label='Bias $\hat{G}_{1}$ Chirp windowing', color=colors['One'])
ax.plot(f_range, 10*np.log10(chirp_window['var']), '--', label='Variance $\hat{G}_{1}$ Chirp windowing', color=colors['One'])

ax.plot(f_range_trans, rhs.DB(chirp_transient['G']), '-', label='$\hat{G}_{2}$ Chirp transient clip', color=colors['Two'])
ax.plot(f_range_trans, rhs.DB(chirp_transient['bias']), '^-', label='Bias $\hat{G}_{2}$ Chirp transient clip', color=colors['Two'])
ax.plot(f_range_trans, 10*np.log10(chirp_transient['var']), '--', label='Variance $\hat{G}_{2}$ Chirp transien clip', color=colors['Two'])

ax.plot(f_range, rhs.DB(chirp_lpm['G']), '-', label='$\hat{G}_{3}$ Chirp LPM', color=colors['Three'])
ax.plot(f_range, rhs.DB(chirp_lpm['bias']), '^-', label='Bias $\hat{G}_{3}$ Chirp LPM', color=colors['Three'])
ax.plot(f_range, 10*np.log10(chirp_lpm['var']), '--', label='Variance $\hat{G}_{3}$ Chirp LPM', color=colors['Three'])

ax.plot(f_range, rhs.DB(multisine_window['G']), '-', label='$\hat{G}_{4}$ Multisine windowing', color=colors['Four'])
ax.plot(f_range, rhs.DB(multisine_window['bias']), '^-', label='Bias $\hat{G}_{4}$ Multisine windowing', color=colors['Four'])
ax.plot(f_range, 10*np.log10(multisine_window['var']), '--', label='Variance $\hat{G}_{4}$ Multisine windowing', color=colors['Four'])

ax.plot(f_range_trans, rhs.DB(multisine_transient['G']), '-', label='$\hat{G}_{5}$ Multisine transient clip', color=colors['Five'])
ax.plot(f_range_trans, rhs.DB(multisine_transient['bias']), '^-', label='Bias $\hat{G}_{5}$ Multisine transient clip', color=colors['Five'])
ax.plot(f_range_trans, 10*np.log10(multisine_transient['var']), '--', label='Variance $\hat{G}_{5}$ Multisine transient clip', color=colors['Five'])

ax.plot(f_range, rhs.DB(multisine_lpm['G']), '-', label='$\hat{G}_{6}$ Multisine LPM', color=colors['Six'])
ax.plot(f_range, rhs.DB(multisine_lpm['bias']), '^-', label='Bias $\hat{G}_{6}$ Multisine LPM', color=colors['Six'])
ax.plot(f_range, 10*np.log10(multisine_lpm['var']), '--', label='Variance $\hat{G}_{6}$ Multisine LPM', color=colors['Six'])

