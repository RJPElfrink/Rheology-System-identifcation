import numpy as np
import Rheosys as rhs
import matplotlib.pyplot as plt
from scipy.io import loadmat

#LPM_multisine=loadmat('LPM_multisine.mat')

N=2000
f_s=20
f=np.linspace(0,f_s,N,endpoint=False)
#f_range=len(freq)

# Save specified signal
G_0  =np.load('G_0.npy')
#G_1_m=np.load('G_1_m.npy')
#G_2_m=np.load('G_2_m.npy')
#G_3_m=np.load('G_3_m.npy')
#G_4_m=np.load('G_4_m.npy')
G_5_m=np.load('G_5_m.npy')


#G_1_multi_trans=np.mean(G_1_m,axis=0)
#G_2_multi_window=np.mean(G_2_m,axis=0)
#G_3_multi_LPM=np.mean(G_3_m,axis=0)
#G_4_chirp_window=np.mean(G_4_m,axis=0)
G_5_chirp_LPM=np.mean(G_5_m,axis=0)

#var_G_1_multi_trans=np.var(G_1_m,axis=0)
#var_G_2_multi_window=np.var(G_2_m,axis=0)
#var_G_3_multi_LPM=np.var(G_3_m,axis=0)
#var_G_4_chirp_window=np.var(G_4_m,axis=0)
var_G_5_chirp_LPM=np.var(G_5_m,axis=0)

#dif_G_1_multi_trans=G_1_multi_trans-G_0
#dif_G_2_multi_window=G_2_multi_window-G_0
#dif_G_3_multi_LPM=G_3_multi_LPM-G_0
#dif_G_4_chirp_window=G_4_chirp_window-G_0
dif_G_5_chirp_LPM=G_5_chirp_LPM-G_0




break
"""
PLOTTING FOR VISUALIZATION OF THE FRF RESULTS

"""


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(G_0[:int(band_range[1])]),'-',label='$\hat{G}_{0}$')
plt.plot(f_range,rhs.DB(G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine transient$')
#plt.plot(f_range,rhs.DB(G_2_multi_window),'o',label='$\hat{G}_{2}Multisine windowing$')
#plt.plot(f_range,rhs.DB(G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine LPM$')
#plt.plot(f_range,rhs.DB(G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
#plt.plot(f_range,rhs.DB(G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')
plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

"""
PLOTTING THE VARIANCE OF EACHT SIGNAL

"""


# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(var_G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine transient$')
plt.plot(f_range,rhs.DB(var_G_2_multi_window),'o',label='$\hat{G}_{2}Multisine windowing$')
plt.plot(f_range,rhs.DB(var_G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine LPM$')
plt.plot(f_range,rhs.DB(var_G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
plt.plot(f_range,rhs.DB(var_G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')

plt.title(f'FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()

"""
PLOTTING THE BIAS OF EACH SIGNAL

"""

# Phase plot of the FRF transferfunction with multisine
plt.plot(f_range,rhs.DB(dif_G_1_multi_trans),'o',label='$\hat{G}_{1} Multisine transient$')
plt.plot(f_range,rhs.DB(dif_G_2_multi_window),'o',label='$\hat{G}_{2}Multisine windowing$')
plt.plot(f_range,rhs.DB(dif_G_3_multi_LPM),'o',label='$\hat{G}_{3}Multisine LPM$')
plt.plot(f_range,rhs.DB(dif_G_4_chirp_window),'o',label='$\hat{G}_{4}Chirp windowing$')
plt.plot(f_range,rhs.DB(dif_G_5_chirp_LPM),'o',label='$\hat{G}_{5}Chirp LPM$')

plt.title(f'Bias of the FRF plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_input_u[1]} dB, SNR$_y$ {noise_output_y[1]} dB, ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()


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



