import numpy as np
import Rheosys as rhs
import matplotlib.pyplot as plt

N=2000
f_s=20
f=np.linspace(0,f_s,N,endpoint=False)

G_chirp=np.load('Chirpwindow.npy')
G_multiwindow=np.load('Multisinewindow.npy')
G_multitransient=np.load('Multisinetransient.npy')

# Phase plot of the FRF transferfunction with multisine
plt.plot(f,rhs.DB(abs(G_chirp)),'-',label='$Chirp&windowing}$')
plt.plot(f,rhs.DB(abs(G_multiwindow)),'-',label='$Multisine&windowing}$')
plt.plot(f,rhs.DB(abs(G_multitransient)),'-',label='$Multisine&transient}$')
plt.title(f'Comparison of the FRF of different signal excitations ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()