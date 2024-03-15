import numpy as np
import Rheosys as rhs
import matplotlib.pyplot as plt
from scipy.io import loadmat

#LPM_multisine=loadmat('LPM_multisine.mat')


# Load the .mat file
LPM_multisine_data = loadmat('LPM_multisine.mat')

# Accessing the 'output_data' structure
LPM_multisine = LPM_multisine_data['output_data']

CZn_multisineLPM = np.squeeze(LPM_multisine['CZn'][0,0])
varCvecGn_multisineLPM = np.squeeze(LPM_multisine['varCvecGn'][0,0])
varCvecGNL_multisineLPM = np.squeeze(LPM_multisine['varCvecGNL'][0,0])
freq = np.squeeze(LPM_multisine['freq'][0,0])
G_multisineLPM = np.squeeze(LPM_multisine['G'][0,0])


# Load the .mat file
LPM_chirp_data = loadmat('LPM_chirp.mat')
# Accessing the 'output_data' structure
LPM_chirp = LPM_chirp_data['output_data']

CZn_chirpLPM = np.squeeze(LPM_chirp['CZn'][0,0])
varCvecGn_chirpLPM = np.squeeze(LPM_chirp['varCvecGn'][0,0])
varCvecGNL_chirpLPM = np.squeeze(LPM_chirp['varCvecGNL'][0,0])
freq = np.squeeze(LPM_chirp['freq'][0,0])
G_chirpLPM = np.squeeze(LPM_chirp['G'][0,0])


N=2000
f_s=20
f=np.linspace(0,f_s,N,endpoint=False)
f_range=len(freq)

#G_0=np.load('G_0.npy')
G_0=np.load('G_0_chirp.npy')
#G_0=np.load('G_0_multisine.npy')
G_chirp=np.load('G_Chirp.npy')
G_chirpwindow=np.load('G_Chirpwindow.npy')
G_chirptransient=np.load('G_Chirptransient.npy')
G_multisine=np.load('G_multisine.npy')
G_multisinewindow=np.load('G_multisinewindow.npy')
G_multisinetransient=np.load('G_multisinetransient.npy')


# Phase plot of the FRF transferfunction with multisine
#plt.plot(freq,rhs.DB(G_chirpLPM),'-', label='Chirp_LPM')
#plt.plot(freq,rhs.DB(G_multisineLPM),'-',label="Multisine_LPM")
plt.plot(freq,rhs.DB(G_0[:f_range]),'-',label="G_0")
#plt.plot(freq,rhs.DB(G_0[:f_range]-G_multisine[:f_range]),'o',label="G_0-G_multisine")
plt.plot(freq,rhs.DB(G_0[:f_range]-G_multisinetransient[:f_range]),'-',label="G_0-G_multisine_transient")
plt.plot(freq,rhs.DB(G_0[:f_range]-G_multisineLPM),'-',label="G_0-G_multisine_LPM")
plt.plot(freq,rhs.DB(G_0[:f_range]-G_chirp[:f_range]),'-',label="G_0-G_chirp")
plt.plot(freq,rhs.DB(G_0[:f_range]-G_chirpwindow[:f_range]),'-',label="G_0-G_chirp_window")
plt.plot(freq,rhs.DB(G_0[:f_range]-G_chirpLPM[:f_range]),'-',label="G_0-G_chrip_LPM")
plt.title(f'Comparison of the FRF of different signal excitations M=1, P=20 ,f_range= 1-500 ')
plt.legend()
plt.xlim([f_s/N,f_s/2])
#plt.ylim(-100,25)
plt.xlim(0.01,5)
plt.xscale('log')
plt.ylabel('Magnitude [dB]')
plt.xlabel('Frequency (Hz)')
plt.show()