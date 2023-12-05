import numpy as np
import matplotlib.pyplot as plt
from rheosys import DB

def input_signal_sample (t_DT,u_DT,t_CT,u_CT,title=str('Continuous signal with sampled points')):
    #Plot with sample points and continous signal
    plt.plot(t_DT, u_DT, ".", t_CT, u_CT, "-")
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.axis('tight')
    return plt.show()

def stem_split_freqency_input(f_0,Udsplit,fdsplit,title=str('Amplitude spectrum of DFT and FT should coincide with the $f_0$ frequency')):
    #Stemplot of split frequency disribution
    plt.stem([-f_0,f_0],[2*max(Udsplit),2*max(Udsplit)],linefmt='blue', markerfmt='D',label='Sample frequency $kf_0=kf_s/N$')
    plt.stem(fdsplit,Udsplit,linefmt='red', markerfmt='d',label='DFT input frequency')
    plt.title(title)
    plt.xlabel('f[Hz]')
    plt.ylabel('$|U_{DFT}|$')
    plt.yscale('log')
    plt.legend()
    return plt.show()

def line_frequency_input(fd,Ud,title=str('The position of the FFT components on the frequency axis in Hz ')):
    #Lineplot over the frequency distribuation in decibels
    plt.plot(fd,DB(Ud),'-')
    plt.title(title)
    plt.xlabel('f[Hz]')
    plt.ylabel('$dB$')
    plt.legend()
    return plt.show()

def reconstructer_sampling(t_DT,u_DT,t_CT,U_DT_int,title=str(f'Reconstructed signal between sample points with interpolation kind of')):
    # Plot of the reconstructed signal between sample points
    plt.plot(t_DT, u_DT, ".")
    plt.plot(t_CT, U_DT_int, "-")
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.axis('tight')
    return plt.show()

def maxwel_plot (time, solution_y,title=str('ODE Maxwell')):
    # Plot of the solution of the maxwell differential\
    plt.plot(time,solution_y)
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=False)
    plt.title(title)
    return plt.show()