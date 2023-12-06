import numpy as np
import matplotlib.pyplot as plt
import rheosys as rhs

def input_signal_sample (t_DT,u_DT,t_CT,u_CT,title=str('Continuous signal with sampled points')):
    #Plot with sample points and continous signal
    plt.plot(t_DT, u_DT, ".", t_CT, u_CT, "-")
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.axis('tight')
    return plt.show()

def input_stem_split_freqency(f_0,Udsplit,fdsplit,title=str('Amplitude spectrum of DFT and FT should coincide with the $f_0$ frequency')):
    #Stemplot of split frequency disribution
    plt.stem([-f_0,f_0],[2*max(Udsplit),2*max(Udsplit)],linefmt='blue', markerfmt='D',label='Sample frequency $kf_0=kf_s/N$')
    plt.stem(fdsplit,Udsplit,linefmt='red', markerfmt='d',label='DFT input frequency')
    plt.title(title)
    plt.xlabel('f[Hz]')
    plt.ylabel('$|U_{DFT}|$')
    plt.yscale('log')
    plt.legend()
    return plt.show()

def input_line_frequency(fd,Ud,title=str('The position of the FFT components on the frequency axis in Hz ')):
    #Lineplot over the frequency distribuation in decibels
    plt.plot(fd,rhs.DB(Ud),'-')
    plt.title(title)
    plt.xlabel('f[Hz]')
    plt.ylabel('$dB$')
    plt.legend()
    return plt.show()

def input_reconstructer_sampling(t_DT,u_DT,t_CT,U_DT_int,title=str(f'Reconstructed signal between sample points with interpolation kind of')):
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
    plt.xlabel('time [s]')
    plt.legend(['x', 'y'], shadow=False)
    plt.title(title)
    return plt.show()

def simple_plot (x_value, y_value,title=str(''),x_label='',y_label=''):
    # Plot of the solution of the maxwell differential\
    plt.plot(x_value,y_value)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(['x', 'y'], shadow=False)
    plt.title(title)
    return plt.show()