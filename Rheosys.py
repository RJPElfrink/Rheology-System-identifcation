import numpy as np
import scipy as sc
import matplotlib.pyplot as plt


#Excitation signals used as input for the analysis,
#   input must be defined as one value with the unit of ...
#   output is defined with the unit of ...
def sine (period):
    return np.sin(period)

def cos (period):
    return np.cos(period)


def visualization(f_0,t_DT,u_DT,t_CT,u_CT,fdsplit,Udsplit,U_DT_int,sol):
    plt.plot(t_DT, u_DT, "o", t_CT, u_CT, "-")
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.axis('tight')
    plt.show()

    fig0, (ax0) = plt.subplots(1, 1, layout='constrained')
    ax0.stem([-f_0,f_0],[0.5,0.5],linefmt='blue', markerfmt='D',label='Sample frequency $kf_0=kf_s/N$')
    ax0.stem(fdsplit,Udsplit,linefmt='red', markerfmt='D',label='DFT input frequency')
    fig0.suptitle('Amplitude spectrum of DFT and FT should coincide with the $f_0$ frequency')
    fig0.supxlabel('f[Hz]')
    fig0.supylabel('$|U_{DFT}|$')
    ax0.legend()
    plt.show()

    plt.plot(t_DT, u_DT, "o")
    plt.plot(t_CT, U_DT_int, "-")
    plt.xlabel('Time[s]')
    plt.ylabel('Amplitude')
    plt.axis('tight')
    plt.show()

    plt.plot(np.squeeze(t_DT), np.squeeze(sol.y))
    plt.xlabel('t')
    plt.legend(['x', 'y'], shadow=False)
    plt.title('ODE Maxwell')
    plt.show()

