import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal
import pandas as pd

import sys
sys.path.append(r'C:\Users\R.J.P. Elfrink\OneDrive - TU Eindhoven\Graduation\Git\Rheology-System-identifcation')
import Rheosys as rhs
import os
# Specify the new desired working directory
new_directory = r'C:\Users\R.J.P. Elfrink\OneDrive - TU Eindhoven\Graduation\Git\Rheology-System-identifcation\Data_export'

# Change the current working directory
os.chdir(new_directory)
print("Current Working Directory:", os.getcwd())

def run_analysis(data_path_set, plot_signal_set, window_set, lpm_set, optimizecrest_set, N, P_tf,P, noise_input_set, noise_input_val, noise_output_set, noise_output_val):
    print(f"Running analysis with settings: {data_path_set}, Plot Signal: {plot_signal_set}")
    print(f"Window Enabled: {window_set}, LPM Enabled: {lpm_set}, Optimize Crest: {optimizecrest_set}")
    print(f"Sample Points (N): {N}, Transient Removal (P_tf): {P_tf}")
    print(f"Noise - Input Enabled: {noise_input_set}, Input dB: {noise_input_val}, Output Enabled: {noise_output_set}, Output dB: {noise_output_val}")

    """Set transfer function parameters"""
    Lambda_constant=1.5                        # Value for lambda n/g
    g_constant=2.5                             # Value for spring constant g

    f_s = 20                                    # Sample frequency
    N = N                                  # Number of points
    P = P                                      # Number of repeated periods
    P_tf = P_tf                                # Number of removed transient periods

    M = 2
    f_0   = f_s/N                                         # Excitation frequency
    data_path = data_path_set
    plot_signal=plot_signal_set                    # Select 'Chirp' or 'Multsine' to change plots

    multisine_param     = {'phase'  : 'Random',     # Select phase for the multisine, (Schroeder,Random,Linear,Zero,Rudin,Newman)
                            'j1'    : 1,            # Starting frequency multisine
                            'j2'    : N/2,}         # Stopping freqency multisine

    chirp_param         = { 'j1'    : 1,            # Starting frequency chirp
                            'j2'    : N/2,}         # Stopping frequency chirp

    interpolation_param = { 'set'   : True,         # Set to True to activate interpolation of raw created signal
                            'kind'  : 0}            # Set kind to 0 for Zero Order Hold, optional every odd power number including 2

    normalization_param = { 'set'   : True,         # Set to True to activate normalization
                            'method': 'STD',      # Select normalization methode (None, STDev, Amplitude, RMS)
                            'max'   : 1,            # Optional to multiply signal with a value to set amplitude hight, default =1
                            'x_norm': 0,}           # Equal normalization of previous dignal, x_norm is 'method' value of previous normalizaiton, 0 is default set 'method'

    noise_param     = { 'inputset'  : noise_input_set,         # Set the value for noise on input to true or false and include SNR in decibels
                        'inputdb'   : noise_input_val,
                        'outputset'  : noise_output_set,         # Set the value for noise on output to true or false and include SNR in decibels
                        'outputdb'   : noise_output_val}

    window_param        = { 'set'   : window_set,        # Set to True to activate windwo
                            'r'     : 0.15}         # Set value of R between 0-1, 0 is rectangular, 1 is Hann window

    optimizecrest_param  = {'set'   : optimizecrest_set,       # Set to True to optimize the signal the crest clipping algorithm
                        'clip'   : 0.8,          # Set clipping value which clips the crest value
                        'R'      : 1000,          # Amount of realization to test the optimized signal u
                        'variable': False,}       # Varible clipping value, True to linearly change 'clip' value from set value to 1

    multisine_log_param = {'set'    : False,        # Optional Logarithmic amplitude vector A in multisin
                        'decade' : 10}           # Amount of frequencies per decade, j1 and j2 are defined in 'multisine_params'

    LPM_param           = {'set'    : lpm_set,        # Set to True To calculate the LocalPolynomialMeasurement set it to True
                    'order'      : 4,            # method.order[2] order of approximation
                    'dof'        : 1,            # method.dof degrees of freedom, independent experiments default = 1
                    'transient'  : 1}            # method.transient, include transien estimation = 1, exclude transient estimation = 0

    alias_filter        = { 'set'   : True,     # Set anti-alias filter to true to use it
                           'cutoff' : 1-f_0,    # Set cut off frequency, max 1-f_0
                           'order'  : 4}        # Set buttering order
    """ Calculation of Time window"""
    check = rhs.check_variables(f_s,N,P,P_tf,str(window_param['set']))


    T     = N/f_s                                         # Time length
    t     = np.linspace(0, T,N,endpoint=False)            # Time vector
    f     = np.linspace(0,f_s,N,endpoint=False)           # Frequency range
    N_tf  = int(P_tf*N)                                   # Number of transient points
    t_transient= np.linspace(0, T*(P+P_tf),(N*P+N_tf),endpoint=False)   # Transient-time vector

    ### Transfer function of the maxwell system ###
    # G=g/((1j*f)+1/L)
    s1 = signal.lti([g_constant], [1, 1/Lambda_constant])

    # Convert transfer function to magnitude and phase of FRF
    w, mag, phase = signal.bode(s1,f*2*np.pi)
    w_tf=w/(2*np.pi)                            # Frequency of transfer function in Hz
    phase_tf= np.deg2rad(phase)                 # Fase of transfer function in radians
    mag_tf=10**(mag/20)                         # Magnitude of transfer function [-]
    G_0=mag_tf*np.exp(1j*(phase_tf))            # FRF from defined transfer function

    # analytical transient curve
    transient_analytic = np.exp(-t_transient[:(N)]/Lambda_constant)

    """
    CALCULATION FOR G_0
    The first steady state periode of the total signal is used to calculate G_0
    """

    # Creation of the multisine signal, included frequency vector logarithmic or standard
    if multisine_log_param['set']:
        A_multi,f_log=rhs.log_amplitude(N,multisine_param['j1'],multisine_param['j2'],multisine_log_param['decade'])
        u_mutlisine, phase_multisine=rhs.multisine(f_s,N, [multisine_param['j1'],multisine_param['j2']],phase_response=multisine_param['phase'],A_vect=A_multi)
        f_log=f_log[:-1]+1
    else:
        A_multi=np.ones(N)
        u_mutlisine, phase_multisine=rhs.multisine(f_s,N, [multisine_param['j1'],multisine_param['j2']],phase_response=multisine_param['phase'],A_vect=A_multi)

    # Calculation of the input signal for the multisine signal over the entire transient time period
    u_chirp=rhs.chirp_exponential(f_s,N,[chirp_param['j1'],chirp_param['j2']])


    #Select u_chirp or u_multi as tranient signal for figure generation
    if plot_signal=='Multisine':
        u_select   = u_mutlisine
        band_range = [int(multisine_param['j1']),int(multisine_param['j2'])]
    elif plot_signal=='Chirp':
        u_select = u_chirp
        band_range = [int(chirp_param['j1']),int(chirp_param['j2'])]

    # Crest optimization algorithm is activated if optimizecrest_param is set to true, use of clipping
    if optimizecrest_param['set']:
        u_select,crst=rhs.crest_optimization(u_select,optimizecrest_param['clip'],optimizecrest_param['R'],1,optimizecrest_param['variable'])


    excitedharm_range = np.linspace(band_range[0],band_range[1],int(band_range[1]-band_range[0])+1,endpoint=True)
    #f_range = excitedharm_range*f_0
    f_range = np.arange(0,int(band_range[1])*f_0,f_0)
    u_transient= np.tile(u_select, P)
    if P_tf!=0:
        u_transient = np.insert(u_transient,0,u_transient[-N_tf:])

    # Save the set transient basic signal and apply further normalization an interpolation
    u_0 = u_select
    u_0_transient=u_transient

    # Calculations of the multiphase crest factor
    fact_crest = rhs.crest_fac(u_0)
    fact_effic = rhs.power_efficiency(fft(u_0),band_range)
    fact_loss  = rhs.power_loss(fft(u_0),band_range)
    fact_quali = rhs.quality_factor(fact_crest,fact_effic,fact_loss)

    print('Crest factor: {0:.2f} '.format(fact_crest))
    print('Power efficiency: {0:.2f} '.format(fact_effic))
    print('Power loss: {0:.2f} '.format(fact_loss))
    print('Signal quality: {0:.2f} '.format(fact_quali))

    # Apply the window function individually to each period,
    if window_param['set']:
        window = rhs.window_function(t, T, window_param['r'])
        window_transient = np.tile(window, P+P_tf)
        u_0_transient = u_transient * window_transient
    else:
        u_0_transient = u_transient

    # Sampled transient signal as a function of t
    if interpolation_param['set']:
        u_0_transient_t=interp1d(t_transient,u_0_transient,kind=interpolation_param['kind'])
        u_0_transient=u_0_transient_t(t_transient)

    # Normalize the input signal to None, STDev, Amplitude or RMS
    if normalization_param['set']:
        u_0_transient=rhs.normalization(u_0_transient,normalization_param['method'],normalization_param['max'])

    # Design an anti-aliasing filter
    b, a = signal.butter(alias_filter['order'], alias_filter['cutoff'], btype='low')

    # Initialize lists to collect data
    U_M = []
    Y_M = []
    G_etfe_p=[]
    G_etfe_m=[]
    G_ML_m=[]
    G_LPM_m =[]

    for m in range(M):
        """
        CALCULATIONS OF NOISE
        Starting with the addition of measurement noise on either inpot or output
        Noise on input is not transfer  trough the transfer function in calculation of the output
    """
        # Apply measurement noise to the input
        if noise_param['inputset']:
            n_u=rhs.add_noise_with_snr(u_0_transient,noise_param['inputdb'])
            u_n_transient=u_0_transient + n_u
        else:
            u_n_transient=u_0_transient
        if alias_filter['set']:
            u_n_transient= signal.filtfilt(b, a, u_n_transient)
            u_n_transient= np.ascontiguousarray(u_n_transient)
        else:
            u_n_transient=u_n_transient


        # Compute the response output y by
        t_out,y_out, _ = signal.lsim(s1, U=u_0_transient, T=t_transient)

        # Application of measurement noise on the output
        if noise_param['outputset']:
            n_y = rhs.add_noise_with_snr(y_out,noise_param['outputdb'])
            y_n_transient= y_out + n_y
            y_0_transient=np.squeeze(y_out)          # y output in time domain entire transient signal
        else:
            y_0_transient=np.squeeze(y_out)          # y output in time domain entire transient signal
            y_n_transient=y_0_transient

            if alias_filter['set']:
                y_n_transient=signal.filtfilt(b, a, y_n_transient)
                y_n_transient= np.ascontiguousarray(y_n_transient)

            else:
                y_n_transient=y_n_transient

        """
        CALCULATION OF G^ IN MULTIPLE PERIODS
        Depending on the amount of selected transfer free periods
        The overall Frequency responce is calculated

        """
        u_p=[]
        y_p=[]
        U_p=[]
        Y_p=[]


        for p in range(P):

            # Calculate indices for extracting one period
            start_idx = (N *   p)+ N_tf
            end_idx = start_idx + N

            if p==0:
                y_n_steady = y_n_transient[start_idx:end_idx]
            else:
                y_n_steady = y_n_transient[(N *   p)+ N_tf:]
            # Extract one period from the transient signals
            y_p_period = y_n_transient[start_idx:end_idx]
            u_p_period = u_n_transient[start_idx:end_idx]

            u_p.append(u_p_period)
            y_p.append(y_p_period)

            # Calculate FFT of the extracted period
            Y_p_period = fft(y_p_period)
            U_p_period = fft(u_p_period)
            #U_p_period=U_p_period[:int(band_range[1])]
            #Y_p_period=Y_p_period[:int(band_range[1])]
            if multisine_log_param['set']:
                U_p_period=U_p_period[f_log]
                Y_p_period=Y_p_period[f_log]
            # Append FFT results to the lists
            Y_p.append(Y_p_period)
            U_p.append(U_p_period)

            # Calculate FRF for each period and append to G_n
            G_etfe_p.append(Y_p_period/U_p_period)
        # Defined the steady state measurement
        y_p_steady = y_n_transient[N_tf:]
        u_p_steady = u_n_transient[N_tf:]

        # Calculate FFT of the steady state
        Y_p_steady = fft(y_p_steady)
        U_p_steady = fft(u_p_steady)

        U_M.append(U_p)
        Y_M.append(Y_p)
        G_ML_m.append(np.sum(Y_p,axis=0)/np.sum(U_p,axis=0))
        G_etfe_m.append(np.mean(G_etfe_p,axis=0))
        if LPM_param['set']:
            G_LPM_m.append(rhs.matpy_LPM(u_p_steady,y_p_steady,u_select,N,f_s,excitedharm_range,LPM_param['order'],LPM_param['dof'],LPM_param['transient']))

    U_M=np.array(U_M)
    Y_M=np.array(Y_M)
    G_etfe=np.array(np.mean(G_etfe_m,axis=0))
    G_ML=np.array(np.mean(G_ML_m,axis=0))
    G_LPM=np.array(np.squeeze(np.mean(G_LPM_m,axis=0)))




    """
    Calculations have been made, to export the right
    """
    f_range=f_range[:-1]
    G_0=G_0[1:int(band_range[1])]

    if LPM_param['set']:
        G_export=G_LPM[:int(band_range[1])-1]
        G_export_m=G_LPM_m
    else:
        G_export=G_ML[1:int(band_range[1])]
        G_export_m=G_ML_m





    plot_title=rhs.create_plot_title(f_s, N, P, M, plot_signal, multisine_param, chirp_param, interpolation_param, normalization_param, noise_param, window_param, optimizecrest_param, multisine_log_param, LPM_param)
    rhs.export_data_for_visualization_pickle(data_path, G_export, G_export_m,  f_range, G_0, band_range, N, P, M, u_p_steady,t, noise_param, plot_title,fact_crest,fact_effic,fact_loss,fact_quali)

    bias_export=G_0-G_export
    var_export=np.var(G_etfe_m,axis=0)
    var_export=var_export[1:int(band_range[1])]


    plt.plot(t_transient,u_0_transient)
    plt.show()

    # Figure of the created multisine signals frequency domain in Decibel
    plt.plot(f,rhs.DB(U_p_period))
    plt.ylabel('Amplitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.xscale('log')
    plt.xlim([f_s/N,f_s/2])
    plt.title(f'Frequency range of the {plot_signal} input signal U')
    plt.show()



    # Phase plot of the FRF transferfunction with multisine
    plt.plot(f_range,rhs.DB(G_0),'-',label='$\hat{G}_{0}$')
    plt.plot(f_range,rhs.DB(G_export),'o',label='$\hat{G}_{1} Export$')
    plt.plot(f_range,rhs.DB(bias_export),'o',label='$\hat{G}_{1} Bias export$')
    plt.plot(f_range,10*np.log10(var_export),'-',label='$\hat{G}_{1} Variance export$')

    plt.title(f'FRF {plot_signal} ,window {window_param["set"]}, plot with {M} measurments and P={P} periods, N is equal to {N}.\n Noise values SNR$_u$ {noise_param["inputdb"]} dB, SNR$_y$ {noise_param["outputdb"]} dB, ')
    plt.legend(loc='best')
    plt.xlim([f_s/N,f_s/2])
    plt.ylim(-100,25)
    plt.xscale('log')
    plt.ylabel('Magnitude [dB]')
    plt.xlabel('Frequency (Hz)')
    plt.show()


# Configuration Dictionaries
sixsignals = [
    {'plot_signal_set': 'Chirp', 'window_set': True, 'optimizecrest_set': False, 'lpm_set': False, 'N': 20000, 'P_tf': 0, 'P':1},
    {'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': True, 'N': 20000, 'P_tf': 0, 'P':1},
    {'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': False, 'N': 16000, 'P_tf': 0.25, 'P':1},
    {'plot_signal_set': 'Multisine', 'window_set': True, 'optimizecrest_set': True, 'lpm_set': False, 'N': 20000, 'P_tf': 0, 'P':1},
    {'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': True, 'N': 20000, 'P_tf': 0, 'P':1},
    {'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': False, 'N': 16000, 'P_tf': 0.25,'P':1},
    #{'plot_signal_set': 'Chirp', 'window_set': True, 'optimizecrest_set': False, 'lpm_set': False, 'N': 10000, 'P_tf': 0, 'P':2},
    #{'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': True, 'N': 10000, 'P_tf': 0, 'P':2},
    #{'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': False, 'N': 8000, 'P_tf': 0.25, 'P':2},
    #{'plot_signal_set': 'Multisine', 'window_set': True, 'optimizecrest_set': True, 'lpm_set': False, 'N': 10000, 'P_tf': 0, 'P':2},
    #{'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': True, 'N': 10000, 'P_tf': 0, 'P':2},
    #{'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': False, 'N': 8000, 'P_tf': 0.25,'P':2},
    # {'plot_signal_set': 'Chirp', 'window_set': True, 'optimizecrest_set': False, 'lpm_set': False, 'N': 5000, 'P_tf': 0, 'P':4},
    #{'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': True, 'N': 5000, 'P_tf': 0, 'P':4},
    #{'plot_signal_set': 'Chirp', 'window_set': False, 'optimizecrest_set': False, 'lpm_set': False, 'N': 4000, 'P_tf': 0.25, 'P':4},
    #{'plot_signal_set': 'Multisine', 'window_set': True, 'optimizecrest_set': True, 'lpm_set': False, 'N': 5000, 'P_tf': 0, 'P':4},
    #{'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': True, 'N': 5000, 'P_tf': 0, 'P':4},
    #{'plot_signal_set': 'Multisine', 'window_set': False, 'optimizecrest_set': True, 'lpm_set': False, 'N': 4000, 'P_tf': 0.25,'P':4}
]

noise_levels = [
    {'noise_input_set': False, 'noise_input_val': 999, 'noise_output_set': False, 'noise_output_val': 999},
    #{'noise_input_set': False, 'noise_input_val': 999, 'noise_output_set': True, 'noise_output_val': 40},
    #{'noise_input_set': False, 'noise_input_val': 999, 'noise_output_set': True, 'noise_output_val': 20},
    #{'noise_input_set': False, 'noise_input_val': 999, 'noise_output_set': True, 'noise_output_val': 0},
    #{'noise_input_set': True, 'noise_input_val': 0, 'noise_output_set': True, 'noise_output_val': 40},
    #{'noise_input_set': True, 'noise_input_val': 0, 'noise_output_set': True, 'noise_output_val': 20},
    #{'noise_input_set': True, 'noise_input_val': 0, 'noise_output_set': True, 'noise_output_val': 0},
    #{'noise_input_set': True, 'noise_input_val': 20, 'noise_output_set': True, 'noise_output_val': 40},
    #'noise_input_set': True, 'noise_input_val': 20, 'noise_output_set': True, 'noise_output_val': 20},
    #{'noise_input_set': True, 'noise_input_val': 20, 'noise_output_set': True, 'noise_output_val': 0},
    #{'noise_input_set': True, 'noise_input_val': 40, 'noise_output_set': True, 'noise_output_val': 40},
    #{'noise_input_set': True, 'noise_input_val': 40, 'noise_output_set': True, 'noise_output_val': 20},
    #{'noise_input_set': True, 'noise_input_val': 40, 'noise_output_set': True, 'noise_output_val': 0}

]




# Loop Over Configurations
for signal_config in sixsignals:
    for noise_config in noise_levels:
        # Build a descriptive data path
        data_path_set = f"{signal_config['plot_signal_set']}"
        data_path_set += '_crest' if signal_config['optimizecrest_set'] else ''
        data_path_set += '_window' if signal_config['window_set'] else ''
        data_path_set += '_transient' if signal_config['P_tf'] > 0 else ''
        data_path_set += '_lpm' if signal_config['lpm_set'] else ''
        data_path_set += f"_input_{noise_config['noise_input_val']}dB"
        data_path_set += f"_output_{noise_config['noise_output_val']}dB"
        data_path_set += f"_N_{signal_config['N']}"
        data_path_set += f"_P_{signal_config['P']}"

        run_analysis(
            data_path_set=data_path_set,
            plot_signal_set=signal_config['plot_signal_set'],
            window_set=signal_config['window_set'],
            lpm_set=signal_config['lpm_set'],
            optimizecrest_set=signal_config['optimizecrest_set'],
            N=signal_config['N'],
            P_tf=signal_config['P_tf'],
            P=signal_config['P'],
            noise_input_set=noise_config['noise_input_set'],
            noise_input_val=noise_config['noise_input_val'],
            noise_output_set=noise_config['noise_output_set'],
            noise_output_val=noise_config['noise_output_val']
        )