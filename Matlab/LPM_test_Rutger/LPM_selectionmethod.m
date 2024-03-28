%% TRY FastLocalPolyAnal for periodic signals in time domain


clear all; close all;

method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 

% load multisine data and append values
multi=load('Matlab_input_randmulti_nonoise.mat');

data_multi = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data_multi.u = multi.u;                             % row index is the input number; the column index the time instant (in samples) 
data_multi.y = multi.y;                             % row index is the output number; the column index the time instant (in samples) 
data_multi.r = multi.r;                             % one period of the reference signal is sufficient 
data_multi.N = multi.N;                             % number of samples in one period 
data_multi.Ts = multi.Ts;                           % sampling period 
data_multi.ExcitedHarm = multi.ExcitedHarm;         % excited harmonics multisine excitation

[CZ_m, Z_m, freq_m, G_m, CvecG_m, dof_m, CL_m] = ArbLocalPolyAnal(data_multi, method);

% FRF and its variance
G_multi = squeeze(G_m).';

% Procduce the LPM for the chirp signal with equal method 
chirp=load('Matlab_input_chirp.mat');

data_chirp = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data_chirp.u = chirp.u;                             % row index is the input number; the column index the time instant (in samples) 
data_chirp.y = chirp.y;                             % row index is the output number; the column index the time instant (in samples) 
data_chirp.r = chirp.r;                             % one period of the reference signal is sufficient 
data_chirp.N = chirp.N;                             % number of samples in one period 
data_chirp.Ts = chirp.Ts;                           % sampling period 
data_chirp.ExcitedHarm = chirp.ExcitedHarm;         % excited harmonics multisine excitation

[CZ_c, Z_c, freq_c, G_c, CvecG_c, dof_c, CL_c] = ArbLocalPolyAnal(data_chirp, method);

% FRF and its variance
G_chirp = squeeze(G_c).';


%% Transfer function FRF
g = 2.5; lambda = 1.5;
sys = tf(g,[1, 1/lambda]);

% Bode Plot for Designed System
[mag,phase,wout] = bode(sys,freq_m*2*pi);
mag_tf = squeeze(mag);
phase_tf = deg2rad(squeeze(phase));
G_tf=mag_tf.*exp(1i*phase_tf);

% Plot of G0 reffered to G_LPM
figure()
semilogx(freq_m, db(G_tf),'b',freq_m, db(G_multi), 'k', freq_c, db(G_chirp),'r')
xlabel('Frequency (Hz)')
ylabel('Amplitude (dB)')
title('FRF of LPM with multisine and chirp systems, method: order=2,dof=1,transient=0')
legend( '{\itG}_{0}', '{\itG}_{Multisine}','{\itG}_{Chirp}', 'Location', 'EastOutside');
zoom on
shg

%% different output 
%varGn = squeeze(CvecG.n).';              % noise variance
%varGNL = squeeze(CvecG.NL).';            % total variance (noise + NL distortions)

% data output
%output_data = struct('CZn', [], 'varCvecGn', [],'varCvecGNL', [], 'freq', [], 'G', []);
%output_data.CZn = squeeze(CZ.n);                   % noise covariance matrix of the sample mean Z.n over the periods 
%output_data.varCvecGn = squeeze(CvecG.n);           % noise variance covariance matrix of vec(G)
%output_data.varCvecGNL = squeeze(CvecG.NL);         % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
%output_data.freq = squeeze(freq);                  % frequency of the excited harmonics; size 1 x F
%output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 

%% TRY LOCACLPOLYANAL For arbitrary signals in frequency domain


method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 

MULTI=load('Matlab_U&Y_multisine.mat');

data_MULTI =    struct('Y', [], 'U', [], 'freq', []);
data_MULTI.Y              = MULTI.Y;  %output signal; size ny x F
data_MULTI.U              = MULTI.U;  %input signal; size nu x F
data_MULTI.freq           = MULTI.freq;  %frequency vector in Hz or in DFT numbers; size 1 x F (optional) default: [1:1:F] 

[CY_M, Y_M, TY_M, G_M, CvecG_M, dof_M, CL_M] = LocalPolyAnal(data_MULTI, method);

% FRF and its variance
G_MULTI = squeeze(G_M).';

%CHIRP data
CHIRP=load('Matlab_U&Y_chirp.mat');

data_CHIRP =    struct('Y', [], 'U', [], 'freq', []);
data_CHIRP.Y              = CHIRP.Y;  %output signal; size ny x F
data_CHIRP.U              = CHIRP.U; %input signal; size nu x F
data_CHIRP.freq           = CHIRP.freq;  %frequency vector in Hz or in DFT numbers; size 1 x F (optional) default: [1:1:F] 

[CY_C, Y_C, TY_C, G_C, CvecG_C, dof_C, CL_C] = LocalPolyAnal(data_CHIRP, method);

% FRF and its variance
G_CHIRP = squeeze(G_C).';




%% Transfer function FRF
g = 2.5; lambda = 1.5;
sys = tf(g,[1, 1/lambda]);

% Bode Plot for Designed System
[mag,phase,wout] = bode(sys,data_MULTI.freq*2*pi);
mag_tf = squeeze(mag);
phase_tf = deg2rad(squeeze(phase));
G_tf=mag_tf.*exp(1i*phase_tf);

% Plot of G0 reffered to G_LPM
figure()
semilogx(data_MULTI.freq, db(G_tf),'b',data_MULTI.freq, db(G_MULTI), 'k', data_CHIRP.freq, db(G_CHIRP),'r')
xlabel('Frequency (Hz)')
ylabel('Amplitude (dB)')
title('FRF of LPM with multisine and chirp systems, method: order=2,dof=1,transient=0')
legend( '{\itG}_{0}', '{\itG}_{Multisine}','{\itG}_{Chirp}', 'Location', 'EastOutside');
zoom on
shg


