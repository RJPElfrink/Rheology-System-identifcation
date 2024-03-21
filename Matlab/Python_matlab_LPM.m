%% LOOP TO CALCULATE G OVER MULTIPLE MEASURMENTS, INPUT IS PYTHON .MAT FILE
clear all; close all



method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   2; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 
% Load the data structure from the .mat file
data = load('Matlab_input.mat');

% Assuming 'data' has fields 'u' and 'y', each [100 x 2000]
numMeasurements = size(data.u, 1);  % Assuming the first dimension is the number of measurements

% Initialize 'output_data' as an empty structure with defined fields
%output_data = struct('CZn', [], 'varCvecGn', [], 'varCvecGNL', [], 'freq', [], 'G', []);
output_data = struct('G', []);

for i = 1:numMeasurements
    % Extract the i-th measurement for u and y
    currentData.u = data.u(i, :);
    currentData.y = data.y(i, :);
    currentData.r= data.r;
    currentData.N = data.N;
    currentData.Ts = data.Ts;
    currentData.ExcitedHarm = data.ExcitedHarm;


    % Perform analysis with FastLocalPolyAnal on the current set of measurements
    [CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(currentData);


    %output_data(i).CZn = squeeze(CZ.n);                % noise covariance matrix of the sample mean Z.n over the periods 
    %output_data.varCvecGn = squeeze(CvecG.n);          % noise variance covariance matrix of vec(G)
    %output_data(i).varCvecGNL = squeeze(CvecG.NL);     % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
    %output_data(i).freq = squeeze(freq);               % frequency of the excited harmonics; size 1 x F
    output_data(i).G = squeeze(G).';                   % estimated frequency response matrix; size ny x nu x F
    
end

% Save the output_data structure to a new .mat file
save(['G_output.mat'], 'output_data');



%% CALCULATE G FOR JUST ONE MEASUREMENT, INPUT IS .MAT FILE
clear all; 
method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	0;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 

test=load('Matlab_input_multisine.mat')

data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
%data = struct('u', [], 'y', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data.u = test.u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = test.y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = test.r;                             % one period of the reference signal is sufficient 
data.N = test.N;                             % number of samples in one period 
data.Ts = test.Ts;                         % sampling period 
data.ExcitedHarm = test.ExcitedHarm;         % excited harmonics multisine excitation

[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data, method);

% FRF and its variance
G = squeeze(G).';
%varGn = squeeze(CvecG.n).';              % noise variance
%varGNL = squeeze(CvecG.NL).';            % total variance (noise + NL distortions)

% data output
%output_data = struct('CZn', [], 'varCvecGn', [],'varCvecGNL', [], 'freq', [], 'G', []);
output_data = struct('G', []);
%output_data.CZn = squeeze(CZ.n);                   % noise covariance matrix of the sample mean Z.n over the periods 
%output_data.varCvecGn = squeeze(CvecG.n);           % noise variance covariance matrix of vec(G)
%output_data.varCvecGNL = squeeze(CvecG.NL);         % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
%output_data.freq = squeeze(freq);                  % frequency of the excited harmonics; size 1 x F
output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 





save('G_output.mat', 'output_data');

%G1=G;varGn1=varGn;varGNL1=varGNL;freq1=freq;


%% Transfer function FRF
g = 2.5; lambda = 1.5;
sys = tf(g,[1, 1/lambda]);

% Bode Plot for Designed System
[mag,phase,wout] = bode(sys,freq*2*pi);
mag_tf = squeeze(mag);
phase_tf = deg2rad(squeeze(phase));
G_tf=mag_tf.*exp(1i*phase_tf);

% estimated BLA, its noise and total variances
figure()
%semilogx(freq, db(G), 'k', freq, db(varGn)/2, 'g', freq, db(varGNL)/2, 'r')
semilogx(freq, db(G), 'k', freq, db(G_tf),'g')
%semilogx(freq, db(G), 'k', freq,  db(varGNL)/2, 'r',freq1, db(G1), 'b', freq1, db(varGNL1)/2, 'g')
%xlim(0.01,5)
xlabel('Frequency (Hz)')
ylabel('{\itG}_{BLA} (dB)')
title('FRF of Multisine, method: order=2,dof=1,transient=0')
legend(  '{\itG}_{Multisine}','{\itG}_{0}', 'Location', 'EastOutside');
zoom on
shg

%% FUNCTION FOR THE PYTHON MATLAB ENGINE


function output_data = nieuwe(u,y,r,N,fs,ExcitedHarm)
% data 


data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data.u = u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = r;                             % one period of the reference signal is sufficient 
data.N = N;                             % number of samples in one period 
data.Ts = 1/fs;                         % sampling period 
data.ExcitedHarm = ExcitedHarm;         % excited harmonics multisine excitation


%method = load("Matlab_Method.mat");

% Load the data structure from the .mat file
%data = nieuw;
%method = load('Matlab_method.mat');

% Assuming 'data' has fields 'u' and 'y', each [100 x 2000]
numMeasurements = size(data.u, 1);  % Assuming the first dimension is the number of measurements

% Initialize 'output_data' as an empty structure with defined fields
%output_data = struct('CZn', [], 'varCvecGn', [], 'varCvecGNL', [], 'freq', [], 'G', []);
%output_data = struct('G', []);
output_data = struct();
for i = 1:numMeasurements
    % Extract the i-th measurement for u and y
    currentData.u = data.u(i, :);
    currentData.y = data.y(i, :);
    currentData.r= data.r;
    currentData.N = data.N;
    currentData.Ts = data.Ts;
    currentData.ExcitedHarm = data.ExcitedHarm;


    % Perform analysis with FastLocalPolyAnal on the current set of measurements
    [CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(currentData);


    %output_data(i).CZn = squeeze(CZ.n);                % noise covariance matrix of the sample mean Z.n over the periods 
    %output_data.varCvecGn = squeeze(CvecG.n);          % noise variance covariance matrix of vec(G)
    %output_data(i).varCvecGNL = squeeze(CvecG.NL);     % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
    %output_data(i).freq = squeeze(freq);               % frequency of the excited harmonics; size 1 x F
    output_data(i).G = squeeze(G).';                   % estimated frequency response matrix; size ny x nu x F
    
end


end
