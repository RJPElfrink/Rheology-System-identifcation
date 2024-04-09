%% LOOP TO CALCULATE G OVER MULTIPLE MEASURMENTS, INPUT IS PYTHON .MAT FILE
clear all; %close all



method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   2; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 
% Load the data structure from the .mat file
%data = load('Matlab_input.mat');


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
clear all; close all;
method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	0;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 
freq=[];
datapy=load('Matlab_input_chirplpm_test.mat');
Gpylpm=datapy.G;
test = load('Matlab_input_chirp_test.mat');
%test=load(['Matlab_input_randmulti_test.mat']);
Gpython=test.G;

%data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', [],'order',[],'dof',[],'transient',[]);
%data = struct('u', [], 'y', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data.u = test.u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = test.y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = test.r;                             % one period of the reference signal is sufficient 
data.N = test.N;                             % number of samples in one period 
data.Ts = test.Ts;                         % sampling period 
data.ExcitedHarm = test.ExcitedHarm;         % excited harmonics multisine excitation

[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data, method);
%[G] = nieuwe(data.u, data.y, data.r, data.N, data.Ts, data.ExcitedHarm, method.order, method.dof, method.transient);


% FRF and its variance
%G = squeeze(G).';
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

G=output_data.G;

freq(1:2)
save('G_output.mat', 'output_data');

%G1=G;varGn1=varGn;varGNL1=varGNL;freq1=freq;


%% FIGURE TEST FOR INPUT SIGNAL
N = double(test.N);
Ts=double(test.Ts);
r = double(test.r);
u = double(test.u);
y = double(test.y);
excitedharm= double(test.ExcitedHarm);


fs = 1/Ts;
f0 = fs/N;
lines = 2:N/2-1; nLines = length(lines);tlines = 1:N;
%freq=[]
f= 0:f0:fs-f0;
t = 0:Ts:(N-1)*Ts;

if isempty(freq)
    freq = f(1:length(excitedharm));
end



g = 2.5;
L = 1.5;
B = g;
A = [1 1/L];
sys = tf(B,A);
G0 = freqs(B,A,2*pi*f);

%set scale


utest=u(tlines);
ytest=y(tlines);

Utest=fft(utest); 
Ytest=fft(ytest);
Gtest=Ytest./Utest;

%set scale
G0=G0(1:N/2)';
Gtest=Gtest(1:N/2)';
G=G(1:N/2);
Gpylpm=Gpylpm(1:N/2)';
Gpython=Gpython(1:N/2)';
%%
figure()
plot(t,u(tlines))


figure()
semilogx(freq,db(G0),' d' ,freq,db(Gtest),' d' ,freq,db(G),' d' ,freq,db(Gpython),' d' ,freq,db(Gpylpm),' d' )
legend(  '{G0}','{Gtest}',' {G} ','{Gpython} ',' {Gpylpm }',  'Location', 'EastOutside');


figure()
semilogx(freq,db(G0),freq,db(G0-Gtest),freq,db(G0-G),freq,db(G0-Gpython),freq,db(G0-Gpylpm))
legend(  '{G0}','{G0-Gtest}',' {G0-G} ','{G0-Gpython} ',' {G0-Gpylpm }',  'Location', 'EastOutside');



%% Transfer function FRF


% estimated BLA, its noise and total variances
figure()
%semilogx(freq, db(G), 'k', freq, db(varGn)/2, 'g', freq, db(varGNL)/2, 'r')
semilogx(freq, db(G), 'k', freq, db(G0),'g',freq, db(difG), 'k')
%semilogx(freq, db(G), 'k', freq,  db(varGNL)/2, 'r',freq1, db(G1), 'b', freq1, db(varGNL1)/2, 'g')
%xlim(0.01,5)
xlabel('Frequency (Hz)')
ylabel('{\itG}_{BLA} (dB)')
title('FRF of Multisine, method: order=2,dof=1,transient=0')
legend(  '{\itG}_{Multisine}','{\itG}_{0}', 'Location', 'EastOutside');
zoom on
shg

%% FUNCTION FOR THE PYTHON MATLAB ENGINE


function output_data = nieuwe(u,y,r,N,fs,ExcitedHarm,order,dof,transient)
% data 




data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data.u = u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = r;                             % one period of the reference signal is sufficient 
data.N = N;                             % number of samples in one period 
data.Ts = 1/fs;                         % sampling period 
data.ExcitedHarm = ExcitedHarm;         % excited harmonics multisine excitation

method = struct('order', [], 'dof', [], 'transient', []);
method.order        =   order; %order of the local polynomial approximation (default 2) 
method.dof          =   dof; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	transient;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 



[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data,method);


% data output
%output_data = struct('CZn', [], 'varCvecGn', [],'varCvecGNL', [], 'freq', [], 'G', []);
output_data = struct('G', []);
%output_data.CZn = squeeze(CZ.n);                   % noise covariance matrix of the sample mean Z.n over the periods 
%output_data.varCvecGn = squeeze(CvecG.n);           % noise variance covariance matrix of vec(G)
%output_data.varCvecGNL = squeeze(CvecG.NL);         % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
%output_data.freq = squeeze(freq);                  % frequency of the excited harmonics; size 1 x F
output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 


end

