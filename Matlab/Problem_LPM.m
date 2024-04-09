%% CALCULATE G FOR JUST ONE MEASUREMENT, INPUT IS .MAT FILE
clear all; close all;
method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	0;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 


% ALL EXPORTS AND PLOTS ARE WITHOUT TRANSIENT ESTIMATION
% All data is without input noise, with output noise 40db with 1 transient free periode of N=2000
% G in python is averaged over 10 measurements

% This G is calculated by the LPM but called from python
% Data is derectly exported
datapylpm=load('Matlab_input_chirplpm_test.mat');
%datapylpm=load('Matlab_input_multischroederlpm_test.mat');
Gpylpm=datapylpm.G;

% G calculated in python without the use of the LPM function
datapy = load('Matlab_input_chirp_test.mat');
%datapy = load('Matlab_input_multischroeder_test.mat');
Gpy=datapy.G;

data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', [],'order',[],'dof',[],'transient',[]);
data.u = datapy.u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = datapy.y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = datapy.r;                             % one period of the reference signal is sufficient 
data.N = datapy.N;                             % number of samples in one period 
data.Ts = datapy.Ts;                         % sampling period 
data.ExcitedHarm = datapy.ExcitedHarm;         % excited harmonics multisine excitation

% G calculated by defined function below matlab script, same as the
% function which is called by python
G1 = nieuwe(data.u, data.y, data.r, data.N, data.Ts, data.ExcitedHarm, method.order, method.dof, method.transient);
Gmatlpmfunction=G1.G;


% G calculated directly by matlab FastLocalPolyAnal
[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data, method);
output_data = struct('G', []);
output_data.G = squeeze(G);    
Gmatlpm=output_data.G;



%% FIGURE TEST FOR INPUT SIGNAL
N = double(datapy.N);
Ts=double(datapy.Ts);
r = double(datapy.r);
u = double(datapy.u);
y = double(datapy.y);
excitedharm= double(datapy.ExcitedHarm);


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


umat=u(tlines);
ymat=y(tlines);
Umat=fft(umat); 
Ymat=fft(ymat);
% G calculate by export u and y form python but G calculated in matlab
Gmat=Ymat./Umat;

%set scale
%set scale
lines2 = 1:N/2-1; nLines = length(lines);
lines3 = 2:N/2;
freq=freq(lines2);

G0=G0(lines2);
Gmat=Gmat(lines2);
Gmatlpm=Gmatlpm(lines3)';
Gpylpm=Gpylpm(lines3);
Gpy=Gpy(lines2);
%Gmatlpmfunction=Gmatlpmfunction(lines2);
%%
figure()
plot(t,u(tlines))

%% plots
figure; hold on
semilogx(freq,db(G0))
semilogx(freq,db(Gmat))
semilogx(freq,db(Gmatlpm))
semilogx(freq,db(Gpy))
semilogx(freq,db(Gpylpm))
set(gca, 'XScale', 'log');
% figure()
% semilogx(freq,db(G0),' d' ,freq,db(Gmat),' d' ,freq,db(Gmatlpm),' d' ,freq,db(Gpy),' d' ,freq,db(Gpylpm),' d')
legend(  '{G0}','{Gmat}',' {Gmatlpm} ','{Gpy} ',' {Gpylpm }',   'Location', 'EastOutside');


figure; hold on
semilogx(freq,db(G0),'-')
semilogx(freq,db(G0-Gmat),'-o')
semilogx(freq,db(G0-Gmatlpm),'-+')
semilogx(freq,db(G0-Gpy),'-*')
semilogx(freq,db(G0-Gpylpm),'-<')
set(gca, 'XScale', 'log');

% figure()
% semilogx(freq,db(G0),freq,db(G0-Gmat),freq,db(G0-Gmatlpm),freq,db(G0-Gpy),freq,db(G0-Gpylpm))
legend(  '{G0}','{G0-Gmat}',' {G0-Gmatlpm} ','{G0-Gpy} ',' {G0-Gpylpm }',  'Location', 'EastOutside');

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
output_data = struct('G', []);
output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 


end
