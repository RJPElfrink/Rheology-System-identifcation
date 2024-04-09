%% CALCULATE G FOR JUST ONE MEASUREMENT, INPUT IS .MAT FILE
clear all; %close all;
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
%datapylpm=load('Matlab_input_chirplpm_test.mat');
datapylpm=load('Matlab_input_multischroederlpm_test.mat');
Gpylpm=datapylpm.G;

% G calculated in python without the use of the LPM function
%datapy = load('Matlab_input_chirp_test.mat');
datapy = load('Matlab_input_multischroeder_test.mat');
Gpy=datapy.G;

data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', [],'order',[],'dof',[],'transient',[]);
data.u = datapy.u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = datapy.y;                             % row index is the output number; the column index the time instant (in samples) 
data.r = datapy.r;                             % one period of the reference signal is sufficient 
data.N = datapy.N;                             % number of samples in one period 
data.Ts = datapy.Ts;                         % sampling period 
data.ExcitedHarm = datapy.ExcitedHarm;         % excited harmonics multisine excitation

% G calculated by defined Matlab function that is called from python,
% Script is called Matpy, supplied in auxilary
G1 = Matpy(data.u, data.y, data.r, data.N, data.Ts, data.ExcitedHarm, method.order, method.dof, method.transient);
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
lines = 1:N/2-1; nLines = length(lines);tlines = 1:N;
%freq=[]
f= 0:f0:fs-f0;
t = 0:Ts:(N-1)*Ts;

g = 2.5;
L = 1.5;
B = g;
A = [1 1/L];
sys = tf(B,A);
G0 = freqs(B,A,2*pi*f);


U=fft(u); 
Y=fft(y);
% G calculate by export u and y form python but G calculated in matlab
Gdatamat=zeros(N);
Gdatamat=Y./U;
%% MATLAB CREATED MULTISINE

phaserand=2*pi*rand(nLines,1);
k1=1;
k2=nLines;
k = k1:k2;
phasemulti = -k.*(k-1).*pi/nLines;
P=4;
t2 = 0:Ts:(P*N-1)*Ts;
R = zeros(N,1);

%select phaserand or phasemulti 
R(lines) = exp(1i*phasemulti);
umat = 2*real(ifft(R)); umat = umat/std(umat); umat=umat.';

umat = [umat umat umat umat];

ymat = lsim(sys,umat,t2); ymat=ymat';ymat=ymat(N*(P-1)+1:end);

Umat=fft(umat(N*(P-1)+1:end));Ymat=fft(ymat);
Gmat=zeros(N);
Gmat=Ymat./Umat;
%% PLOTTING THE DIFFERENT VERSIONS OF G
%set scale
%set scale
lines2 = 1:N/2-1; nLines = length(lines);
lines3 = 2:N/2;
freq=freq(lines2);

G0=G0(lines3-1);
Gmat=Gmat(lines3-1);
Gdatamat=Gdatamat(lines3-1);
Gpy=Gpy(lines3-1);
Gpylpm=Gpylpm(lines3-1);
Gmatlpm=Gmatlpm(lines3-1)';
Gmatlpmfunction=Gmatlpmfunction(lines3-1)';
%%
figure()
plot(t,umat(tlines))

%% plots
figure; hold on
plot(freq,db(G0))
plot(freq,db(Gmat))
plot(freq,db(Gdatamat))
plot(freq,db(Gpy))
plot(freq,db(Gmatlpm))
plot(freq,db(Gmatlpmfunction))
plot(freq,db(Gpylpm))
set(gca, 'XScale', 'log');
% figure()
% semilogx(freq,db(G0),' d' ,freq,db(Gmat),' d' ,freq,db(Gmatlpm),' d' ,freq,db(Gpy),' d' ,freq,db(Gpylpm),' d')
legend(  '{G0}','{Gmat}','Gdatamat', ' {Gpy} ',' {Gmatlpm} ',' {Gpylpm }',' {Gmatlpmfunction} ',   'Location', 'EastOutside');


figure; hold on
plot(freq,db(G0),'-')
plot(freq,db(G0-Gmat),'-o')
plot(freq,db(G0-Gdatamat),'->')
plot(freq,db(G0-Gpy),'-*')
plot(freq,db(G0-Gmatlpm),'-+')
plot(freq,db(G0-Gmatlpmfunction),'-^')
plot(freq,db(G0-Gpylpm),'-<')
set(gca, 'XScale', 'log');

% figure()
% semilogx(freq,db(G0),freq,db(G0-Gmat),freq,db(G0-Gmatlpm),freq,db(G0-Gpy),freq,db(G0-Gpylpm))
legend(  '{G0}','{G0-Gmat}','{G0-Gdatamat}','{G0-Gpy} ',' {G0-Gmatlpm} ',' {G0-Gmatlpmfunction} ',' {G0-Gpylpm }',  'Location', 'EastOutside');

%%


ij= 4:9;
ix= 3:7;
ef=10:15;

figure()
plot(ef(ix),ij(ix-2))

