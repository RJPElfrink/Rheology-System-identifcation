%% test LPM with Schroeder Multisine

clear all
%close all
clc

%% load data

data = load('Matlab_input_chirp_test.mat');
%data = load('Matlab_input_randmulti_noise_20.mat');

%% variables and data

N = double(data.N);
r = double(data.r); rstd = std(r); rMax = 1.1*max(abs(r));
Ts = double(data.Ts);

fs = 1/Ts;
f0 = fs/N;
lines = 2:N/2-1; nLines = length(lines);
f= 0:f0:fs-f0;
t = 0:Ts:(N-1)*Ts;

%% Define phases

phaserand=2*pi*rand(nLines,1);
k1=1;
k2=nLines;
k = k1:k2;
phasemulti = -k.*(k-1).*pi/nLines;

%% generate input


nUp = 10; % generate data oversampled such that simulation is accurate - otherwhise we have significant zoh effects with lsim simulation
tUp = 0:Ts/nUp:(nUp*N-1)*Ts/nUp;
R = zeros(N*nUp,1);

%select phaserand or phasemulti 
R(lines) = exp(1i*phaserand);
rUp = 2*real(ifft(R)); rUp = rstd*rUp/std(rUp); rUp=rUp.';
rTest=rUp;

% improve crest factor 
figure; plot(tUp,rUp);
for iter=1:1000
rUp = min(rUp,rMax); rUp=max(rUp,-rMax);
Rtemp = fft(rUp);
R = zeros(N*nUp,1);
R(lines) = Rtemp(lines); R(lines) = R(lines)./abs(R(lines));
rUp = 2*real(ifft(R)); rUp = rstd*rUp/std(rUp); rUp=rUp.';
crest=max(abs(rUp))/LocalRMS(rUp)
end
hold on; plot(tUp,rUp); title('Reference')
legend('not optimized','Crest optimized')

%% generate data

% g/(s+1/L). Waarbij g gelijk is aan 2.5 en L gelijk is aan 1.5

g = 2.5;
L = 1.5;
B = g;
A = [1 1/L];
sys = tf(B,A);
G0 = freqs(B,A,2*pi*f);
uUp = rUp;

yUp = lsim(sys,uUp,tUp); yUp=yUp';

% downsample data
u = uUp(1:nUp:end);
r = rUp(1:nUp:end);
%r = rTest(1:nUp:end);
y = yUp(1:nUp:end);

Ytest=fft(y);Utest=fft(u);Gtest=Ytest./Utest;

%%


data_multi = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data_multi.u = u;                             % row index is the input number; the column index the time instant (in samples) 
data_multi.y = y;                             % row index is the output number; the column index the time instant (in samples) 
data_multi.r = r;                             % one period of the reference signal is sufficient 
data_multi.N = N;                             % number of samples in one period 
data_multi.Ts = Ts;                           % sampling period 
data_multi.ExcitedHarm = lines-1;             % excited harmonics multisine excitation


method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  

[CZ_m, Z_m, freq_m, G_m, CvecG_m, dof_m, CL_m] = FastLocalPolyAnal(data_multi, method);
% [CZ_m, Z_m, freq_m, G_m, CvecG_m, dof_m, CL_m] = ArbLocalPolyAnal(data_multi, method);

% FRF and its variance
G_multi_tran = squeeze(G_m).';
%% LPM
test = load('Matlab_input_chirp_test.mat');
data_chirp = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', [],'order',[],'dof',[],'transient',[]);
%data = struct('u', [], 'y', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data_chirp.u = test.u;                             % row index is the input number; the column index the time instant (in samples) 
data_chirp.y = test.y;                             % row index is the output number; the column index the time instant (in samples) 
data_chirp.r = test.r;                             % one period of the reference signal is sufficient 
data_chirp.N = test.N;                             % number of samples in one period 
data_chirp.Ts = test.Ts;                         % sampling period 
data_chirp.ExcitedHarm = test.ExcitedHarm;         % excited harmonics multisine excitation
%% no transient removal
method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	0;%determines the estimation of the transient term (optional; default 1)  

% [CZ_m, Z_m, freq_m, G_m, CvecG_m, dof_m, CL_m] = FastLocalPolyAnal(data_multi, method);
[CZ_m, Z_m, freq_m, G_m, CvecG_m, dof_m, CL_m] = FastLocalPolyAnal(data_chirp, method);

% FRF and its variance
G_multi_notran = squeeze(G_m).';

%% plots
dif=ones(1,998)*0.005-zeros(1,998);
figure; hold on
plot(f(lines),db(G0(lines)),'-o')
plot(f(lines),db(Gtest(lines)),'-+')
%plot(f(lines),db(G_multi_tran(lines-1)))
%plot(f(lines),db(G_multi_notran(lines-1)))
%plot(f(lines),db(G0(lines)-G_multi_tran(lines-1)))
plot(f(lines),db(G0(lines)-G_multi_notran(lines-1)))
plot(f(lines),db(G0(lines)-Gtest(lines)),'-<')
plot(f(lines),db(dif))
set(gca, 'XScale', 'log');

legend('G_0','Gtest ' ,'\hat{G} trans. est.','\hat{G} without trans. est.','resid \hat{G} trans. est.','resid \hat{G} without trans. est.','resid of G0-Gtest value')
%%
figure;

semilogx(f(lines),db(G0(lines)),'d',f(lines),db(G_multi_tran(lines-1)),'d',f(lines),db(G_multi_notran(lines-1)),'D',f(lines),db(G0(lines)-G_multi_tran(lines-1)),f(lines),db(G0(lines)-G_multi_notran(lines-1)))
title("Random multisine with crest optimization")
legend('G_0','G trans. est.','G without trans. est.','Bias G trans. est.','Bias G without trans. est.')

function x=LocalRMS(u)
% calculate the rms value of a signal
x=sqrt(sum(u.^2)/length(u));
end