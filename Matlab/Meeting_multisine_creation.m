% test Rutger
clear all
close all
clc

%% system

g = 2.5; lambda = 1.5;
sys = tf(g,[1, 1/lambda])

%% crest factor
P = 1;
N = 2000;
fs = 20; %Hz
f0 = fs/N;
Ts = 1/fs;
t = 0:Ts:(P*N-1)*Ts;
f = 0:f0:fs-f0;

Umss = zeros(N,1);
F = 1000;
k = 1:F;
phases = -k.*(k-1).*pi/F;
Umss = zeros(N,1);
Umss(2:F+1) = exp(1i*phases);
umss = 2*real(ifft(Umss)); umss = umss/std(umss);

u = repmat(umss,P,1);
%% generate output

y = lsim(sys,u,t);

figure; plot(u(end-N+1:end))
figure; plot(y(end-N+1:end))
%% FRF

% Frequency Response Function (FRF) Calculation
Y = fft(y(end-N+1:end));
U = fft(u(end-N+1:end));
G = Y./U;

% Plot for the used frequencies in Gd
figure; semilogx(f,db(U))
xlim([0,fs/2])

% Magnitude and Phase for FRF
G_magnitude = abs(G); % Magnitude
G_phase = angle(G); % Phase in radians

% Plot Magnitude and Phase Response for both sys and FRF
figure;

% Bode Plot for Designed System
[mag,phase,wout] = bode(sys,{min(f)*2*pi, max(f)*2*pi});
mag = squeeze(mag);
phase = squeeze(phase);

% Overlay FRF Magnitude
subplot(2,1,1); % Magnitude plot
semilogx(wout/(2*pi), 20*log10(mag), 'b', f, 20*log10(G_magnitude), 'r--');
ylim([-40,40])
ylabel('Magnitude (dB)');
legend('System TF', 'Simulated FRF');
title('Magnitude Response');

% Overlay FRF Phase
subplot(2,1,2); % Phase plot
semilogx(wout/(2*pi), (phase), 'b', f, rad2deg(G_phase), 'r--');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
legend('System TF', 'Simulated FRF');
title('Phase Response');

