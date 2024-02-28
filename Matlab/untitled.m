clear all; close all


%double=load('arrays.mat');



data=load('Matlab_test_chirp.mat');
method.dof = 6;                         % degrees of freedom of the noise variance estimate
method.transient = 1;                   % transient removal is needed because:
                                        %	1. the steady state response is calculated
                                        %	2. white input-output measurement noise

                                   
[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data, method);

G_LPM=squeeze(squeeze(G));

save('LPM_matlab.mat', 'G_LPM');

% comparison true and estimated input-output noise variances
% true input noise var = stdu^2 / (P*N): averaging over P periods + scaling by N for Fourier coefficients 
% true output noise var = stdy^2 / (P*N): averaging over P periods + scaling by N for Fourier coefficients 
figure(1)
subplot(211)
plot(freq, db(squeeze(CZ.n(2,2,:)))/2, 'g', freq, db(stdu*ones(size(freq))/sqrt(P*N)), 'k');
xlabel('Frequency (Hz)')
ylabel('Input noise variance (dB)')
legend('estimate', 'true value', 'Location', 'EastOutside');
subplot(212)
plot(freq, db(squeeze(CZ.n(1,1,:)))/2, 'g', freq, db(stdy*ones(size(freq))/sqrt(P*N)), 'k');
xlabel('Frequency (Hz)')
ylabel('Output noise variance (dB)')
legend('estimate', 'true value', 'Location', 'EastOutside');
