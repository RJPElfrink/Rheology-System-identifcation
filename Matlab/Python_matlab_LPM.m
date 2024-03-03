clear all; close all


%double=load('arrays.mat');

data=load('Matlab_multisinewindow.mat');

method.order        =   2; %order of the local polynomial approximation (default 2) 
method.dof          =   1; %degrees of freedom of the (co-)variance estimates = equivalent number of 
%                                               independent experiments - 1 (default ny) 
method.transient    = 	1;%determines the estimation of the transient term (optional; default 1)  
%                                                   1: transient term is estimated 
%                                                   0: no transient term is estimated 
                                   
[CZ, Z, freq, G, CvecG, dof, CL] = FastLocalPolyAnal(data, method);

% FRF and its variance
G = squeeze(G).';
varGn = squeeze(CvecG.n).';              % noise variance
varGNL = squeeze(CvecG.NL).';            % total variance (noise + NL distortions)

% data output
output_data = struct('CZn', [], 'varCvecGn', [],'varCvecGNL', [], 'freq', [], 'G', []);
output_data.CZn = squeeze(CZ.n);                   % noise covariance matrix of the sample mean Z.n over the periods 
output_data.varCvecGn = squeeze(CvecG.n);           % noise variance covariance matrix of vec(G)
output_data.varCvecGNL = squeeze(CvecG.NL);         % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
output_data.freq = squeeze(freq);                  % frequency of the excited harmonics; size 1 x F
output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 





save('LPM_multisinewindow.mat', 'output_data');

%G1=G;varGn1=varGn;varGNL1=varGNL;freq1=freq;
%%
% estimated BLA, its noise and total variances
figure(2)
%semilogx(freq, db(G), 'k', freq, db(varGn)/2, 'g', freq, db(varGNL)/2, 'r')
semilogx(freq, db(G), 'k', freq,  db(varGNL)/2, 'r',freq1, db(G1), 'b', freq1, db(varGNL1)/2, 'g')
xlabel('Frequency (Hz)')
ylabel('{\itG}_{BLA} (dB)')
title('Estimated BLA and its variances')
legend('{\itG}_{BLA}',  'total variance','{\itG}_{BLA1}',  'total variance 1', 'Location', 'EastOutside');
zoom on
shg

