

function output_data = nieuwe(u,y,r,N,fs,ExcitedHarm,order,dof,transient)
% data 




data = struct('u', [], 'y', [], 'r', [], 'N', [], 'Ts', [], 'ExcitedHarm', []);
data.u = u;                             % row index is the input number; the column index the time instant (in samples) 
data.y = y;                             % row index is the output number; the column index the time instant (in samples) 
%data.r = r;                             % one period of the reference signal is sufficient 
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



[CZ, Z, freq, G, CvecG, dof, CL] = RobustLocalPolyAnal(data,method);


% data output
%output_data = struct('CZn', [], 'varCvecGn', [],'varCvecGNL', [], 'freq', [], 'G', []);
output_data = struct('G', []);
%output_data.CZn = squeeze(CZ.n);                   % noise covariance matrix of the sample mean Z.n over the periods 
%output_data.varCvecGn = squeeze(CvecG.n);           % noise variance covariance matrix of vec(G)
%output_data.varCvecGNL = squeeze(CvecG.NL);         % total variance (noise + nonlinear distortion) covariance matrix of vec(G) 
%output_data.freq = squeeze(freq);                  % frequency of the excited harmonics; size 1 x F
output_data.G = squeeze(G);                        % estimated frequency response matrix; size ny x nu x F 


end
