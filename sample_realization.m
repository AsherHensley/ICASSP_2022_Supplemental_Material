function [y,lambda,x,xc] = sample_realization(N,a0,b0,alpha)
%SAMPLE_REALIZATION() Generates Yule-Simon Gaussian Process Realization
%   SAMPLE_REALIZATION(N,A0,B0,ALPHA) returns N sample Yule-Simon Gaussian
%   process realization Y with Yule-Simon parameter ALPHA and precisions 
%   sampled from a gamma prior with shape factor A0 and rate factor B0. 
%   Additionally the true precisions are returned in the vector LAMBDA 
%   [1-by-numStates], partition indicators are returned in the vector X
%   [1-by-N], and the sample counters are return in the vector XC [1-by-N].
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Init
x = zeros(1,N);
xc = zeros(1,N);

% Assign First Partition
x(1) = 1;
xc(1) = 1;
n = 1;
lambda = gamrnd(a0,1/b0);

% Process Remaining Samples
for kk = 2:N
    
    % Sample Yule-Simon State Transition
    isNewState = rand <  alpha / (alpha + n);
    
    % Apply State Update Logic
    if isNewState
        
        % Increment State Index
        x(kk) = x(kk-1) + 1;
        
        % Reset Counter
        n = 1;
        
        % Append New Lambda
        lambda = [lambda,gamrnd(a0,1/b0)];
        
    else      
        
        % Assign Same State index
        x(kk) = x(kk-1);
        
        % Increment Counter
        n = n + 1;
        
    end
    
    % Update Running Counter
    xc(kk) = n;
    
end

% Generate Gaussian Process Realization
y = 1./sqrt(lambda(x)) .* randn(1,N);


