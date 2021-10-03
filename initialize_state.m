function state = initialize_state(y0,varargin)
%INITIALIZE_STATE() Set Initial State for Yule-Simon Inference Algorithm
%   STATE = INITIALIZE_STATE(Y0) returns initial state for Yule-Simon
%   inference algorithm with default parameters.
%
%   STATE = INITIALIZE_STATE(Y0,'PARAM',VALUE) returns initial state for 
%   Yule-Simon inference algorithm with parameters configured using 'PARAM'
%   VALUE pairs:
%       'log2m'     sets number of active hypotheses = 2^log2m (default: 6)
%       'method'    sets inference method - can be 'average' or 'max' 
%                   (default: 'max') 
%       'alpha'     sets Yule-Simon parameter (default: 1)
%       'a0'        sets Gamma prior shape factor (default: 1)
%       'b0'        sets Gamma prioe rate factor (default: 1)
%
%   Example: Set initial state with log2m = 10
%       state0 = initialize_state(y(1),'log2m',10);
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Default Parameters
m = 2^6;
a0 = 1;
b0 = 1;
alpha = 1;
method = 'max';

% Varargin
if ~isempty(varargin)
    for kk = 1:2:length(varargin)
        switch varargin{kk}
            case 'log2m'
                m = 2^varargin{kk+1};
            case 'method'
                method = varargin{kk+1};
            case 'alpha'
                alpha = varargin{kk+1};
            case 'a0'
                a0 = varargin{kk+1};
            case 'b0'
                b0 = varargin{kk+1};
            otherwise
                error('Unknown Parameter')
        end
    end
end

% State Structure
state.m = m;
state.a0 = a0;
state.b0 = b0;
state.alpha = alpha;
state.method = method;
state.p = zeros(m,1);
state.q = zeros(2*m,1);
state.a = zeros(2*m,1) + a0;
state.b = zeros(2*m,1) + b0;
state.n = zeros(2*m,1);

% Likelihood
state.F = @(z,L,dof)exp(gammaln(dof/2+0.5) - gammaln(dof/2)) * ...
    sqrt(L/pi/dof)*(1+L/dof*z.^2).^(-dof/2-0.5);

% Initial State
state.a(1) = state.a(1) + 1/2;
state.b(1) = state.b(1) + y0^2;
state.n(1) = 1;
state.p(1) = 1;
state.mu_update = state.a(1) / state.b(1);
state.std_update = sqrt(state.a(1) / state.b(1)^2);
state.mu_predict = nan;
state.std_predict = nan;
state.count = 1;

