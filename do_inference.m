function [result,state] = do_inference(y,state,varargin)
%DO_INFERENCE() Run Yule-Simon Inference Algorithm
%   [RESULT,STATE] = DO_INFERENCE(Y,STATE) runs sequential inference 
%   algorithm on observation sequence Y with initial state STATE.
%
%   [RESULT,STATE] = DO_INFERENCE(Y,STATE,USE_WAITBAR) runs sequential 
%   inference algorithm on observation sequence Y with initial state STATE
%   with waitbar controlled by USE_WAITBAR flag (default: FALSE).
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Varargin
use_waitbar = false;
if nargin>2
    use_waitbar = varargin{1};
end

% Init Results Structure
Ny = length(y);
result.mu_predict = nan(1,Ny+1);
result.std_predict = nan(1,Ny+1);
result.mu_update = nan(1,Ny);
result.std_predict = nan(1,Ny);
result.count = nan(1,Ny);

% Do Predict Step
state = inference_predict(state);

% Save Results
result.mu_predict(2) = state.mu_predict;
result.std_predict(2) = state.std_predict;
result.mu_update(1) = state.mu_update;
result.std_update(1) = state.std_update;
result.count(1) = state.count;

% Init Waitbar
if use_waitbar
    hw = waitbar(0,'Processing Data...');
end

% Process Measurements
for kk = 2:Ny
    
    % Do Update Step
    state = inference_update(y(kk),state);
    
    % Do Predict Step
    state = inference_predict(state);
    
    % Save Results
    result.mu_predict(kk+1) = state.mu_predict;
    result.std_predict(kk+1) = state.std_predict;
    result.mu_update(kk) = state.mu_update;
    result.std_update(kk) = state.std_update;
    result.count(kk) = state.count;
    
    % Update Waitbar
    if use_waitbar
        waitbar(kk/Ny,hw,'Processing Data...');
    end
    
end

% Delete Waitbar
if use_waitbar
    delete(hw)
end



