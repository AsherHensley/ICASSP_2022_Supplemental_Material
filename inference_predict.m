function state = inference_predict(state)
%INFERENCE_PREDICT() Do Yule-Simon Predict Step
%   STATE = INFERENCE_PREDICT(STATE) updates state based on predict step
%   and does state prediction estimate.
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Branch Split Probabilities
for kk = 1:state.m
    state.q(kk) = state.n(kk) / (state.n(kk) + state.alpha) * state.p(kk);
    state.q(kk + state.m) = state.alpha / (state.n(kk) + state.alpha) * state.p(kk);
end

% State Prediction Estimate
switch state.method
    case 'average'
        mask = state.q > 0;
        temp = state.a(mask) ./ state.b(mask);
        state.mu_predict = temp'* state.q(mask);
        temp = state.a(mask) ./ state.b(mask).^2;
        state.std_predict = sqrt(temp'* state.q(mask));     
    case 'max'
        temp = state.a(1) ./ state.b(1);
        state.mu_predict = temp;
        temp = state.a(1) ./ state.b(1).^2;
        state.std_predict = sqrt(temp); 
    otherwise
        error('Unknown Method')
end



