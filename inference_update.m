function state = inference_update(yt,state)
%INFERENCE_UPDATE() Do Yule-Simon Update Step
%   STATE = INFERENCE_UPDATE(STATE) updates state based on update step
%   and does state update estimate.
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Update State Parameters
mask = state.q > 0;
state.a(mask) = state.a(mask) + 0.5;
state.b(mask) = state.b(mask) + 0.5 * yt^2;

% Update
m = state.m;
for kk = 1:state.m
    
    % Same State
    lambda = state.a(kk) / state.b(kk);
    dof = 2 * state.a(kk);
    state.q(kk) = state.F(yt, lambda, dof) * state.q(kk);

    % New State
    lambda = state.a(kk + m) / state.b(kk + m);
    dof = 2 * state.a(kk + m);
    state.q(kk + m) = state.F(yt, lambda, dof) * state.q(kk+m);
    
end

% Normalize
state.q = state.q / sum(state.q);

% Update State Counters
state.n(state.q > 0) = state.n(state.q > 0) + 1;

% Sort
[~,I] = sort(state.q,'descend');
state.q = state.q(I);
state.a = state.a(I);
state.b = state.b(I);
state.n = state.n(I);

% Prune
state.p = state.q(1:m) / sum(state.q(1:m));
state.a(m+1:end) = state.a0;
state.b(m+1:end) = state.b0;
state.n(m+1:end) = 0;

% Save Most Likely Trajectory Count
state.count = state.n(1);

% State Update Estimate
switch state.method
    case 'average'
        at = state.a(1:m);
        bt = state.b(1:m);
        mask = state.p > 0;
        temp = at(mask) ./ bt(mask);
        state.mu_update = temp'* state.p(mask);
        temp = at(mask) ./ bt(mask).^2;
        state.std_update = sqrt(temp'* state.p(mask));
    case 'max'
        temp = state.a(1) ./ state.b(1);
        state.mu_update = temp;
        temp = state.a(1) ./ state.b(1)^2;
        state.std_update = sqrt(temp);
    otherwise
        error('Unkown Method')
end

