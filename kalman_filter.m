function [mu_hat,P_hat] = kalman_filter(y,P0,R)
%KALMAN_FILTER() Kalman Filtering Algorithm
%   MU_HAT = KALMAN_FILTER(Y,P0,R) runs scalar Kalman filtering algorithm
%   on measurement sequence Y with initial state variance P0 and sigma
%   sequence R. It is assumed the state variable is always 0 with zero
%   process noise, therefore y(k) ~ Normal(0,R(k)).
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Init
Ny = length(y);
mu_hat = zeros(1,Ny);
P_hat = zeros(1,Ny);
P = P0;
mu_hat(1) = y(1);
P_hat(1) = P;

% Run Filter
for t = 2:Ny
    S = P + R(t);
    K = P / S;
    innov = y(t) - mu_hat(t-1);
    P = (1-K) * P;
    mu_hat(t) = mu_hat(t-1) + K * innov;
    P_hat(t) = P;
end