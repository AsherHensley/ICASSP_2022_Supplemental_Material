%RUN_EXPERIMENT() Yule-Simon Simulation Experiment Script
%   RUN_EXPERIEMT() Simulates Yule-Simon Gaussian process and applies
%   online inference algorithm for sample realizations. Results are
%   averaged and compared to posterior Cramer-Rao Lower Bound (CRLB)
%   for log2m = 2:2:10.
%
%   Author: Asher A. Hensley
%
%   Revision History
%       1.0     10.02.2021      Initial release

% Clean Up
clear
close all
clc

% Set Experiment Configuration
RANDOM_SEED = 1;
ntrials = 1000;         % Number of Monte-Carlo runs
nsamples = 2000;        % Number of samples in each sequence
log2m_list = 2:2:10;     % List of active branches to run
alpha = 1;              % Yule-Simon parameter (ground truth)
a0 = 1;                 % Gamma prior shape factor (ground truth)
b0 = 1;                 % Gamma prior rate factor (ground truth)

% Init Aggregators
agg_MSE = zeros(nsamples,length(log2m_list));
agg_CRLB = zeros(nsamples,length(log2m_list));

% Init Waitbar
hw = waitbar(0,'...');

% Loop Over Number of Active Branchs
tic
for ii = 1:length(log2m_list)
    
    % Reset Random Number Generator
    rng(RANDOM_SEED);
    
    % Set Active Branch Count
    log2m = log2m_list(ii);
    
    % Process Sample Realizations
    for kk = 1:ntrials
        
        % Update Waitbar
        k_str = num2str(kk);
        N_str = num2str(ntrials);
        m_str = num2str(log2m);
        msg = ['Processing Trial ',k_str,' of ',N_str,', log2m = ',m_str];
        waitbar(kk/ntrials,hw,msg)
        
        % Generate Gaussian Process Realization
        [y,lambda,x,xc] = sample_realization(nsamples,a0,b0,alpha);
        
        % Initialize
        state0 = initialize_state(y(1),'log2m',log2m);
        
        % Run Inference Algorithm
        [result,state] = do_inference(y,state0);
        
        % Run Kalman Filter with Estimated Sigmas
        P0 = b0 / a0;
        R = 1 ./ result.mu_update;
        mu_hat = kalman_filter(y,P0,R);
        
        % Run Kalman Filter with True Sigmas
        P0 = b0 / a0;
        R = 1 ./ lambda(x);
        [~,CRLB] = kalman_filter(y,P0,R);
        
        % Update Results
        agg_MSE(:,ii) = agg_MSE(:,ii) + (mu_hat' - 0).^2 ;
        agg_CRLB(:,ii)  = agg_CRLB(:,ii) + CRLB';
        
    end
end
delete(hw)
toc

% Normalize
agg_MSE = agg_MSE / ntrials;
agg_CRLB = agg_CRLB / ntrials;

% Error Gain
G = 10*log10(agg_MSE) - 10*log10(agg_CRLB);

% Plot Results for Largest m
figure
subplot(121)
semilogy(agg_MSE(:,end),'k')
hold on
semilogy(agg_CRLB(:,end),'k--')
grid on
legend('MSE','CRLB')
subplot(122)
plot(G(:,end),'k')
grid on
set(gcf,'outerposition',[441   574   560   302])

% Print Average Error Gain
fprintf(1,'%s\t%s\n','log2m','Error Gain')
fprintf(1,'%s\n','------------------')
for kk = 1:length(log2m_list)
    fprintf(1,'%u\t%1.2f\n',log2m_list(kk),mean(G(500:end,kk)))
end

% Save Results 
save ICASSP_2022_RESULTS.mat



