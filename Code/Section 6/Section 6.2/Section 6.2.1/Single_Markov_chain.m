% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Estimate a two-regime Markov-switching ADL (Autoregressive Distributed
%   Lag) model by maximum likelihood. The state variable s_t follows a
%   first-order Markov chain with two regimes {0, 1}. Both the slope
%   coefficients on the contemporaneous and lagged values of an exogenous
%   regressor xi_t and the residual standard deviation are allowed to
%   switch between regimes; the autoregressive coefficients on X_{t-i}
%   and the constant are common across regimes. Wald tests for regime
%   equality of beta and sigma parameters are also calculated. 
%
%   Model:
%       X_t = alpha
%             + sum_{j=0}^{q_lag} beta_j^{(s_t)} * xi_{t-j}
%             + sum_{i=1}^{p_lag} gamma_i        * X_{t-i}
%             + u_t,             u_t ~ N( 0, sigma_{s_t}^2 )
%
%   Regime dynamics are governed by transition probabilities p11 and p22
%   (the probability of remaining in regime 0 and 1 respectively).
%
%   External dependencies:
%       - Data.xlsx (input data)
%       - Function_1.m (negative log-likelihood with Hamilton filter)
% =========================================================================

clear;
global X Z_reg p_lag q_lag

% -------------------------------------------------------------------------
% Lag specification
% -------------------------------------------------------------------------
q_lag = 0;     % number of lags of the exogenous variable xi
p_lag = 10;     % number of autoregressive lags of the dependent variable X

% -------------------------------------------------------------------------
% Load data and discard the first 99 observations to skip the
% burn-in / transient phase of the upstream construction.
% -------------------------------------------------------------------------
data       = readmatrix('Data.xlsx', 'Sheet', 'Tabelle2', 'Range', 'B2');
dates_raw  = readtable('Data.xlsx', 'Sheet', 'Tabelle2');
dates_all  = datetime(dates_raw{:, 1});

data  = data(100:end, :);
dates = dates_all(100:end);

X_raw  = data(:, 1);   % dependent variable
xi_raw = data(:, 4);   % exogenous regressor

% -------------------------------------------------------------------------
% Build the design matrix Z_reg.
% Effective sample size is T = T_raw - max_lag.
% -------------------------------------------------------------------------
max_lag = max(p_lag, q_lag);
T_raw   = length(X_raw);
T       = T_raw - max_lag;

X         = X_raw(max_lag+1:end);
dates_eff = dates(max_lag+1:end);

% Lags of xi (current value + q_lag lagged values).
Z_xi = zeros(T, q_lag+1);
for j = 0:q_lag
    Z_xi(:, j+1) = xi_raw(max_lag+1-j : T_raw-j);
end

% Lags of X.
Z_x = zeros(T, p_lag);
for i = 1:p_lag
    Z_x(:, i) = X_raw(max_lag+1-i : T_raw-i);
end

% Final regressor matrix: [constant, xi-lags, X-lags].
Z_reg = [ones(T,1), Z_xi, Z_x];

% -------------------------------------------------------------------------
% OLS estimate (used as starting value for the ML estimator)
% -------------------------------------------------------------------------
beta_ols = inv(Z_reg'*Z_reg) * Z_reg' * X;
resid    = X - Z_reg * beta_ols;

fprintf('\nOLS Estimation:\n');
fprintf('  alpha  = %f\n', beta_ols(1));
for j = 0:q_lag
    fprintf('  beta%d  = %f\n', j, beta_ols(1+j+1));
end
for i = 1:p_lag
    fprintf('  gamma%d = %f\n', i, beta_ols(1+(q_lag+1)+i));
end

% -------------------------------------------------------------------------
% Initial values for the ML estimator (in transformed parameterisation)
% sigma_{0,1} are perturbed slightly to break symmetry between regimes.
% -------------------------------------------------------------------------
sigma0 = (((resid)'*(resid))/(length(X)-2))^0.5;
sigma1 = (((resid)'*(resid))/(length(X)-2))^0.5 + 0.000001;
p11_0  = 0.95;     % initial Pr(s_t = 0 | s_{t-1} = 0)
p22_0  = 0.95;     % initial Pr(s_t = 1 | s_{t-1} = 1)

% Two-state transition matrix P (columns sum to 1).
P = [p11_0      1-p22_0;
     1-p11_0   p22_0];

% Transformations: log for variances, logit for transition probabilities.
sigma_0tr = log(sigma0);
sigma_1tr = log(sigma1);
p11_tr    = log(p11_0/(1-p11_0));
p22_tr    = log(p22_0/(1-p22_0));

% Stack initial parameters in transformed space:
%   [ alpha, beta_0^{(0)}, beta_0^{(1)}, ..., beta_qlag^{(0)}, beta_qlag^{(1)},
%     gamma_1, ..., gamma_plag,
%     log(sigma0), log(sigma1), logit(p11), logit(p22) ]
thtr0 = beta_ols(1);
for j = 0:q_lag
    b_j = beta_ols(1+j+1);
    thtr0 = [thtr0, b_j, b_j + 0.001];        % perturbed regime-1 start
end
for i = 1:p_lag
    thtr0 = [thtr0, beta_ols(1+(q_lag+1)+i)];
end
thtr0 = [thtr0, sigma_0tr, sigma_1tr, p11_tr, p22_tr];

% -------------------------------------------------------------------------
% Maximum-likelihood estimation via fminsearch
% The objective is the negative log-likelihood returned by
% Function_1 with the second argument set to 1 (signaling that the
% parameters are still in transformed space and must be back-transformed
% inside the function before the filter is run).
% -------------------------------------------------------------------------
options  = optimset('Display','iter','MaxFunEvals',10000, ...
                    'MaxIter',10000,'TolFun',0.00001);
thtropt  = fminsearch('Function_1', thtr0, options, 1);

% -------------------------------------------------------------------------
% Back-transform the optimum to the natural parameterisation.
% -------------------------------------------------------------------------
n_par = length(thtr0);
n_reg = 1 + 2*(q_lag+1) + p_lag;   % index of the LAST regression coefficient

thopt = thtropt(:);
thopt(n_reg+1) = exp(thtropt(n_reg+1));                                   % sigma0
thopt(n_reg+2) = exp(thtropt(n_reg+2));                                   % sigma1
thopt(n_reg+3) = exp(thtropt(n_reg+3))/(1+exp(thtropt(n_reg+3)));         % p11
thopt(n_reg+4) = exp(thtropt(n_reg+4))/(1+exp(thtropt(n_reg+4)));         % p22

p11_opt = thopt(n_reg+3);
p22_opt = thopt(n_reg+4);

% Update the transition matrix with the ML estimates.
P = [p11_opt    1-p22_opt;
     1-p11_opt  p22_opt];

% -------------------------------------------------------------------------
% Numerical Hessian (finite differences) -> standard errors
% -------------------------------------------------------------------------
x0 = thopt;  n = length(x0);
H0 = zeros(n, n);
auxi = diag(x0 * 1e-4);
for i = 1:n
    for j = 1:n
        H0(i,j) = (feval('Function_1', x0+auxi(:,i)+auxi(:,j), 0) - ...
                   feval('Function_1', x0+auxi(:,i),           0) - ...
                   feval('Function_1', x0+auxi(:,j),           0) + ...
                   feval('Function_1', x0,                      0)) ...
                  /(auxi(i,i) * auxi(j,j));
    end
end
informd = inv(H0);
sgd     = sqrt(diag(informd));
format long;

% =========================================================================
% Wald tests for regime equality of beta_0, beta_1, and sigma
% =========================================================================
% Asymptotic covariance matrix (already computed above as inv(Hessian)).
V = informd;

% --- H0: beta_0^{(0)} = beta_0^{(1)} -------------------------------------
R1       = zeros(1, n_par);
R1(2)    =  1;
R1(3)    = -1;
diff1    = R1 * x0;
W1       = diff1^2 / (R1 * V * R1');
p_W1     = 1 - chi2cdf(W1, 1);

% --- H0: beta_1^{(0)} = beta_1^{(1)} (only when q_lag >= 1) --------------
if q_lag >= 1
    R2    = zeros(1, n_par);
    R2(4) =  1;
    R2(5) = -1;
    diff2 = R2 * x0;
    W2    = diff2^2 / (R2 * V * R2');
    p_W2  = 1 - chi2cdf(W2, 1);
end

% --- H0: sigma_{(0)} = sigma_{(1)} ---------------------------------------
R3            = zeros(1, n_par);
R3(n_reg+1)   =  1;
R3(n_reg+2)   = -1;
diff3         = R3 * x0;
W3            = diff3^2 / (R3 * V * R3');
p_W3          = 1 - chi2cdf(W3, 1);

% --- Joint test of all regime-switching parameters -----------------------
if q_lag >= 1
    R_joint = [R1; R2; R3];
else
    R_joint = [R1; R3];
end
diff_joint = R_joint * x0;
W_joint    = diff_joint' * ((R_joint * V * R_joint') \ diff_joint);
q_joint    = size(R_joint, 1);
p_joint    = 1 - chi2cdf(W_joint, q_joint);

% --- Report --------------------------------------------------------------
disp('=========================================================================')
disp('Wald Tests for Regime Equality of Switching Parameters')
disp('=========================================================================')
fprintf('H0: beta_0^{(0)} = beta_0^{(1)}        W = %8.4f   df = 1   p = %.4f\n', W1, p_W1);
if q_lag >= 1
fprintf('H0: beta_1^{(0)} = beta_1^{(1)}        W = %8.4f   df = 1   p = %.4f\n', W2, p_W2);
end
fprintf('H0: sigma_{(0)}  = sigma_{(1)}         W = %8.4f   df = 1   p = %.4f\n', W3, p_W3);
fprintf('H0: joint (all regime-switching pars)  W = %8.4f   df = %d   p = %.4f\n', ...
        W_joint, q_joint, p_joint);
disp('=========================================================================')

% -------------------------------------------------------------------------
% Run the filter once more to get the filtered (xitt) and one-step-ahead
% predicted (xit1t) regime probabilities, then run Kim's smoother to get
% smoothed probabilities xitT.
% -------------------------------------------------------------------------
[lfv, xitt, xit1t] = Function_1(x0, 0);

xitT = zeros(T-1, 2);
xitT = [xitT; xitt(T, :)];
for t = 1:T-1
    a = xitt(T-t, :)' .* (P' * (xitT(T-t+1, :)' ./ xit1t(T+1-t, :)'));
    xitT(T-t, :) = a';
end

% -------------------------------------------------------------------------
% Build a labelled results table
% -------------------------------------------------------------------------
names_var = cell(n_par, 1);
k = 1;
names_var{k} = sprintf('alpha          '); k = k+1;
for j = 0:q_lag
    names_var{k} = sprintf('beta%d-Regime 0 ', j); k = k+1;
    names_var{k} = sprintf('beta%d-Regime 1 ', j); k = k+1;
end
for i = 1:p_lag
    names_var{k} = sprintf('gamma%d         ', i); k = k+1;
end
names_var{k} = sprintf('sigma-Regime 0 '); k = k+1;
names_var{k} = sprintf('sigma-Regime 1 '); k = k+1;
names_var{k} = sprintf('p11            '); k = k+1;
names_var{k} = sprintf('p22            ');

coefficient = x0;
Std_Error   = sgd;
t_statistic = coefficient ./ Std_Error;
p_value     = 1 - normcdf(abs(t_statistic), 0, 1);  p_value = 2*p_value;

disp('=========================================================================')
disp('Markov-Switching Regression Results')
disp('=========================================================================')
fprintf('Model: X_t = alpha + sum_{j=0}^{%d} beta_j^{(s_t)} * xi_{t-j}\n', q_lag);
fprintf('             + sum_{i=1}^{%d} gamma_i * X_{t-i} + u_t\n', p_lag);
disp('=========================================================================')
TABLA1 = table(coefficient, Std_Error, t_statistic, p_value, 'RowNames', names_var)

% Expected duration of each regime: E[duration] = 1/(1 - p_ii).
disp('=========================================================================')
fprintf('Expected Duration Regime 0: %.2f periods\n', 1/(1-p11_opt));
fprintf('Expected Duration Regime 1: %.2f periods\n', 1/(1-p22_opt));
fprintf('Log-Likelihood: %.4f\n', -lfv);
disp('=========================================================================')

% -------------------------------------------------------------------------
% Plot filtered and smoothed regime probabilities
% -------------------------------------------------------------------------
figure;
subplot(2,1,1);
plot(dates_eff, xitt(:,2), 'b', 'LineWidth', 0.8); hold on;
plot(dates_eff, xitT(:,2), 'r', 'LineWidth', 0.8);
title('Probability of being in Regime 1 (S_t=1)');
legend('\xi_{t|t} (filtered)', '\xi_{t|T} (smoothed)');
ylabel('Probability');
xlabel('Date');
ylim([0 1]);
grid on;

subplot(2,1,2);
plot(dates_eff, xitt(:,1), 'b', 'LineWidth', 0.8); hold on;
plot(dates_eff, xitT(:,1), 'r', 'LineWidth', 0.8);
title('Probability of being in Regime 0 (S_t=0)');
legend('\xi_{t|t} (filtered)', '\xi_{t|T} (smoothed)');
ylabel('Probability');
xlabel('Date');
ylim([0 1]);
grid on;