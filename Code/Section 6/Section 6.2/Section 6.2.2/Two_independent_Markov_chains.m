% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Estimate a Markov-switching ADL model in which two independent
%   first-order Markov chains drive the parameters:
%
%       - Chain 1 (regimes 0 / 1) drives the slope coefficients
%         beta_j on the contemporaneous and lagged exogenous regressor xi.
%       - Chain 2 (regimes 0 / 1) drives the residual standard deviation
%         sigma.
%
%   Because the two chains are independent, the combined state space has
%   four regimes:
%       Regime 1: (beta-R0, sigma-R0)
%       Regime 2: (beta-R0, sigma-R1)
%       Regime 3: (beta-R1, sigma-R0)
%       Regime 4: (beta-R1, sigma-R1)
%
%   Model:
%       X_t = alpha
%             + sum_{j=0}^{q_lag} beta_j^{(s1_t)} * xi_{t-j}
%             + sum_{i=1}^{p_lag} gamma_i        * X_{t-i}
%             + u_t,             u_t ~ N(0, sigma_{s2_t}^2)
%
%   Pipeline mirrors Single_Markov_chain.m but uses Function_1 for two independent Markov chains
%   for the negative log-likelihood and a 4 x 4 transition matrix that is 
%   the Kronecker product of the two chains' 2 x 2 transition matrices.
%
%   External dependencies:
%       - Data.xlsx       (input data)
%       - Function_1.m    (negative log-likelihood with Hamilton filter)
% =========================================================================

clear;
global X Z_reg p_lag q_lag

% -------------------------------------------------------------------------
% Lag specification
% -------------------------------------------------------------------------
q_lag = 1;     % number of lags of the exogenous variable xi
p_lag = 10;    % number of autoregressive lags of the dependent variable X

% -------------------------------------------------------------------------
% Load data and discard the first 99 observations to skip the
% burn-in / transient phase of the upstream construction.
% -------------------------------------------------------------------------
data       = readmatrix('Data.xlsx', 'Sheet', 'Tabelle4', 'Range', 'B2');
dates_raw  = readtable( 'Data.xlsx', 'Sheet', 'Tabelle4');
dates_all  = datetime(dates_raw{:, 1});

data  = data(100:end, :);
dates = dates_all(100:end);

X_raw  = data(:, 1);   % dependent variable
xi_raw = data(:, 2);   % exogenous regressor

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
% Initial values for the four-state Markov chain.
% The 4 x 4 transition matrix is the Kronecker product of two
% independent 2 x 2 transition matrices governing chains 1 and 2.
% -------------------------------------------------------------------------
sigma0 = (((resid)'*(resid))/(length(X)-2))^0.5;
sigma1 = (((resid)'*(resid))/(length(X)-2))^0.5 + 0.000001;

% Chain 1 (drives beta) initial probabilities.
p11_0 = 0.95;
p22_0 = 0.95;
% Chain 2 (drives sigma) initial probabilities.
p11_1 = 0.9;
p22_1 = 0.9;

% Combined 4 x 4 transition matrix expressed entry-by-entry.
P = [p11_0*p11_1            p11_0*(1-p22_1)        (1-p22_0)*p11_1        (1-p22_0)*(1-p22_1);
     p11_0*(1-p11_1)        p11_0*p22_1            (1-p22_0)*(1-p11_1)    (1-p22_0)*p22_1;
     (1-p11_0)*p11_1        (1-p11_0)*(1-p22_1)    p22_0*p11_1            p22_0*(1-p22_1);
     (1-p11_0)*(1-p11_1)    (1-p11_0)*p22_1        p22_0*(1-p11_1)        p22_0*p22_1];

% Transformations: log for variances, logit for transition probabilities.
sigma_0tr = log(sigma0);
sigma_1tr = log(sigma1);
p11_0tr   = log(p11_0/(1-p11_0));
p22_0tr   = log(p22_0/(1-p22_0));
p11_1tr   = log(p11_1/(1-p11_1));
p22_1tr   = log(p22_1/(1-p22_1));

% Stack initial parameters in transformed space:
%   [ alpha, (beta_j^{(0)}, beta_j^{(1)})_{j=0..q_lag},
%     gamma_1..gamma_plag,
%     log(sigma0), log(sigma1),
%     logit(p11_0), logit(p22_0), logit(p11_1), logit(p22_1) ]
thtr0 = beta_ols(1);
for j = 0:q_lag
    b_j = beta_ols(1+j+1);
    thtr0 = [thtr0, b_j, b_j + 0.001];
end
for i = 1:p_lag
    thtr0 = [thtr0, beta_ols(1+(q_lag+1)+i)];
end
thtr0 = [thtr0, sigma_0tr, sigma_1tr, p11_0tr, p22_0tr, p11_1tr, p22_1tr];

% -------------------------------------------------------------------------
% Maximum-likelihood estimation via fminsearch
% The objective is the negative log-likelihood returned by
% Function_1 with the second argument set to 1 (signaling that the
% parameters are still in transformed space and must be back-transformed
% inside the function before the filter is run).
% -------------------------------------------------------------------------
options = optimset('Display','iter','MaxFunEvals',10000, ...
                   'MaxIter',10000,'TolFun',0.00001);
thtropt = fminsearch('Function_1', thtr0, options, 1);

% -------------------------------------------------------------------------
% Back-transform the optimum to the natural parameterisation.
% -------------------------------------------------------------------------
n_par = length(thtr0);
n_reg = 1 + 2*(q_lag+1) + p_lag;     % index of the LAST regression coefficient

thopt = thtropt(:);
thopt(n_reg+1) = exp(thtropt(n_reg+1));                                 % sigma0
thopt(n_reg+2) = exp(thtropt(n_reg+2));                                 % sigma1
thopt(n_reg+3) = exp(thtropt(n_reg+3))/(1+exp(thtropt(n_reg+3)));       % p11_0
thopt(n_reg+4) = exp(thtropt(n_reg+4))/(1+exp(thtropt(n_reg+4)));       % p22_0
thopt(n_reg+5) = exp(thtropt(n_reg+5))/(1+exp(thtropt(n_reg+5)));       % p11_1
thopt(n_reg+6) = exp(thtropt(n_reg+6))/(1+exp(thtropt(n_reg+6)));       % p22_1

p11_0opt = thopt(n_reg+3);
p22_0opt = thopt(n_reg+4);
p11_1opt = thopt(n_reg+5);
p22_1opt = thopt(n_reg+6);

% Updated the 4 x 4 transition matrix with the ML estimates.
P = [p11_0opt*p11_1opt          p11_0opt*(1-p22_1opt)      (1-p22_0opt)*p11_1opt      (1-p22_0opt)*(1-p22_1opt);
     p11_0opt*(1-p11_1opt)      p11_0opt*p22_1opt          (1-p22_0opt)*(1-p11_1opt)  (1-p22_0opt)*p22_1opt;
     (1-p11_0opt)*p11_1opt      (1-p11_0opt)*(1-p22_1opt)  p22_0opt*p11_1opt          p22_0opt*(1-p22_1opt);
     (1-p11_0opt)*(1-p11_1opt)  (1-p11_0opt)*p22_1opt      p22_0opt*(1-p11_1opt)      p22_0opt*p22_1opt];

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

% -------------------------------------------------------------------------
% Run the filter once more to get the filtered (xitt) and one-step-ahead
% predicted (xit1t) regime probabilities, then run Kim's smoother to get
% smoothed probabilities xitT.
% -------------------------------------------------------------------------
[lfv, xitt, xit1t] = Function_1(x0, 0);

xitT = zeros(T-1, 4);
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
names_var{k} = sprintf('alpha           '); k = k+1;
for j = 0:q_lag
    names_var{k} = sprintf('beta%d-Regime 0  ', j); k = k+1;
    names_var{k} = sprintf('beta%d-Regime 1  ', j); k = k+1;
end
for i = 1:p_lag
    names_var{k} = sprintf('gamma%d          ', i); k = k+1;
end
names_var{k} = sprintf('sigma-Regime 0  '); k = k+1;
names_var{k} = sprintf('sigma-Regime 1  '); k = k+1;
names_var{k} = sprintf('p11-Regime 0    '); k = k+1;
names_var{k} = sprintf('p22-Regime 0    '); k = k+1;
names_var{k} = sprintf('p11-Regime 1    '); k = k+1;
names_var{k} = sprintf('p22-Regime 1    ');

coefficient = x0;
Std_Error   = sgd;
t_statistic = coefficient ./ Std_Error;
p_value     = 2 * (1 - normcdf(abs(t_statistic), 0, 1));

disp('_________________________________________________________________________________________________________________')
disp('Estimated Coefficients (Two Independent Markov Processes)')
TABLA1 = table(coefficient, Std_Error, t_statistic, p_value, 'RowNames', names_var)

% -------------------------------------------------------------------------
% Resolve label-switching automatically:
% Identify which raw regime corresponds to "low/high sensitivity" and
% "low/high volatility" from the magnitudes of the estimated coefficients.
% -------------------------------------------------------------------------
beta0_R0 = thopt(2);   % beta_0 in Chain 1, Regime 0
beta0_R1 = thopt(3);   % beta_0 in Chain 1, Regime 1

if abs(beta0_R0) >= abs(beta0_R1)
    beta_high = 0;  beta_low = 1;
else
    beta_high = 1;  beta_low = 0;
end

sigma_R0 = thopt(n_reg+1);
sigma_R1 = thopt(n_reg+2);

if sigma_R0 >= sigma_R1
    sigma_high = 0;  sigma_low = 1;
else
    sigma_high = 1;  sigma_low = 0;
end

% xitT column order: (β-R0,σ-R0)=1, (β-R0,σ-R1)=2, (β-R1,σ-R0)=3, (β-R1,σ-R1)=4
idx = @(b, s) 2*b + s + 1;

% Desired display order 
regime_order = [idx(beta_low,  sigma_low ), ...   % Low-Sens & Low-Vol
                idx(beta_low,  sigma_high), ...   % Low-Sens & High-Vol
                idx(beta_high, sigma_low ), ...   % High-Sens & Low-Vol
                idx(beta_high, sigma_high)];      % High-Sens & High-Vol

% -------------------------------------------------------------------------
% Plot smoothed regime probabilities
% -------------------------------------------------------------------------
regime_titles = {'Low-Sensitivity & Low-Volatility Regime', ...
                 'Low-Sensitivity & High-Volatility Regime', ...
                 'High-Sensitivity & Low-Volatility Regime', ...
                 'High-Sensitivity & High-Volatility Regime'};

figure('Name', 'Smoothed Regime Probabilities', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.78 0.72]);

line_color = [0.25 0.45 0.75];  
gray_color = [0.40 0.40 0.40];   

for k = 1:4
    subplot(2, 2, k);
    plot(dates_eff, xitT(:, regime_order(k)), ...
         'Color', line_color, 'LineWidth', 0.6);

    title(regime_titles{k}, ...
          'FontAngle',  'italic', ...
          'FontWeight', 'normal', ...
          'FontName',   'Times New Roman', ...
          'FontSize',   12, ...
          'Color',      gray_color);

    if ismember(k, [1, 3])
        ylabel('Probability', 'FontName', 'Times New Roman', 'FontSize', 10, 'Color', gray_color);
    end
    
    if ismember(k, [3, 4])
        xlabel('Date', 'FontName', 'Times New Roman', 'FontSize', 10, 'Color', gray_color);
    end

    ylim([0 1]);
    yticks(0:0.2:1);
    xlim([min(dates_eff) max(dates_eff)]);

    set(gca, ...
        'FontName',   'Times New Roman', ...
        'FontSize',   9, ...
        'Box',        'off', ...
        'TickDir',    'out', ...
        'TickLength', [0.008 0.008], ...
        'XColor',     gray_color, ...
        'YColor',     gray_color, ...
        'GridColor',  gray_color, ...
        'LineWidth',  0.5);

    text(0.97, 0.88, sprintf('p = %d, q = %d', p_lag, q_lag), ...
         'Units',               'normalized', ...
         'HorizontalAlignment', 'right', ...
         'FontAngle',           'italic', ...
         'FontName',            'Times New Roman', ...
         'FontSize',            9, ...
         'Color',               gray_color);
end