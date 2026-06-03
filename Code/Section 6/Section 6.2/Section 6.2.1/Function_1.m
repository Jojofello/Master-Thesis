% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Negative log-likelihood and Hamilton filter for a two-regime Markov-
%   switching ADL model. The function is called by Single_Markov_chain.m 
%   both for optimization (it returns -log L) and afterwards for inference
%   (it also returns the filtered and one-step-ahead predicted regime
%   probabilities).
%
%   Model:
%       X_t = alpha
%             + sum_{j=0}^{q_lag} beta_j^{(s_t)} * xi_{t-j}
%             + sum_{i=1}^{p_lag} gamma_i        * X_{t-i}
%             + u_t,             u_t ~ N( 0, sigma_{s_t}^2 )
%
%   Two regimes (N = 2). The constant alpha and gamma_i are common across
%   regimes; the slope coefficients beta_j and the residual standard
%   deviation are regime-specific.
% =========================================================================

function [lfv, xitt, xit1t] = Function_1(thtr, ind)
% -------------------------------------------------------------------------
% FUNCTION: Function_1
% -------------------------------------------------------------------------
% INPUTS
%   thtr : 1 x n_par double row vector
%          Stacked parameter vector ordered as
%             [ alpha,
%               beta_0^{(0)}, beta_0^{(1)}, ...,
%               beta_qlag^{(0)}, beta_qlag^{(1)},
%               gamma_1, ..., gamma_plag,
%               sigma0_tr, sigma1_tr, p11_tr, p22_tr ].
%   ind  : double / logical scalar
%          1 -> the parameters in thtr are still in their TRANSFORMED
%               parameterisation (log for sigmas, logit for transition
%               probabilities). The function back-transforms internally.
%               Used during fminsearch optimisation.
%          0 -> thtr is already in NATURAL parameterisation. Used to
%               evaluate the likelihood at the maximum and to compute
%               the numerical Hessian for standard errors.
%
% OUTPUTS
%   lfv   : double scalar
%           Negative log-likelihood evaluated at thtr. (Negative because
%           fminsearch minimises.)
%   xitt  : T x 2 double matrix
%           Filtered regime probabilities Pr(s_t = i | F_t), i = 0, 1.
%   xit1t : (T+1) x 2 double matrix
%           One-step-ahead predicted regime probabilities
%           Pr(s_{t+1} = i | F_t), i = 0, 1, with the steady-state
%           distribution prepended in row 1.
%
% GLOBALS USED
%   X     : T x 1 double vector of the dependent variable.
%   Z_reg : T x k double matrix with [const, xi-lags, X-lags].
%   p_lag : double scalar, number of AR lags in the model.
%   q_lag : double scalar, number of lags of xi in the model.
% -------------------------------------------------------------------------

global X Z_reg p_lag q_lag

T    = length(X);
thtr = real(thtr);     % drop any imaginary noise from the optimiser
N    = 2;              % number of regimes

% -------------------------------------------------------------------------
% Unpack the parameters from thtr.
% -------------------------------------------------------------------------
alpha0 = thtr(1);

% Regime-specific coefficients on xi and its lags.
idx        = 2;
beta_xi_R0 = zeros(q_lag+1, 1);
beta_xi_R1 = zeros(q_lag+1, 1);
for j = 0:q_lag
    beta_xi_R0(j+1) = thtr(idx);
    beta_xi_R1(j+1) = thtr(idx+1);
    idx = idx + 2;
end

% Common AR coefficients.
gamma_vec = thtr(idx:idx+p_lag-1)';
idx       = idx + p_lag;

% -------------------------------------------------------------------------
% Sigma and transition probabilities (back-transform if needed).
% -------------------------------------------------------------------------
if ind == 1
    sigma0 = exp(thtr(idx));
    sigma1 = exp(thtr(idx+1));
    p11_0  = exp(thtr(idx+2)) / (1 + exp(thtr(idx+2)));
    p22_0  = exp(thtr(idx+3)) / (1 + exp(thtr(idx+3)));
else
    sigma0 = thtr(idx);
    sigma1 = thtr(idx+1);
    p11_0  = thtr(idx+2);
    p22_0  = thtr(idx+3);
end

% -------------------------------------------------------------------------
% Build the regime-specific full coefficient vectors used in Z_reg*beta.
% beta0 / beta1 contain alpha, the regime-specific xi-coefficients and
% the common gamma_i coefficients.
% -------------------------------------------------------------------------
beta0 = [alpha0; beta_xi_R0; gamma_vec(:)];
beta1 = [alpha0; beta_xi_R1; gamma_vec(:)];

% -------------------------------------------------------------------------
% Conditional Gaussian densities eta(t,i) = phi(X_t | s_t = i).
% -------------------------------------------------------------------------
eta = zeros(T, N);
for t = 1:T
    eta(t,1) = (1/(sigma0*(2*pi)^0.5)) * exp(-((X(t) - Z_reg(t,:)*beta0).^2) / (2*sigma0^2));
    eta(t,2) = (1/(sigma1*(2*pi)^0.5)) * exp(-((X(t) - Z_reg(t,:)*beta1).^2) / (2*sigma1^2));
end

% -------------------------------------------------------------------------
% Transition matrix and steady-state (ergodic) distribution ppi.
% ppi solves (I - P)' ppi = 0  with  sum(ppi) = 1, computed by adding
% the constraint as the last equation in A*ppi = e_{N+1} and using OLS.
% -------------------------------------------------------------------------
P = [p11_0    1-p22_0;
     1-p11_0  p22_0];

A   = [eye(N) - P;  ones(1,N)];
eN1 = [zeros(N,1);  1];
ppi = inv(A'*A) * A' * eN1;     % steady-state probabilities (column)

% -------------------------------------------------------------------------
% Hamilton filter: iterate filtering and prediction equations.
% xitt(t,:)  = Pr(s_t = i | F_t)        (filtered)
% xit1t(t,:) = Pr(s_{t+1} = i | F_t)    (one-step-ahead predicted)
% -------------------------------------------------------------------------
xi10_init = ppi';
xi10      = ppi';
xitt      = zeros(T, N);
xit1t     = zeros(T, N);

for t = 1:T
    xitt(t,:)  = (xi10 .* eta(t,:)) / ((xi10 .* eta(t,:)) * ones(N,1));
    xit1t(t,:) = xitt(t,:) * P';
    xi10       = xit1t(t,:);
end

% Prepend the initial steady-state distribution to xit1t.
xit1t = [xi10_init; xit1t];

% -------------------------------------------------------------------------
% Log-likelihood = sum_t log f(X_t | F_{t-1}) where
% f(X_t | F_{t-1}) = sum_i Pr(s_t = i | F_{t-1}) * eta(t,i).
% Returned as negative for minimisation by fminsearch.
% -------------------------------------------------------------------------
f   = (xit1t(1:T,:) .* eta) * ones(N,1);
lf  = log(f);
lfv = ones(1,T) * lf;

lfv = -lfv;     % negative log-likelihood

end