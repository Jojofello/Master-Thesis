% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Bayesian VAR with a Minnesota prior used to identify a news shock
%   and the corresponding surprise shock following the Barsky-Sims
%   maximum-FEV identification. 
%
%   External dependencies:
%       - Data.txt   (input data, ASCII)
%       - Function_1.m         (Wold/MA-infinity coefficients)
%       - Function_2.m         (Minnesota prior)
%       - Function_3.m         (companion-matrix stability check)
%       - Function_4.m         (Sequence of Values)
%       - Function_5.m         (Companion Matrix)
% =========================================================================

clear all;
clc;

% -------------------------------------------------------------------------
% Load data
% -------------------------------------------------------------------------
% Data.txt is a plain-ASCII matrix with three columns containing the
% supply, demand and risk-shock series respectively.
load Data.txt -ascii
s_supply = Data(:, 1);
s_demand = Data(:, 2);
s_risk   = Data(:, 3);

% Stack the data so that the first variable is the one whose forecast-
% error variance the news shock will maximize.
Y = [s_risk  s_supply  s_demand];

% -------------------------------------------------------------------------
% Model and sampler hyper-parameters
% -------------------------------------------------------------------------
prior  = 0.5;         % Minnesota tightness (smaller = tighter)
n      = size(Y, 2);  % number of endogenous variables (3)
p      = 22;          % number of VAR lags
nlags  = p;
nvars  = size(Y, 2);

% -------------------------------------------------------------------------
% Construct the regressor matrix X of the VAR
% X = [ const,  Y_{t-1},  Y_{t-2},  ...,  Y_{t-p} ]
% -------------------------------------------------------------------------
X = ones(size(Y,1) - p, 1);
for i = 1:p
    X = [X  Y(p+1-i:end-i, :)];
end
Y = Y(p+1:end, :);  y = Y;     % drop the first p observations to match X
T = length(X);

% -------------------------------------------------------------------------
% Frequentist OLS estimate (used as posterior centre and starting value)
% -------------------------------------------------------------------------
beta = inv(X'*X) * X' * y;       % OLS coefficient matrix, (n*p+1) x n
e    = y - X * beta;             % OLS residuals
vmat = e' * e / T;               % residual covariance matrix

% -------------------------------------------------------------------------
% Bayesian setup
% kfset = (X'X)^(-1)        Cholesky factors used for posterior draws.
% -------------------------------------------------------------------------
kfset   = inv(X'*X);
sxx     = chol(kfset)';          % lower Cholesky of (X'X)^(-1)
svtr    = chol(vmat);            % upper Cholesky of OLS residual cov
betaols = beta;
ncoefs  = size(sxx, 1);

% -------------------------------------------------------------------------
% Sampler size
% -------------------------------------------------------------------------
Reps = 12000;        % total Gibbs iterations
burn = 9000;         % number of draws retained (last `burn` of `Reps`)

surprise_shock_post = [];
news_shock_post     = [];

% -------------------------------------------------------------------------
% Minnesota prior
% lev = ones(nvars,1) imposes a unit-root prior on every equation.
% -------------------------------------------------------------------------
lev = ones(nvars, 1);
[hm, bm] = Function_2(Y, X, nlags, 1, 1, lev, prior);
hm = diag(hm);

% =========================================================================
% MAIN GIBBS LOOP
% =========================================================================
for j = 1:Reps
    int8(100*j/Reps)        % crude progress indicator (printed each step)

    % ---------------------------------------------------------------------
    % Draw Sigma | data, beta from an inverse-Wishart distribution
    % (we use the OLS residual covariance scaled by an iWishart draw).
    % ---------------------------------------------------------------------
    temp01 = iwishrnd(size(y,1)*eye(nvars), size(y,1));   % iWishart draw
    sigmad = svtr' * temp01 * svtr;
    % swish = chol(sigmad)';  % (alternative parameterization, unused)

    % ---------------------------------------------------------------------
    % Draw beta | data, Sigma from its conditional normal posterior.
    % Repeat until the resulting VAR is stable (eigenvalues < 1).
    % ---------------------------------------------------------------------
    check = -1;
    while check < 0
        % ranc  = randn(ncoefs, nvars);
        % betau = sxx * ranc * swish';
        Vstar = inv(hm + kron(inv(sigmad), X'*X));
        uu    = randn(1, ncoefs*nvars) * chol(Vstar);
        % betau = reshape(uu, ncoefs, nvars);
        % vbetadraw is the stacked vector of posterior coefficient draws
        % ordered as: const, lag1 var1, lag1 var2, lag1 var3, lag2 var1, ...
        vbetadraw = Vstar * (bm + kron(inv(sigmad), X'*X) * betaols(:)) + uu';
        betadraw  = reshape(vbetadraw, ncoefs, nvars);
        [CH, ee, FF] = Function_3(betadraw, nvars, nlags);
        if CH == 0
            check = 10;     % stable -> exit the while-loop
        end
    end

    % VAR residuals implied by the current coefficient draw.
    udraw = y - X * betadraw;

    % ---------------------------------------------------------------------
    % Identification of news + surprise shocks (only for retained draws)
    % ---------------------------------------------------------------------
    if j > Reps - burn
        % Cholesky factor of Sigma (lower triangular).
        A0 = chol(sigmad)';

        % Wold (MA-infinity) coefficients up to horizon `steps`.
        steps = 50;
        wold  = Function_1(betadraw', 1, p, steps + 1);

        % Build the cumulative variance-share matrix S that the orthogonal
        % rotation Q (in the (n-1)-dim subspace spanned by columns 2..n)
        % must maximise to identify the news shock.
        SUM = zeros(n, n);  sum = zeros(n, n);  sumsig = 0;
        for k = 1:steps + 1
            Ck     = reshape(wold(:,:,k), n, n);
            sumsig = sumsig + Ck(1,:) * sigmad * Ck(1,:)';
            sum    = sum    + A0' * Ck(1,:)' * Ck(1,:) * A0;
            SUM    = SUM    + sum;
            % SUM = SUM + (1/sumsig) .* sum;     % alternative weighting
        end
        S = SUM;

        % Restrict S to the (n-1) x (n-1) block orthogonal to the surprise
        % shock and find its eigenvectors. The news shock is the
        % combination of structural innovations associated with the
        % LARGEST eigenvalue (Barsky-Sims).
        Stilde            = S(2:n, 2:n);
        [kvec, keig]      = eig(Stilde);
        keig2             = diag(keig);
        [keig3, kord]     = sort(abs(keig2), 'descend');
        keig4             = keig2(kord);
        Mlambda           = diag(keig4);
        Qtilde            = kvec(:, kord);

        % Build the rotation matrix Q. Column 1 is fixed to identify the
        % surprise shock as the standard Cholesky shock to the first
        % variable; columns 2..n carry the eigenvector rotation.
        Q = zeros(n, n);
        Q(1, 1)         = 1;
        Q(2:n, 2:n)     = Qtilde;

        % Recover the structural shocks: u_struct = (A0*Q)^(-1) * u_reduced
        sshocks         = inv(A0*Q) * udraw';
        news_shock      = sshocks(2, :)';
        surprise_shock  = sshocks(1, :)';

        % -----------------------------------------------------------------
        % Sign normalization of the news shock (FRI = Forni-Reichlin-Imbs):
        % a positive news shock should RAISE the first variable
        % (here: risk) at horizon 1. If it does not, flip the sign.
        % -----------------------------------------------------------------
        u           = zeros(p+5, n);
        ns          = 2;
        u(p+1, ns)  = 1;
        yhat        = zeros(p+5, n);
        for i = p+1:p+5
            myhat = 0;
            for jj = 1:p
                myhat = [myhat  yhat(i-jj, :)];
            end
            % yhat(i,:) = myhat * reshape(beta, n*p+1, n) + u(i,:)*Q'*A0';
            yhat(i, :) = myhat * betadraw + u(i, :) * Q' * A0';
        end
        if ns == 2
            if yhat(p+2, 1) < 0
                Q(:, 2)        = -Q(:, 2);
                sshocks        = inv(A0*Q) * udraw';
                news_shock     = sshocks(2, :)';
                yhat(:, 1)     = -yhat(:, 1);
                yhat(:, 2)     = -yhat(:, 2);
                yhat(:, 3)     = -yhat(:, 3);
            end
        end

        news_shock_post     = [news_shock_post     news_shock];
        surprise_shock_post = [surprise_shock_post surprise_shock];
    end
end

% =========================================================================
% Posterior summaries and output
% =========================================================================
figure;
plot(mean(news_shock_post')');
title('News Shocks');

figure;
plot(mean(surprise_shock_post')');
title('Surprise Shocks');

m_news_shock     = mean(news_shock_post')';
m_surprise_shock = mean(surprise_shock_post')';
m_shocks         = [m_news_shock  m_surprise_shock];

% Save posterior-mean shock series as plain ASCII for downstream use.
save shocks_r_new.prn m_shocks -ascii 
