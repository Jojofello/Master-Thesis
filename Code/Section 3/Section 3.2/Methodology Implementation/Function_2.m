% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Build the prior precision matrix and the prior-mean term of a
%   Minnesota prior for a (reduced-form) VAR. The output is consumed by
%   a Bayesian sampler that combines the prior with the OLS likelihood
%   to obtain the conditional posterior mean of the stacked VAR
%   coefficients vec(B).
%
% THEORY
%   For a single T x 1 regression  y = X*b + u  with prior  b ~ N(bprior,M)
%   the posterior mean is
%       b_post = inv[ inv(M) + X'X ] * [ X'y + inv(M)*bprior ].
%
%   For a multivariate T x n regression  Y = X*B + U  where X is T x (np)
%   and B is (np) x n, with stacked prior  vec(B) ~ N(bprior, H) and H
%   diagonal, the posterior mean is
%       vec(B_post) = inv[ inv(H) + kron(I, X'X) ]
%                     * [ vec(X'Y) + inv(H)*bprior ].

function [hm, bm] = Function_2(y, x, nlag, quarterly, const, lev, prior)

% -------------------------------------------------------------------------
% FUNCTION: Function_2
% -------------------------------------------------------------------------
% Build the prior precision matrix and the prior-mean term of a
% Minnesota prior for a (reduced-form) VAR.
%
% INPUTS
%   y         : T x nvar double matrix
%               LHS data of the VAR regressions (rows = observations).
%   x         : T x k    double matrix
%               RHS data: the constant column followed by the np lagged
%               regressors. Used here only to compute the residual
%               variances of univariate AR scaling regressions.
%   nlag      : double scalar (positive integer)
%               Number of lags p of the VAR.
%   quarterly : double scalar / logical
%               1 if the data is quarterly, 0 if monthly. Controls the
%               shape of the lag-decay vector.
%   const     : double scalar / logical
%               1 if the regressions include a constant term, 0 otherwise.
%   lev       : nvar x 1 double vector with elements
%                 1 -> equation estimated in levels (unit-root prior on
%                      own first lag),
%                 0 -> otherwise (zero-mean prior).
%   prior     : double scalar (positive)
%               Overall tightness of the prior on lags:
%                 prior = 1   -> standard prior,
%                 prior = inf -> uninformative (no prior),
%                 prior < 1   -> tighter than standard.
%
% OUTPUTS
%   hm : (k*nvar) x 1 double vector
%        Diagonal elements of inv(H), i.e. of the prior precision matrix
%        of vec(B).
%   bm : (k*nvar) x 1 double vector
%        Equals inv(H)*bprior, i.e. the prior-mean contribution to the
%        posterior. For lev(i) == 1 the entry corresponding to the own
%        first lag is centred at 1 (unit-root prior); other entries are 0.
% =========================================================================
 
[t, nvar] = size(y);          % number of variables and observations
k = const + nvar*nlag;        % number of regressors per equation
 
% -------------------------------------------------------------------------
% Tightness hyper-parameters of the Minnesota prior
% -------------------------------------------------------------------------
lambda0 = prior;              % overall tightness
lambda1 = .2;                 % gamma in Hamilton p. 361 (own-lag tightness)
lambda2 = .5;                 % w (cross-equation tightness)
lambda3 = 1;                  % lag-decay exponent
lambda4 = 1e5;                % prior on intercept (very loose)
 
% -------------------------------------------------------------------------
% Lag-decay vector ld
% Quarterly: standard 1/q^lambda3 decay.
% Monthly  : Tao Zha's decay matched to the quarterly schedule so that
%            the lag-3*j coefficient receives the same a-priori variance
%            as the quarterly lag-j coefficient.
% -------------------------------------------------------------------------
if quarterly == 1
    ld = Function_4(1, 1, nlag).^(-lambda3);   % (1, 2, ..., nlag).^(-lambda3)
else
    j = ceil(nlag/3)^(-lambda3);         % last quarter (rounded up)
    b = 0;
    if nlag > 1
        b = ( log(1) - log(j) ) / (1 - nlag);
    end
    a  = exp(-b);
    ld = a * exp(b * Function_4(1, 1, nlag));  % monthly decay matching quarterly
end
 
% Square-inverse with the overall scaling: this is the diagonal of inv(H)
% for the OWN-lag coefficients.  The cross-equation factor lambda2 will
% be peeled off again for the own lags inside the loop below.
ld = (lambda0*lambda1*lambda2*ld).^(-2);
 
% -------------------------------------------------------------------------
% Scale factors s_i: residual variances from univariate AR(p) regressions
% used to rescale the prior across equations.
% -------------------------------------------------------------------------
s = zeros(nvar,1);
i = 1;
 
for i = 1:nvar
   % Build a regressor matrix for variable i: constant + own lags.
   xi = x(:, k);
   for j = 1:nlag
      xi = [xi  x(:, i + (j-1)*nvar)];
      % x = [x  Y(nlag+1-j:rows(Y)-j, i)];   % (older syntax, kept for reference)
   end
   bsh  = inv(xi'*xi) * xi' * y(:, i);
   u    = y(:, i) - xi * bsh;
   s(i) = (u'*u) / t;
end
 
% -------------------------------------------------------------------------
% Build inv(H) for one equation, including the prior on the intercept.
% -------------------------------------------------------------------------
if lambda4 > 0
    if const == 1
        H = [kron(ld, s); ((lambda0*lambda4).^(-2))];
    else
        H = kron(ld, s);
    end
 
elseif lambda4 == 0
    if const
        H = [kron(ld, s); 0];
    else
        H = kron(ld, s);
    end
end
 
% -------------------------------------------------------------------------
% Stack inv(H) and inv(H)*bprior across the nvar equations.
% Within an equation the OWN lag uses a tighter weight (lambda2 squared
% factor removed) and, if lev(i) == 1, the own first lag gets a
% unit-root prior mean.
% -------------------------------------------------------------------------
hm = zeros(k*nvar, 1);
bm = zeros(k*nvar, 1);
 
for i = 1:nvar
    hadd = H;          % NOTE: H is normalized by var(dependent variable)
    for j = 0:(nlag-1)
       % Tighten the OWN lag (variable i, lag j+1) by undoing one factor of lambda2^(-2).
       hadd(i + (j*nvar)) = (lambda2^2) * H(i + (j*nvar));
    end
    hm(k*(i-1)+1:k*i, 1) = hadd;       % stack diagonal of inv(H) for equation i
 
    badd = zeros(k, 1);
 
    % XXXXXXXXXXXXXXXXXXXXXXXXXXXX
    % Unit-root prior on own first lag if the equation is in levels.
    % The (k*(i-1)+i) position picks the i-th regressor of equation i,
    % which is precisely the coefficient on the own first lag.
    if lev(i) == 1
       bm(k*(i-1)+i, 1) = hm(k*(i-1)+i) * 1;   % identical to Sims' formula
    end
end
 
end