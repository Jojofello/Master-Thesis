% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Check whether an estimated VAR(p) is covariance-stationary by inspecting
%   the eigenvalues of its companion-form matrix. A VAR is stable if and
%   only if all eigenvalues of the companion matrix lie strictly inside the
%   unit circle (i.e. modulus < 1). This routine is typically called inside
%   Bayesian Gibbs samplers to discard posterior draws that would imply a
%   non-stationary VAR.
% =========================================================================
 
function [S, ee, FF] = Function_3(beta, n, l)
% -------------------------------------------------------------------------
% FUNCTION: stability
% -------------------------------------------------------------------------
% Given the stacked coefficient matrix of a VAR(p) with n endogenous
% variables and l lags, build the companion-form matrix and return the
% modulus of its largest eigenvalue together with a binary flag that is
% TRUE (==1) when the VAR is UNSTABLE (largest eigenvalue >= 1) and FALSE
% (==0) when it is stable.
%
% INPUTS
%   beta : (n*l + nx) x n  double matrix
%          Stacked coefficient matrix from a VAR(p) estimation. The first
%          nx rows correspond to deterministic regressors (constant,
%          trends, dummies), the remaining n*l rows to the lagged
%          endogenous regressors, ordered lag-by-lag and within each lag
%          variable-by-variable.
%   n    : double scalar (positive integer)
%          Number of endogenous variables.
%   l    : double scalar (positive integer)
%          Number of lags p of the VAR.
%
% OUTPUTS
%   S  : logical (0 or 1)
%        Stability flag. 1 = unstable, 0 = stable.
%   ee : double scalar
%        Modulus of the largest eigenvalue of the companion matrix.
%   FF : (n*l) x (n*l) double matrix
%        Companion-form (first-order representation) of the VAR.
% -------------------------------------------------------------------------
 
% Number of deterministic regressors (constant, trends, ...) implied by
% the row dimension of beta. nx is whatever is left over after the
% n*l autoregressive rows.
nx = size(beta,1) - n*l;
% coef = reshape(beta,n*l+1,n);   % (legacy reshape kept for reference)
% coef
% coef
 
% Pre-allocate the companion matrix FF.

FF = zeros(n*l, n*l);
 
% Fill the lower identity blocks that "shift" lagged states forward.
FF(n+1:n*l, 1:n*(l-1)) = eye(n*(l-1), n*(l-1));
 
% Reshape the stacked coefficient matrix so each column corresponds to
% one of the n VAR equations. Then drop the first nx (deterministic) rows
% and transpose.
temp = reshape(beta, n*l + nx, n);
temp = temp(nx+1:nx+n*l, 1:n)';
 
% Insert the autoregressive coefficient block into the top of FF.
FF(1:n, 1:n*l) = temp;
 
% Largest absolute eigenvalue of the companion matrix.
ee = max(abs(eig(FF)));
 
% Stability flag: 1 if unstable (>1), 0 if stable.
S = ee > 1;
 
end
 


