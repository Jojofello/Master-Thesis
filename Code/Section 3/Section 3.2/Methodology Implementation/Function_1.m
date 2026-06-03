% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Compute the infinite-order MA (Wold) coefficient matrices of a stable
%   VAR(p) up to a given horizon. These matrices are the building blocks
%   for impulse response functions, forecast-error variance decompositions
%   and dynamic multipliers.
%
%   For a VAR(p)  y_t = mu + A1 y_{t-1} + ... + Ap y_{t-p} + e_t ,
%   the Wold (MA-infinity) representation is
%       y_t = mu* + sum_{h=0}^{inf} C_h * e_{t-h},  with C_0 = I_n.
%   The C_h matrices are obtained from the companion form A* via
%       C_h = J * (A*)^h * J',     J = [I_n, 0, ..., 0].
% =========================================================================
 
function wold = Function_1(A, ndet, p, nsteps)
% -------------------------------------------------------------------------
% FUNCTION: Function_1
% -------------------------------------------------------------------------
% Build the Wold coefficient matrices C_0, C_1, ..., C_nsteps for a VAR(p).
%
% INPUTS
%   A      : n x (n*p + ndet) double matrix
%            Reduced-form coefficient matrix where each row collects the
%            coefficients of one VAR equation. The first ndet columns
%            correspond to deterministic regressors and are stripped out
%            inside the helper varcompanion (see below); the remaining
%            n*p columns hold the autoregressive coefficients ordered
%            A1, A2, ..., Ap.
%   ndet   : double scalar (non-negative integer)
%            Number of deterministic regressors (constant, trends, ...).
%   p      : double scalar (positive integer)
%            Number of lags of the VAR.
%   nsteps : double scalar (non-negative integer)
%            Maximum horizon h. The function returns C_0, ..., C_nsteps,
%            i.e. (nsteps + 1) matrices.
%
% OUTPUT
%   wold : n x n x (nsteps + 1) double 3-D array
%          wold(:,:,h+1) = C_h, the MA(infinity) coefficient at horizon h.
%
% NOTE
%   Requires the helper function varcompanion which removes the
%   deterministic part of A and returns the (n*p) x (n*p) companion
%   matrix.
% -------------------------------------------------------------------------
 
% Number of endogenous variables.
n = size(A,1);
 
% Convert reduced-form coefficient matrix to companion form A* (after
% stripping the ndet deterministic columns).
A = Function_5(A, ndet, n, p);   % A is now (n*p) x (n*p), no deterministic part
 
% Selection matrix J = [I_n, 0, ..., 0] of size n x (n*p).
% It picks the first n rows/columns of any companion-matrix power.
J = [eye(n) zeros(n, (p-1)*n)];
 
% Allocate the 3-D output array by concatenating along the third
% dimension.  At horizon h the Wold coefficient is C_h = J * (A*)^h * J'.
wold = [];
for h = 0:nsteps
    wold = cat(3, wold, (J * (A^h) * J'));
    % cat concatenates along the requested dimension:  cat(dim, X, Y).
end
 
% multiplier = cumsum(wold, 3);   % cumulative (interim/dynamic) multipliers
end
 