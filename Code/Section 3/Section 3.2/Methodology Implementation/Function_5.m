% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE
%   Construct the companion form matrix of a Vector Autoregression (VAR). 
%   This matrix rearranges a VAR(p) process into a VAR(1) stack, which 
%   facilitates the calculation of eigenvalues for stability analysis, 
%   impulse responses, and multi-step forecasts.
%
% THEORY
%   For a VAR(p) process: 
%       y_t = v + A1*y_{t-1} + A2*y_{t-2} + ... + Ap*y_{t-p} + u_t
%   The companion matrix A is the transition matrix of the stacked system:
%       Y_t = V + A * Y_{t-1} + U_t
%   where Y_t = [y_t', y_{t-1}', ..., y_{t-p+1}']'. The matrix A has the 
%   coefficients in the first n rows and an identity matrix below to 
%   propagate the lags.

function A=Function_5(A,ndet,n,p)
% -------------------------------------------------------------------------
% FUNCTION: Function_5
% -------------------------------------------------------------------------
% Creates the companion matrix of the VAR coefficients.
%
% INPUTS
%   A     : n x (ndet + n*p) double matrix
%           The estimated VAR coefficient matrix (including deterministic
%           terms like constant or trend).
%   ndet  : double scalar (non-negative integer)
%           Number of deterministic terms (columns) at the beginning of A.
%   n     : double scalar (positive integer)
%           Number of endogenous variables in the system.
%   p     : double scalar (positive integer)
%           Number of lags included in the VAR.
%
% OUTPUTS
%   A     : (n*p) x (n*p) double matrix
%           The companion matrix in VAR(1) form.
% =========================================================================
% -------------------------------------------------------------------------
% Remove deterministic terms
% -------------------------------------------------------------------------
% We isolate the autoregressive coefficients [A1, A2, ..., Ap] by 
% discarding the first 'ndet' columns (e.g., constant, trend).
A=A(:,ndet+1:end); 

% -------------------------------------------------------------------------
% Build the companion structure
% -------------------------------------------------------------------------
A=[A; eye(n*(p-1)) zeros(n*(p-1),n)];