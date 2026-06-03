% Original Code by Jesús Ruiz, modified by Johannes Felchner, 08.05.2026
% PURPOSE 
%   Produce a sequence of values.
% =========================================================================

function seq=Function_4(a,b,c);
% -------------------------------------------------------------------------
% FUNCTION: Function_4
% -------------------------------------------------------------------------
% Produce a sequence of values
% -----------------------------------------------------
% INPUTS
%   a = int
%       Initial value in sequence 
%   b = int
%       Increment
%   c = int
%       Number of values in the sequence  
% OUTPUTS
%   seq = int
%       A sequence, (a:b:(a+b*(c-1)))' in MATLAB notation

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jpl@jpl.econ.utoledo.edu
% =========================================================================
       
% Seqa Gauss eqivalent of seqa(a,b,c)
seq=(a:b:(a+b*(c-1)))';
return;
