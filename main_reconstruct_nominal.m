% MAIN_RECONSTRUCT_NOMINAL Entry point for the MATLAB reconstruction pipeline.
% Simply run this script (F5) to execute the full workflow:
%   1. Load Dual_cost_identification.mat
%   2. Integrate the nominal dual-quaternion model (sin/con drag)
%   3. Save Dual_cost_identification_with_nominal.mat and plots
%
% The heavy lifting lives in reconstruct_nominal_trajectory_matlab.m.

run('reconstruct_nominal_trajectory_matlab.m');
