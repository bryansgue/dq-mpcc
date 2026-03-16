%% Load and Compare Nominal vs Plant Trajectories
% Script to analyze the reconstruction of nominal trajectory from identification data
% 
% This compares:
%   X - Plant states (from simulator)
%   X_nom - Reconstructed nominal states (from model integration)
%   X_d - Reference trajectory

clear all; close all; clc;

% =========================================================================
% 1. Load Data
% =========================================================================
fprintf('Loading identification data...\n');
data = load('Dual_cost_identification_with_nominal.mat');

X = data.X;           % Plant states: (N_exp, 14, N_steps)
X_nom = data.X_nom;   % Nominal reconstructed: (N_exp, 14, N_steps)
X_d = data.X_d;       % Reference: (N_exp, 14, N_steps)
u = data.u;           % Control inputs: (N_exp, 4, N_steps)
t = data.t;           % Time: (N_exp, N_steps)

error = X - X_nom;
error_norm = data.error_norm;

[N_exp, N_states, N_steps] = size(X);
fprintf('✓ Loaded %d experiments, %d states, %d time steps\n', N_exp, N_states, N_steps);
fprintf('  X shape: (%d, %d, %d)\n', size(X, 1), size(X, 2), size(X, 3));
fprintf('  X_nom shape: (%d, %d, %d)\n', size(X_nom, 1), size(X_nom, 2), size(X_nom, 3));

% Use first experiment for analysis
exp = 1;
X_exp = squeeze(X(exp, :, :));
X_nom_exp = squeeze(X_nom(exp, :, :));
X_d_exp = squeeze(X_d(exp, :, :));
t_exp = squeeze(t(exp, :));
error_exp = X_exp - X_nom_exp;

% State indices
% 1-4: Real quaternion [qw, qx, qy, qz]
% 5-8: Dual quaternion [dw, dx, dy, dz]
% 9-11: Angular velocity [wx, wy, wz]
% 12-14: Linear velocity [vx, vy, vz]

% =========================================================================
% 2. Compare Linear Velocities (Key Parameter for Identification)
% =========================================================================
figure('Name', 'Linear Velocities Comparison', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 600]);

% vx
subplot(2,3,1);
plot(t_exp, X_exp(12,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant (vx)');
hold on;
plot(t_exp, X_nom_exp(12,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model (vx)');
plot(t_exp, X_d_exp(12,:), 'g:', 'LineWidth', 1.5, 'DisplayName', 'Reference (vx)');
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Linear Velocity X');
legend('Location', 'best');
grid on;

% vy
subplot(2,3,2);
plot(t_exp, X_exp(13,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant (vy)');
hold on;
plot(t_exp, X_nom_exp(13,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model (vy)');
plot(t_exp, X_d_exp(13,:), 'g:', 'LineWidth', 1.5, 'DisplayName', 'Reference (vy)');
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Linear Velocity Y');
legend('Location', 'best');
grid on;

% vz
subplot(2,3,3);
plot(t_exp, X_exp(14,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant (vz)');
hold on;
plot(t_exp, X_nom_exp(14,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model (vz)');
plot(t_exp, X_d_exp(14,:), 'g:', 'LineWidth', 1.5, 'DisplayName', 'Reference (vz)');
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Linear Velocity Z');
legend('Location', 'best');
grid on;

% Error in linear velocities
subplot(2,3,4);
plot(t_exp, error_exp(12,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (m/s)');
title('Error in vx: X_{plant} - X_{nom}');
grid on;

subplot(2,3,5);
plot(t_exp, error_exp(13,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (m/s)');
title('Error in vy: X_{plant} - X_{nom}');
grid on;

subplot(2,3,6);
plot(t_exp, error_exp(14,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (m/s)');
title('Error in vz: X_{plant} - X_{nom}');
grid on;

% =========================================================================
% 3. Compare Angular Velocities
% =========================================================================
figure('Name', 'Angular Velocities Comparison', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 600]);

% wx
subplot(2,3,1);
plot(t_exp, X_exp(9,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant');
hold on;
plot(t_exp, X_nom_exp(9,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model');
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity X (wx)');
legend('Location', 'best');
grid on;

% wy
subplot(2,3,2);
plot(t_exp, X_exp(10,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant');
hold on;
plot(t_exp, X_nom_exp(10,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model');
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity Y (wy)');
legend('Location', 'best');
grid on;

% wz
subplot(2,3,3);
plot(t_exp, X_exp(11,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Plant');
hold on;
plot(t_exp, X_nom_exp(11,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Model');
xlabel('Time (s)'); ylabel('Angular Velocity (rad/s)');
title('Angular Velocity Z (wz)');
legend('Location', 'best');
grid on;

% Errors
subplot(2,3,4);
plot(t_exp, error_exp(9,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (rad/s)');
title('Error in wx');
grid on;

subplot(2,3,5);
plot(t_exp, error_exp(10,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (rad/s)');
title('Error in wy');
grid on;

subplot(2,3,6);
plot(t_exp, error_exp(11,:), 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Error (rad/s)');
title('Error in wz');
grid on;

% =========================================================================
% 4. Overall Error Analysis
% =========================================================================
figure('Name', 'Error Analysis', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1200, 500]);

% Error norm
subplot(1,2,1);
error_norm_exp = squeeze(error_norm(exp, :));
plot(t_exp, error_norm_exp, 'LineWidth', 2, 'Color', [0.8, 0, 0]);
xlabel('Time (s)'); ylabel('Error Norm ||X - X_{nom}||');
title('Overall Reconstruction Error');
grid on;
fprintf('  Max error norm: %.6e\n', max(error_norm_exp));
fprintf('  Mean error norm: %.6e\n', mean(error_norm_exp));

% Error in each component
subplot(1,2,2);
error_components = abs(error_exp);
plot(t_exp, error_components', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Absolute Error');
title('Error per State Component');
legend(arrayfun(@(i) sprintf('State %d', i), 1:14, 'UniformOutput', false), ...
       'Location', 'best', 'FontSize', 8);
grid on;

% =========================================================================
% 5. Print Statistics
% =========================================================================
fprintf('\n=== RECONSTRUCTION ERROR STATISTICS ===\n');
fprintf('Experiment: %d\n', exp);
fprintf('Total time steps: %d\n', N_steps);
fprintf('Time duration: %.2f s\n', t_exp(end));

fprintf('\nLinear Velocities Error (vx, vy, vz):\n');
for i = 12:14
    err_i = error_exp(i, :);
    fprintf('  v%c: mean=%.6e, max=%.6e, std=%.6e\n', ...
            char(119+i-11), mean(abs(err_i)), max(abs(err_i)), std(err_i));
end

fprintf('\nAngular Velocities Error (wx, wy, wz):\n');
for i = 9:11
    err_i = error_exp(i, :);
    fprintf('  w%c: mean=%.6e, max=%.6e, std=%.6e\n', ...
            char(120+i-9), mean(abs(err_i)), max(abs(err_i)), std(err_i));
end

fprintf('\nDual Quaternion Error (dw, dx, dy, dz):\n');
for i = 5:8
    err_i = error_exp(i, :);
    fprintf('  d%s: mean=%.6e, max=%.6e, std=%.6e\n', ...
            char(119+i-4), mean(abs(err_i)), max(abs(err_i)), std(err_i));
end

fprintf('\nQuaternion Error (qw, qx, qy, qz):\n');
for i = 1:4
    err_i = error_exp(i, :);
    fprintf('  q%s: mean=%.6e, max=%.6e, std=%.6e\n', ...
            char(119+i-0), mean(abs(err_i)), max(abs(err_i)), std(err_i));
end

fprintf('\n=== INTERPRETATION ===\n');
fprintf('If error is high in linear velocities (vx, vy, vz):\n');
fprintf('  → The kinematic model may not capture velocity dynamics\n');
fprintf('  → Need to add acceleration/dynamics terms\n');
fprintf('  → Check if control inputs (F, tau) should affect velocities\n\n');

fprintf('If error in orientations (quaternions) is high:\n');
fprintf('  → The dual quaternion kinematics may need adjustment\n');
fprintf('  → Check the RK4 integration accuracy\n');
fprintf('  → Verify quaternion normalization\n\n');

fprintf('If error grows over time:\n');
fprintf('  → Indicates model drift or missing dynamics\n');
fprintf('  → Consider adding feedback correction or friction terms\n');
