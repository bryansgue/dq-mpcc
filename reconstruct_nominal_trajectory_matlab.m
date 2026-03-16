%% reconstruct_nominal_trajectory_matlab.m
% MATLAB replica of the Python pipeline.
% Loads Dual_cost_identification.mat, integrates the dual-quaternion
% dynamics twice (without drag / with identified drag), saves
% Dual_cost_identification_with_nominal.mat, and produces summary plots.
%
% Run this file directly (it's a script). Helper functions live below the
% main body so MATLAB still recognizes them.

clc;
fprintf('\n======================================================================\n');
fprintf('RECONSTRUCTING NOMINAL TRAJECTORY (MATLAB)\n');
fprintf('======================================================================\n');

% ----------------------------------------------------------------------
% [1] Load identification bundle
% ----------------------------------------------------------------------
scriptDir = fileparts(mfilename('fullpath'));
defaultMatFile = '/home/bryansgue/catkin_ws/src/dual_quaternion/scripts/MPC_LIE_FINAL/Dual_cost_identification_with_nominal.mat';
fallbackMatFile = fullfile(scriptDir, 'Dual_cost_identification.mat');

if exist(defaultMatFile, 'file')
    matFile = defaultMatFile;
else
    matFile = fallbackMatFile;
end

if ~exist(matFile, 'file')
    error('File not found: %s', matFile);
end

fprintf('[1] Loading identification data ...\n');
data = load(matFile);
if isfield(data, 'X') && isfield(data, 'u') && isfield(data, 't') && isfield(data, 'X_d')
    X = double(data.X);
    X_d = double(data.X_d);
    u = double(data.u);
    t = double(data.t);
else
    error('File %s does not contain the required variables (X, X_d, u, t).', matFile);
end

fprintf('  ▸ X shape: %s\n', mat2str(size(X)));
fprintf('  ▸ X_d shape: %s\n', mat2str(size(X_d)));
fprintf('  ▸ u shape: %s\n', mat2str(size(u)));
fprintf('  ▸ t shape: %s\n', mat2str(size(t)));

% ----------------------------------------------------------------------
% [2] Simulation configuration
% ----------------------------------------------------------------------
fprintf('\n[2] Preparing simulation config ...\n');
substeps = getenv('NOMINAL_RK4_SUBSTEPS');
substeps = str2double(substeps);
if isnan(substeps) || substeps <= 0
    substeps = 4;
end
system_params = struct('m', 1.0, 'J', diag([2.64e-3, 2.64e-3, 4.96e-3]), 'g', 9.8);
ts = compute_sample_time(t);
fprintf('  ▸ Sample time ts: %.6f s\n', ts);
fprintf('  ▸ RK4 sub-steps per sample: %d\n', substeps);

cfg = struct('params', system_params, 'ts', ts, 'substeps', substeps);

% ----------------------------------------------------------------------
% [3] Baseline simulation (no drag)
% ----------------------------------------------------------------------
fprintf('\n[3] Simulating nominal model (baseline: sin drag) ...\n');
X_nom_no_drag = simulate_nominal(X, u, cfg, [0 0 0]);
baseline_metrics = compute_residuals(X, X_nom_no_drag, cfg.ts, cfg.params);
drag_coeffs = estimate_drag_coeffs(baseline_metrics.force_residual, X_nom_no_drag);
fprintf('  ▸ Estimated drag [N·s/m]: [%0.6f  %0.6f  %0.6f]\n', drag_coeffs);

% ----------------------------------------------------------------------
% [4] Drag-compensated simulation
% ----------------------------------------------------------------------
fprintf('\n[4] Simulating nominal model (con drag identificado) ...\n');
X_nom = simulate_nominal(X, u, cfg, drag_coeffs);
metrics = compute_residuals(X, X_nom, cfg.ts, cfg.params);
print_error_summary(metrics);

% ----------------------------------------------------------------------
% [5] Save bundle
% ----------------------------------------------------------------------
fprintf('\n[5] Saving outputs ...\n');
outputFile = fullfile(scriptDir, 'Dual_cost_identification_with_nominal.mat');
save(outputFile, 'X', 'X_d', 'X_nom', 'X_nom_no_drag', 'u', 't', ...
    'metrics', 'drag_coeffs', '-v7.3');
fprintf('  ✓ Saved reconstruction bundle to: %s\n', outputFile);

% ----------------------------------------------------------------------
% [6] Plots
% ----------------------------------------------------------------------
fprintf('\n[6] Generating quick-look plots ...\n');
try
    generate_plots(scriptDir, t, X, X_nom_no_drag, X_nom, metrics);
catch ME
    warning('Plot generation skipped: %s', ME.message);
end

fprintf('\n======================================================================\n');
fprintf('SUMMARY\n');
fprintf('======================================================================\n');
fprintf('Experiments processed: %d\n', size(X, 1));
fprintf('Time steps per experiment: %d\n', size(X, 3));
fprintf('Sample time: %.4f s\n', ts);
fprintf('Output file: %s\n', outputFile);
fprintf('Generated plots: plot_linear_velocities.png, plot_angular_velocities.png,\n');
fprintf('                 plot_velocity_errors.png, plot_error_evolution.png, plot_modeling_residuals.png\n');
fprintf('Key insight: modelo abierto => cualquier desfase = dinámica faltante (drag, filtros, saturaciones).\n');
fprintf('======================================================================\n\n');

return

% ======================================================================
function ts = compute_sample_time(t)
if size(t, 2) < 2
    error('Time vector must contain at least two samples');
end
ts = mean(diff(t(1, :)));
if ~isfinite(ts) || ts <= 0
    error('Invalid sample time computed from t: %.6g', ts);
end
end

% ======================================================================
function X_sim = simulate_nominal(X, u, cfg, drag_coeffs)
[N_exp, ~, N_steps] = size(X);
X_sim = zeros(size(X));
if size(u, 3) == N_steps
    max_steps = N_steps - 1;
else
    max_steps = min(N_steps - 1, size(u, 3));
end

for expIdx = 1:N_exp
    state = X(expIdx, :, 1).';
    X_sim(expIdx, :, 1) = state;
    for k = 1:max_steps
        control = u(expIdx, :, k).';
        state = rk4_step(state, control, cfg, drag_coeffs);
        X_sim(expIdx, :, k + 1) = state;
    end
    if max_steps < N_steps - 1
        X_sim(expIdx, :, max_steps + 2:end) = repmat(state, 1, N_steps - max_steps - 1);
    end
end
end

% ======================================================================
function next_state = rk4_step(state, control, cfg, drag_coeffs)
state_next = state;
dt_local = cfg.ts / cfg.substeps;
for s = 1:cfg.substeps
    k1 = state_derivative(state_next, control, cfg.params, drag_coeffs);
    k2 = state_derivative(state_next + 0.5 * dt_local * k1, control, cfg.params, drag_coeffs);
    k3 = state_derivative(state_next + 0.5 * dt_local * k2, control, cfg.params, drag_coeffs);
    k4 = state_derivative(state_next + dt_local * k3, control, cfg.params, drag_coeffs);
    state_next = state_next + (dt_local / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    state_next(1:8) = normalize_dual_quaternion(state_next(1:8));
end
next_state = state_next;
end

% ======================================================================
function xdot = state_derivative(state, control, params, drag_coeffs)
quat = state(1:8);
omega = state(9:14);
xdot = zeros(14, 1);
xdot(1:8) = quatdot_simple(quat, omega);
xdot(9:14) = dual_acceleration(quat, omega, control, params);
xdot(12:14) = xdot(12:14) - (drag_coeffs(:) ./ params.m) .* omega(4:6);
end

% ======================================================================
function qdot = quatdot_simple(quat, omega)
qr = quat(1:4);
qd = quat(5:8);
K_quat = 10;
norm_r = norm(qr);
quat_error = 1 - norm_r;
aux_dual = [qr * (K_quat * quat_error); zeros(4, 1)];
H_r_plus = h_plus(qr);
H_d_plus = h_plus(qd);
Hplus = [H_r_plus, zeros(4); H_d_plus, H_r_plus];
omega_twist = [0; omega(1:3); 0; omega(4:6)];
qdot = 0.5 * (Hplus * omega_twist) + aux_dual;
end

% ======================================================================
function H = h_plus(q)
H = [ q(1), -q(2), -q(3), -q(4);
      q(2),  q(1), -q(4),  q(3);
      q(3),  q(4),  q(1), -q(2);
      q(4), -q(3),  q(2),  q(1) ];
end

% ======================================================================
function twistdot = dual_acceleration(quat, omega, control, params)
force = control(1);
torques = control(2:4);
J = params.J;
J_inv = diag(1 ./ diag(J));
e3 = [0; 0; 1];
m = params.m;
g = params.g;
w = omega(1:3);
v = omega(4:6);
q_real = quat(1:4);
F_r = -J_inv * cross(w, J * w);
F_d = cross(v, w) - g * rotate_inverse(q_real, e3);
U_r = J_inv * torques;
U_d = (force / m) * e3;
twistdot = [F_r + U_r; F_d + U_d];
end

% ======================================================================
function v_body = rotate_inverse(q, v)
q_conj = [q(1); -q(2:4)];
v_quat = [0; v(:)];
res = quat_multiply(q_conj, quat_multiply(v_quat, q));
v_body = res(2:4);
end

% ======================================================================
function prod = quat_multiply(a, b)
prod = [ a(1)*b(1) - dot(a(2:4), b(2:4));
         a(1)*b(2:4) + b(1)*a(2:4) + cross(a(2:4), b(2:4)) ];
end

% ======================================================================
function q_norm = normalize_dual_quaternion(q)
qr = q(1:4);
qd = q(5:8);
norm_real = norm(qr);
if norm_real < 1e-9
    q_norm = q;
    return;
end
qr = qr / norm_real;
qd = qd - (dot(qr, qd) * qr);
q_norm = [qr; qd];
end

% ======================================================================
function metrics = compute_residuals(X, X_nom, ts, params)
error = X - X_nom;
error_norm = squeeze(vecnorm(error, 2, 2));
error_vel = error(:, 12:14, :);
vel_lin_plant = X(:, 12:14, :);
vel_lin_model = X_nom(:, 12:14, :);
vel_ang_plant = X(:, 9:11, :);
vel_ang_model = X_nom(:, 9:11, :);
lin_acc_plant = gradient(vel_lin_plant, ts, 3);
lin_acc_model = gradient(vel_lin_model, ts, 3);
ang_acc_plant = gradient(vel_ang_plant, ts, 3);
ang_acc_model = gradient(vel_ang_model, ts, 3);
lin_acc_residual = lin_acc_plant - lin_acc_model;
ang_acc_residual = ang_acc_plant - ang_acc_model;
force_residual = params.m * lin_acc_residual;
torque_residual = zeros(size(ang_acc_residual));
J_diag = diag(params.J);
for axis = 1:3
    torque_residual(:, axis, :) = J_diag(axis) * ang_acc_residual(:, axis, :);
end
metrics = struct('error', error, 'error_norm', error_norm, 'error_vel', error_vel, ...
    'lin_acc_residual', lin_acc_residual, 'ang_acc_residual', ang_acc_residual, ...
    'force_residual', force_residual, 'torque_residual', torque_residual);
end

% ======================================================================
function drag = estimate_drag_coeffs(force_residual, X_nom)
vel_lin_model = X_nom(:, 12:14, :);
drag = zeros(1, 3);
for axis = 1:3
    v_axis = squeeze(vel_lin_model(:, axis, :));
    f_axis = squeeze(force_residual(:, axis, :));
    v_flat = v_axis(:);
    f_flat = f_axis(:);
    denom = dot(v_flat, v_flat) + 1e-9;
    drag(axis) = -dot(f_flat, v_flat) / denom;
end
end

% ======================================================================
function print_error_summary(metrics)
fprintf('\n[4] Reconstruction error summary\n');
fprintf('  ▸ Mean ||error||: %0.6e\n', mean(metrics.error_norm(:)));
fprintf('  ▸ Max  ||error||: %0.6e\n', max(metrics.error_norm(:)));
mean_vel = mean(abs(metrics.error_vel), 'all');
max_vel = max(abs(metrics.error_vel), [], 'all');
fprintf('  ▸ Linear velocity error stats (mean / max): %0.6e / %0.6e m/s\n', mean_vel, max_vel);
force_stats = squeeze(mean(mean(abs(metrics.force_residual), 3), 1));
torque_stats = squeeze(mean(mean(abs(metrics.torque_residual), 3), 1));
fprintf('  ▸ Force residual mean |·| per axis [N]: [%0.6f  %0.6f  %0.6f]\n', force_stats);
fprintf('  ▸ Torque residual mean |·| per axis [Nm]: [%0.6e  %0.6e  %0.6e]\n', torque_stats);
end

% ======================================================================
function generate_plots(scriptDir, t, X, X_nom_base, X_nom_drag, metrics)
exp_idx = 1;
t_plot = squeeze(t(exp_idx, :));
X_plot = squeeze(X(exp_idx, :, :)).';
X_base = squeeze(X_nom_base(exp_idx, :, :)).';
X_drag = squeeze(X_nom_drag(exp_idx, :, :)).';
err_plot = squeeze(metrics.error(exp_idx, :, :)).';
N_plot = size(X_drag, 1);
if numel(t_plot) > N_plot
    t_plot = t_plot(1:N_plot);
elseif numel(t_plot) < N_plot
    t_plot = linspace(t_plot(1), t_plot(end), N_plot);
end

figure('Visible', 'off');
titles = {'v_x [m/s]', 'v_y [m/s]', 'v_z [m/s]'};
for i = 1:3
    subplot(3,1,i);
    idx = 11 + i;
    plot(t_plot, X_plot(:, idx), 'b--', 'LineWidth', 1.2); hold on;
    plot(t_plot, X_base(:, idx), 'g-.', 'LineWidth', 1.1);
    plot(t_plot, X_drag(:, idx), 'r-', 'LineWidth', 1.4);
    grid on; ylabel(titles{i}); if i==1, title('Linear Velocities: Plant vs Model'); end
    if i==3, xlabel('Time [s]'); end; legend('Plant','Model (sin drag)','Model (con drag)');
end
saveas(gcf, fullfile(scriptDir, 'plot_linear_velocities.png'));

figure('Visible', 'off');
titles = {'\omega_x [rad/s]', '\omega_y [rad/s]', '\omega_z [rad/s]'};
for i = 1:3
    subplot(3,1,i);
    idx = 8 + i;
    plot(t_plot, X_plot(:, idx), 'b--', 'LineWidth', 1.2); hold on;
    plot(t_plot, X_base(:, idx), 'g-.', 'LineWidth', 1.1);
    plot(t_plot, X_drag(:, idx), 'r-', 'LineWidth', 1.4);
    grid on; ylabel(titles{i}); if i==1, title('Angular Velocities: Plant vs Model'); end
    if i==3, xlabel('Time [s]'); end; legend('Plant','Model (sin drag)','Model (con drag)');
end
saveas(gcf, fullfile(scriptDir, 'plot_angular_velocities.png'));

figure('Visible', 'off');
for i = 1:3
    subplot(3,1,i);
    idx = 11 + i;
    plot(t_plot, err_plot(:, idx), 'k-', 'LineWidth', 1.3); hold on;
    yline(0, 'r--'); grid on;
    ylabel(sprintf('e_{v%d} [m/s]', i));
    if i==1, title('Velocity Errors (Plant - Model)'); end
    if i==3, xlabel('Time [s]'); end
end
saveas(gcf, fullfile(scriptDir, 'plot_velocity_errors.png'));

figure('Visible', 'off');
v_err = vecnorm(err_plot(:, 12:14), 2, 2);
w_err = vecnorm(err_plot(:, 9:11), 2, 2);
subplot(2,1,1); plot(t_plot, v_err, 'b-', 'LineWidth', 1.4); grid on;
ylabel('||e_v|| [m/s]'); title('Error Norms');
subplot(2,1,2); plot(t_plot, w_err, 'r-', 'LineWidth', 1.4); grid on;
ylabel('||e_{\omega}|| [rad/s]'); xlabel('Time [s]');
saveas(gcf, fullfile(scriptDir, 'plot_error_evolution.png'));

figure('Visible', 'off');
force_plot = squeeze(metrics.force_residual(exp_idx, :, :));
torque_plot = squeeze(metrics.torque_residual(exp_idx, :, :));
subplot(2,1,1); plot(t_plot, force_plot'); grid on; ylabel('Force [N]'); title('Residual Forces'); legend('F_x','F_y','F_z');
subplot(2,1,2); plot(t_plot, torque_plot'); grid on; ylabel('Torque [Nm]'); xlabel('Time [s]'); legend('τ_x','τ_y','τ_z');
saveas(gcf, fullfile(scriptDir, 'plot_modeling_residuals.png'));

close all;
end
