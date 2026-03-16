#!/usr/bin/env python
"""Reconstruct the nominal trajectory from identification data in a readable way."""

import os
import sys
import numpy as np
import scipy.io as sio
from scipy.io import savemat
import casadi as ca
from scipy.spatial.transform import Rotation as R

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# ==================== CONFIGURACIÓN ====================
# Parámetros físicos conocidos del sistema
KNOWN_MASS = 1.08
KNOWN_INERTIA = np.array([0.00454981, 0.00454981, 0.00281995], dtype=np.float64)
GRAVITY = 9.8

# Ventana temporal para la identificación (None para usar toda la trayectoria)
T_INIT = 0    # Tiempo inicial [s]
T_FINAL = 30  # Tiempo final [s]
# ======================================================


def banner(title):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def load_identification_data(mat_file):
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"File not found: {mat_file}")

    data = sio.loadmat(mat_file)
    dataset = {
        'X': np.asarray(data['X'], dtype=np.float64),
        'X_d': np.asarray(data['X_d'], dtype=np.float64),
        'u': np.asarray(data['u'], dtype=np.float64),
        't': np.asarray(data['t'], dtype=np.float64),
    }

    def squeeze_first_axis(key, expected_ndim):
        arr = dataset[key]
        if arr.ndim == expected_ndim + 1 and arr.shape[0] == 1:
            dataset[key] = arr[0]
        elif arr.ndim == expected_ndim:
            dataset[key] = arr
        else:
            raise ValueError(f"Solo se admite 1 experimento para '{key}'.")

    squeeze_first_axis('X', 2)
    squeeze_first_axis('X_d', 2)
    squeeze_first_axis('u', 2)
    squeeze_first_axis('t', 1)

    dataset['t'] = dataset['t'].flatten()

    print(f"  ▸ X shape: {dataset['X'].shape}")
    print(f"  ▸ X_d shape: {dataset['X_d'].shape}")
    print(f"  ▸ u shape: {dataset['u'].shape}")
    print(f"  ▸ t shape: {dataset['t'].shape}")
    return dataset


def compute_sample_time(t):
    if t.size < 2:
        raise ValueError("Time vector must contain al menos dos muestras")

    dt = np.diff(t.flatten())
    ts = float(np.mean(dt))
    if not np.isfinite(ts) or ts <= 0:
        raise ValueError(f"Invalid sample time computed from t: {ts}")
    return ts


def select_time_window(data, sample_time, t_init, t_final):
    t_vec = data['t']
    t_start = float(t_vec[0])
    t_end = float(t_vec[-1])

    if t_init is None and t_final is None:
        return data

    if t_init is None:
        t_init = t_start
    if t_final is None:
        t_final = t_end

    if t_init < t_start or t_final > t_end:
        raise ValueError(f"Ventana [{t_init}, {t_final}] fuera de rango ({t_start}, {t_end})")
    if t_init >= t_final:
        raise ValueError("t_init debe ser menor que t_final")

    def compute_slice(arr_len):
        idx_start = int(max(0, np.floor((t_init - t_start) / sample_time + 1e-9)))
        idx_end = int(min(arr_len, np.floor((t_final - t_start) / sample_time + 1e-9) + 1))
        if idx_end - idx_start < 2:
            raise ValueError("La ventana seleccionada es demasiado pequeña para integrar")
        return idx_start, idx_end

    X_start, X_end = compute_slice(data['X'].shape[1])
    data['X'] = data['X'][:, X_start:X_end]

    Xd_start, Xd_end = compute_slice(data['X_d'].shape[1])
    data['X_d'] = data['X_d'][:, Xd_start:Xd_end]

    t_start_idx = min(X_start, t_vec.size - 2)
    t_end_idx = min(t_vec.size, t_start_idx + (X_end - X_start))
    data['t'] = t_vec[t_start_idx:t_end_idx] - t_vec[t_start_idx]

    u_len_target = data['X'].shape[1] - 1
    u_start = min(X_start, max(0, data['u'].shape[1] - u_len_target))
    u_end = min(data['u'].shape[1], u_start + u_len_target)
    data['u'] = data['u'][:, u_start:u_end]

    print(f"  ▸ Ventana aplicada: t_init={t_init:.3f} s, t_final={t_final:.3f} s")
    print(f"  ▸ Muestras seleccionadas: {data['X'].shape[1]} estados, {data['u'].shape[1]} entradas")
    return data


def gravity_world_to_body(quat_seq):
    """Compute gravity vector in body frame given quaternion sequence."""
    quat_stack = np.vstack((quat_seq[1], quat_seq[2], quat_seq[3], quat_seq[0])).T
    rotations = R.from_quat(quat_stack)
    e3 = np.array([0.0, 0.0, 1.0])
    g_body = rotations.apply(e3, inverse=True)
    return g_body.T


def solve_inertia_params(omega, omega_dot, torque):
    N = omega.shape[1]
    rows = []
    rhs = []
    for k in range(N):
        wx, wy, wz = omega[:, k]
        dwx, dwy, dwz = omega_dot[:, k]
        taux, tauy, tauz = torque[:, k]
        rows.append([dwx, wy * wz, -wy * wz])
        rhs.append(taux)
        rows.append([-wx * wz, dwy, wx * wz])
        rhs.append(tauy)
        rows.append([wx * wy, -wx * wy, dwz])
        rhs.append(tauz)
    A = np.asarray(rows, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def direct_identification(data, sample_time, gravity, mass_value, inertia_diag):
    X = data['X']
    U = data['u']
    quat = X[0:4, :]
    omega = X[8:11, :]
    vel = X[11:14, :]
    torque = U[1:4, :]
    force = U[0, :]

    N = min(force.shape[0], vel.shape[1], omega.shape[1])
    quat = quat[:, :N]
    omega = omega[:, :N]
    vel = vel[:, :N]
    torque = torque[:, :N]
    force = force[:N]

    lin_acc = np.gradient(vel, sample_time, axis=1)
    ang_acc = np.gradient(omega, sample_time, axis=1)
    grav_body = gravity * gravity_world_to_body(quat)
    force_body = np.zeros_like(vel)
    force_body[2, :] = force

    cross_term = np.cross(vel.T, omega.T).T
    phi = lin_acc - cross_term + grav_body
    force_needed = mass_value * phi
    drag_vec = np.zeros(3)
    for axis in range(3):
        v_axis = vel[axis, :]
        residual_axis = force_body[axis, :] - force_needed[axis, :]
        mask = np.abs(v_axis) > 1e-6
        if np.count_nonzero(mask) < 10:
            drag_vec[axis] = 0.0
            continue
        drag_vec[axis] = np.dot(residual_axis[mask], v_axis[mask]) / np.dot(v_axis[mask], v_axis[mask])

    drag_mat = drag_vec.reshape(3, 1)
    lin_acc_model_no_drag = cross_term - grav_body + force_body / mass_value
    lin_acc_model = lin_acc_model_no_drag - (drag_mat * vel) / mass_value
    force_residual_no_drag = force_body - (mass_value * phi)
    force_residual = force_residual_no_drag - drag_mat * vel

    J_diag = inertia_diag.reshape(3,)
    Jw = J_diag.reshape(3, 1) * omega
    w_cross_Jw = np.cross(omega.T, Jw.T).T
    ang_acc_model = (torque - w_cross_Jw) / J_diag.reshape(3, 1)
    torque_residual = torque - (J_diag.reshape(3, 1) * ang_acc + w_cross_Jw)

    return {
        "mass": mass_value,
        "drag": drag_vec,
        "inertia": J_diag,
        "lin_acc": lin_acc,
        "lin_acc_model": lin_acc_model,
        "lin_acc_model_no_drag": lin_acc_model_no_drag,
        "ang_acc": ang_acc,
        "ang_acc_model": ang_acc_model,
        "force_residual": force_residual,
        "force_residual_no_drag": force_residual_no_drag,
        "torque_residual": torque_residual,
        "force_body": force_body,
        "torque_cmd": torque,
        "vel": vel,
        "omega": omega,
    }
def save_direct_results(output_file, data, direct_metrics):
    payload = {
        'X': np.expand_dims(data['X'], axis=0),
        'u': np.expand_dims(data['u'], axis=0),
        't': np.expand_dims(data['t'], axis=0),
        'lin_acc': np.expand_dims(direct_metrics['lin_acc'], axis=0),
        'lin_acc_model': np.expand_dims(direct_metrics['lin_acc_model'], axis=0),
    'lin_acc_model_no_drag': np.expand_dims(direct_metrics['lin_acc_model_no_drag'], axis=0),
        'ang_acc': np.expand_dims(direct_metrics['ang_acc'], axis=0),
        'ang_acc_model': np.expand_dims(direct_metrics['ang_acc_model'], axis=0),
        'force_residual': np.expand_dims(direct_metrics['force_residual'], axis=0),
    'force_residual_no_drag': np.expand_dims(direct_metrics['force_residual_no_drag'], axis=0),
        'torque_residual': np.expand_dims(direct_metrics['torque_residual'], axis=0),
        'mass_value': np.array([[direct_metrics['mass']]]),
        'drag_est': np.expand_dims(direct_metrics['drag'], axis=0),
        'inertia_est': np.expand_dims(direct_metrics['inertia'], axis=0),
    }
    savemat(output_file, payload)
    print(f"✓ Saved identification bundle to: {output_file}")


def generate_direct_plots(output_dir, t, direct_metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save_current(fig, name):
        path = os.path.join(output_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {name}")

    lin_acc = direct_metrics['lin_acc'].T
    lin_acc_model = direct_metrics['lin_acc_model'].T
    lin_acc_model_no_drag = direct_metrics['lin_acc_model_no_drag'].T
    ang_acc = direct_metrics['ang_acc'].T
    ang_acc_model = direct_metrics['ang_acc_model'].T
    force_residual = direct_metrics['force_residual']
    force_residual_no_drag = direct_metrics['force_residual_no_drag']
    torque_residual = direct_metrics['torque_residual']

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle('Linear Accelerations: Plant vs Modelo (sin / con drag)', fontsize=14, fontweight='bold')
    labels = ['a_x [m/s²]', 'a_y [m/s²]', 'a_z [m/s²]']
    for i in range(3):
        axes[i].plot(t, lin_acc[:, i], 'k-', label='Planta')
        axes[i].plot(t, lin_acc_model_no_drag[:, i], 'g-.', label='Modelo sin drag')
        axes[i].plot(t, lin_acc_model[:, i], 'r--', label='Modelo con drag')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='best')
    axes[-1].set_xlabel('Time [s]')
    save_current(fig, 'direct_linear_accelerations.png')

    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    fig.suptitle('Angular Accelerations: Data vs Modelo algebraico', fontsize=14, fontweight='bold')
    labels = ['α_x [rad/s²]', 'α_y [rad/s²]', 'α_z [rad/s²]']
    for i in range(3):
        axes[i].plot(t, ang_acc[:, i], 'k-', label='Medido')
        axes[i].plot(t, ang_acc_model[:, i], 'r--', label='Modelo identificado')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='best')
    axes[-1].set_xlabel('Time [s]')
    save_current(fig, 'direct_angular_accelerations.png')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Residual Forces and Torques (Direct ID)', fontsize=14, fontweight='bold')
    for axis, label in enumerate(['F_x [N]', 'F_y [N]', 'F_z [N]']):
        axes[0].plot(t, force_residual_no_drag[axis], label=f'{label} (sin drag)', linestyle='--')
        axes[0].plot(t, force_residual[axis], label=f'{label} (con drag)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')
    axes[0].set_ylabel('Force residual [N]')
    for axis, label in enumerate(['τ_x [Nm]', 'τ_y [Nm]', 'τ_z [Nm]']):
        axes[1].plot(t, torque_residual[axis], label=label)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')
    axes[1].set_ylabel('Torque residual [Nm]')
    axes[1].set_xlabel('Time [s]')
    save_current(fig, 'direct_modeling_residuals.png')



def main():
    banner("DIRECT IDENTIFICATION OF DRAG FROM ACCELERATION DATA")
    print("[1] Loading identification data ...")
    mat_file = os.path.join(script_dir, "Dual_cost_identification.mat")
    data = load_identification_data(mat_file)

    sample_time_full = compute_sample_time(data['t'])
    
    t_min = float(data['t'][0])
    t_max = float(data['t'][-1])
    
    print(f"\n[*] Time range available: [{t_min:.2f}, {t_max:.2f}] s")
    print(f"[*] Window configured: T_INIT={T_INIT}, T_FINAL={T_FINAL}")
    
    t_init = T_INIT if T_INIT is not None else t_min
    t_final = T_FINAL if T_FINAL is not None else t_max
    
    data = select_time_window(data, sample_time_full, t_init, t_final)

    print("\n[2] Direct identification from measured accelerations ...")
    sample_time = compute_sample_time(data['t'])
    print(f"  ▸ Sample time: {sample_time:.6f} s")

    direct_metrics = direct_identification(
        data,
        sample_time,
        GRAVITY,
        KNOWN_MASS,
        KNOWN_INERTIA,
    )
    print(f"  ▸ Identified drag [N·s/m]: {direct_metrics['drag']}")

    print("\n[3] Saving results ...")
    output_file = os.path.join(script_dir, "direct_identification_results.mat")
    save_direct_results(output_file, data, direct_metrics)

    print("\n[4] Generating plots ...")
    t_plot = data['t'][:direct_metrics['lin_acc'].shape[1]]
    generate_direct_plots(script_dir, t_plot, direct_metrics)

    banner("SUMMARY")
    print("Mode: Direct identification (no integration)")
    print(f"Time steps: {direct_metrics['lin_acc'].shape[1]}")
    print(f"Identified drag [N·s/m]: {direct_metrics['drag']}")
    print(f"Output file: {output_file}")
    print("Generated plots:")
    print("  · direct_linear_accelerations.png")
    print("  · direct_angular_accelerations.png")
    print("  · direct_modeling_residuals.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n✗ Reconstruction failed: {exc}")
        sys.exit(1)
