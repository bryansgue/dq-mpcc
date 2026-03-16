#!/usr/bin/env python3
"""
DIAGNÓSTICO DE CUELLO DE BOTELLA - Loop MPCC
============================================
Mide el tiempo de cada fase del loop de control sin modificar MPC2.py.

Fases medidas:
  T1 - set_reference_column (interpolaciones Python en cada paso)
  T2 - set params en acados (loop j en N_prediction)
  T3 - acados_ocp_solver.solve() ← SQP-RTI
  T4 - acados_integrator.solve() ← IRK integración
  T5 - post-proceso (extracción de estado, rotaciones, logs)
  T6 - set_reference_column k+1
  T_total - tiempo total del paso
"""

import numpy as np
import time
import scipy.io
import casadi as ca
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import bisect
import math
import os, sys

# Añadir el directorio al path
sys.path.insert(0, os.path.dirname(__file__))

import rospy
rospy.init_node("timing_diagnosis", disable_signals=True, anonymous=True)

from functions import dualquat_from_pose_casadi
from ode_acados import (dualquat_trans_casadi, dualquat_quat_casadi,
                        rotation_casadi, rotation_inverse_casadi,
                        dual_velocity_casadi, dual_quat_casadi,
                        velocities_from_twist_casadi,
                        f_rk4_casadi_simple, noise,
                        create_casadi_trajectory_interpolator,
                        create_casadi_dual_quaternion_interpolator)
from nmpc_acados import create_ocp_solver
from acados_template import AcadosOcpSolver, AcadosSimSolver

# ── Funciones CasADi ──────────────────────────────────────────────────────────
dualquat_from_pose = dualquat_from_pose_casadi()
get_trans           = dualquat_trans_casadi()
get_quat            = dualquat_quat_casadi()
dual_twist          = dual_velocity_casadi()
velocity_from_twist = velocities_from_twist_casadi()

# ── Datos iniciales ──────────────────────────────────────────────────────────
file_path = os.path.join(os.path.dirname(__file__), "Separed_cost.mat")
Identification = scipy.io.loadmat(file_path)
x_0 = Identification['x_init']
x0  = np.concatenate([x_0[1, :], [0.0]])   # mismo k=1 que MPC2

# ── Parámetros del sistema ────────────────────────────────────────────────────
m, Jxx, Jyy, Jzz, g = 1, 2.64e-3, 2.64e-3, 4.96e-3, 9.8
L = [m, Jxx, Jyy, Jzz, g]

value = 10
t_final   = 30
frec      = 100
t_s       = 1/frec
sample_time = t_s

MIN_PROGRESS_SPEED = 0.2
MAX_PROGRESS_SPEED = 15.0
TARGET_PATH_LENGTH_M = 80

# ── Trayectoria ───────────────────────────────────────────────────────────────
def trayectoria():
    xd   = lambda t: 7*np.sin(value*0.04*t)+3
    yd   = lambda t: 7*np.sin(value*0.08*t)
    zd   = lambda t: 1.5*np.sin(value*0.08*t)+6
    xd_p = lambda t: 7*value*0.04*np.cos(value*0.04*t)
    yd_p = lambda t: 7*value*0.08*np.cos(value*0.08*t)
    zd_p = lambda t: 1.5*value*0.08*np.cos(value*0.08*t)
    return xd, yd, zd, xd_p, yd_p, zd_p

def euler_to_quaternion(roll, pitch, yaw):
    cy,sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp,sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr,sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]

def build_arc_length_splines(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):
    def integrand(t): return np.linalg.norm([xd_p(t), yd_p(t), zd_p(t)])
    def arc_length(tk):
        l, _ = quad(integrand, 0, tk, limit=100)
        return l

    arc_lengths, positions = [], []
    for tk in t_range:
        arc_lengths.append(arc_length(tk))
        positions.append([xd(tk), yd(tk), zd(tk)])
    arc_lengths = np.array(arc_lengths)
    positions   = np.array(positions).T

    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0])
    spline_y = CubicSpline(t_range, positions[1])
    spline_z = CubicSpline(t_range, positions[2])

    yaw_angles   = np.array([np.arctan2(yd_p(tk), xd_p(tk)) for tk in t_range])
    yaw_unwrapped = np.unwrap(yaw_angles)
    spline_yaw   = CubicSpline(t_range, yaw_unwrapped)

    def position_by_arc_length(s):
        te = spline_t(s)
        return np.array([spline_x(te), spline_y(te), spline_z(te)])

    def quaternion_by_arc_length(s):
        te  = spline_t(s)
        psi = spline_yaw(te)
        return np.array(euler_to_quaternion(0, 0, float(psi)))

    def velocity_by_arc_length(s):
        te = spline_t(s)
        return np.array([xd_p(float(te)), yd_p(float(te)), zd_p(float(te))])

    return arc_lengths, position_by_arc_length, quaternion_by_arc_length, velocity_by_arc_length

# ────────────────── SETUP ────────────────────────────────────────────────────
print("\n[TIMING] Construyendo trayectoria...")
xd, yd, zd, xd_p, yd_p, zd_p = trayectoria()
t = np.arange(0, t_final+t_s, t_s)
t_finer = np.linspace(0, t_final, len(t))

t0 = time.perf_counter()
arc_lengths, position_by_arc_length, quaternion_by_arc_length, velocity_by_arc_length = \
    build_arc_length_splines(xd, yd, zd, xd_p, yd_p, zd_p, t_finer, t_final)
print(f"  → Trayectoria construida en {time.perf_counter()-t0:.3f} s")

total_path_length  = arc_lengths[-1]
target_path_length = min(TARGET_PATH_LENGTH_M, total_path_length)
s_nominal_rate     = np.clip(target_path_length/t_final, MIN_PROGRESS_SPEED, MAX_PROGRESS_SPEED)
s_profile          = np.clip(np.arange(t.shape[0])*sample_time*s_nominal_rate, 0, total_path_length)

# CasADi waypoints
N_WAYPOINTS = 30
s_waypoints   = np.linspace(0, target_path_length, N_WAYPOINTS)
pos_waypoints  = np.zeros((3, N_WAYPOINTS))
vel_waypoints  = np.zeros((3, N_WAYPOINTS))
quat_waypoints = np.zeros((4, N_WAYPOINTS))

for i, sv in enumerate(s_waypoints):
    pos_waypoints[:, i]  = position_by_arc_length(sv)
    vel_waypoints[:, i]  = velocity_by_arc_length(sv)
    quat_waypoints[:, i] = quaternion_by_arc_length(sv)

for i in range(1, N_WAYPOINTS):
    if np.dot(quat_waypoints[:, i-1], quat_waypoints[:, i]) < 0:
        quat_waypoints[:, i] = -quat_waypoints[:, i]

for i in range(N_WAYPOINTS):
    vn = np.linalg.norm(vel_waypoints[:, i])
    if vn > 1e-6: vel_waypoints[:, i] /= vn

print("[TIMING] Creando funciones CasADi...")
t0 = time.perf_counter()
gamma_dq = create_casadi_dual_quaternion_interpolator(s_waypoints, pos_waypoints, quat_waypoints)
gamma_pos, gamma_vel, gamma_quat = create_casadi_trajectory_interpolator(
    s_waypoints, pos_waypoints, vel_waypoints, quat_waypoints)
print(f"  → CasADi interpolators: {time.perf_counter()-t0:.3f} s")

# Horizonte
t_N = 0.6
N_arr = np.arange(0, t_N+sample_time, sample_time)
N_prediction = N_arr.shape[0]

# Estado inicial
tx1,ty1,tz1 = x0[0],x0[1],x0[2]
qw1,qx1,qy1,qz1 = x0[3],x0[4],x0[5],x0[6]
dual_1 = dualquat_from_pose(qw1,qx1,qy1,qz1,tx1,ty1,tz1)
angular_linear_1 = np.array([x0[7],x0[8],x0[9],x0[10],x0[11],x0[12]])
dual_twist_1 = dual_twist(angular_linear_1, dual_1)
s_initial = 0.0
X_init = np.array(ca.vertcat(dual_1, dual_twist_1, s_initial)).reshape((15,))

F_max, F_min = L[0]*L[4]+20, 0
tau_max, tau_min = 0.1, -0.1
u_s_min, u_s_max = MIN_PROGRESS_SPEED, MAX_PROGRESS_SPEED

# Referencia nominal
u_d = np.zeros((5, t.shape[0]))

def set_reference_column_fn(X_d, idx, s_value):
    if idx >= X_d.shape[1]: return
    pos_val  = position_by_arc_length(s_value)
    quat_val = quaternion_by_arc_length(s_value)
    vel_val  = velocity_by_arc_length(s_value)
    dq_val   = dualquat_from_pose(quat_val[0],quat_val[1],quat_val[2],quat_val[3],
                                  pos_val[0],pos_val[1],pos_val[2])
    X_d[0:8, idx]  = np.array(dq_val).reshape((8,))
    X_d[8:14, idx] = np.array([0,0,0, vel_val[0],vel_val[1],vel_val[2]])
    X_d[14, idx]   = s_value

X_d = np.zeros((15, t.shape[0]+1))
for k in range(t.shape[0]):
    set_reference_column_fn(X_d, k, s_profile[k])

print("\n[TIMING] Construyendo solver ACADOS...")
t0 = time.perf_counter()
ocp = create_ocp_solver(X_init, N_prediction, t_N, F_max, F_min,
                        tau_max, tau_min, tau_max, tau_min, tau_max, tau_min,
                        L, sample_time, u_s_min=u_s_min, u_s_max=u_s_max,
                        gamma_dq=gamma_dq, gamma_vel=gamma_vel)
acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_"+ocp.model.name+".json",
                                    build=True, generate=True)
acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_"+ocp.model.name+".json",
                                    build=True, generate=True)
print(f"  → Solver construido en {time.perf_counter()-t0:.3f} s")

acados_ocp_solver.reset()
X_state = np.zeros((15, t.shape[0]+1))
X_state[:, 0] = X_init.copy()
for stage in range(N_prediction+1):
    acados_ocp_solver.set(stage, "x", X_init)
for stage in range(N_prediction):
    acados_ocp_solver.set(stage, "u", u_d[:, 0])

s_current = 0.0

# ── Acumuladores de timing ────────────────────────────────────────────────────
N_PROFILE_STEPS = 200   # Cuántos pasos medir
T1 = np.zeros(N_PROFILE_STEPS)   # set_reference_column (inicio del loop)
T2 = np.zeros(N_PROFILE_STEPS)   # set params acados (loop j)
T3 = np.zeros(N_PROFILE_STEPS)   # acados_ocp_solver.solve()
T3_lin = np.zeros(N_PROFILE_STEPS)   # time_lin dentro de solve
T3_sim = np.zeros(N_PROFILE_STEPS)   # time_sim dentro de solve
T3_qp  = np.zeros(N_PROFILE_STEPS)   # time_qp dentro de solve
T4 = np.zeros(N_PROFILE_STEPS)   # acados_integrator.solve()
T5 = np.zeros(N_PROFILE_STEPS)   # post-proceso (rotaciones, extracción)
T6 = np.zeros(N_PROFILE_STEPS)   # set_reference_column siguiente paso

print(f"\n[TIMING] Corriendo {N_PROFILE_STEPS} pasos de diagnóstico...\n")

for k in range(min(N_PROFILE_STEPS, t.shape[0]-N_prediction)):

    if s_current >= target_path_length:
        print(f"[TIMING] Trayectoria completada en paso k={k}")
        N_PROFILE_STEPS = k
        break

    # ── T1: set_reference_column al inicio ──────────────────────────────────
    t0 = time.perf_counter()
    set_reference_column_fn(X_d, k, s_current)
    T1[k] = time.perf_counter() - t0

    # ── T2: set params para cada nodo del horizonte ──────────────────────────
    t0 = time.perf_counter()
    acados_ocp_solver.set(0, "lbx", X_state[:, k])
    acados_ocp_solver.set(0, "ubx", X_state[:, k])

    for j in range(N_prediction):
        s_pred = s_current + j * sample_time * MAX_PROGRESS_SPEED
        s_pred = np.clip(s_pred, 0, target_path_length)
        pos_ref_j  = position_by_arc_length(s_pred)
        quat_ref_j = quaternion_by_arc_length(s_pred)
        vel_ref_j  = velocity_by_arc_length(s_pred)
        dq_ref_j   = dualquat_from_pose(quat_ref_j[0],quat_ref_j[1],quat_ref_j[2],quat_ref_j[3],
                                        pos_ref_j[0],pos_ref_j[1],pos_ref_j[2])
        vel_ref_nominal = np.array([0,0,0, vel_ref_j[0],vel_ref_j[1],vel_ref_j[2]])
        yref  = np.hstack([np.array(dq_ref_j).reshape(8), vel_ref_nominal])
        uref  = u_d[0:4, k+j]
        acados_ocp_solver.set(j, "p", np.hstack((yref, uref)))
    # Nodo terminal
    s_pred_N  = np.clip(s_current + N_prediction*sample_time*MAX_PROGRESS_SPEED, 0, target_path_length)
    pos_N     = position_by_arc_length(s_pred_N)
    quat_N    = quaternion_by_arc_length(s_pred_N)
    vel_N     = velocity_by_arc_length(s_pred_N)
    dq_N      = dualquat_from_pose(quat_N[0],quat_N[1],quat_N[2],quat_N[3],pos_N[0],pos_N[1],pos_N[2])
    yref_N    = np.hstack([np.array(dq_N).reshape(8), np.array([0,0,0,vel_N[0],vel_N[1],vel_N[2]])])
    acados_ocp_solver.set(N_prediction, "p", np.hstack((yref_N, u_d[0:4, k+N_prediction])))
    T2[k] = time.perf_counter() - t0

    # ── T3: solve OCP ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    acados_ocp_solver.solve()
    T3[k] = time.perf_counter() - t0
    # Subtiming interno de ACADOS
    T3_lin[k] = acados_ocp_solver.get_stats('time_lin')
    T3_sim[k] = acados_ocp_solver.get_stats('time_sim')
    T3_qp[k]  = acados_ocp_solver.get_stats('time_qp')

    # ── T4: integrador ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    aux_control = acados_ocp_solver.get(0, "u")
    u_full = np.hstack([aux_control[0], aux_control[1:4], aux_control[4]])
    acados_integrator.set("x", X_state[:, k])
    acados_integrator.set("u", u_full)
    acados_integrator.solve()
    xcurrent = acados_integrator.get("x")
    T4[k] = time.perf_counter() - t0

    # ── T5: post-proceso ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    u_s_optimal = aux_control[4]
    progress_rate = np.clip(u_s_optimal if np.isfinite(u_s_optimal) else s_nominal_rate,
                            u_s_min, u_s_max)
    s_current += progress_rate * sample_time
    s_current  = np.clip(s_current, 0, target_path_length)
    X_state[:, k+1]    = xcurrent.copy()
    X_state[14, k+1]   = s_current

    q_real   = X_state[0:4, k]
    v_body   = X_state[11:14, k]
    w,x,y,z  = q_real
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])
    v_inertial = R @ v_body
    tangent_dir = velocity_by_arc_length(s_current)
    tang_unit   = tangent_dir / (np.linalg.norm(tangent_dir)+1e-8)
    _ = np.dot(v_inertial, tang_unit)
    T5[k] = time.perf_counter() - t0

    # ── T6: set_reference_column siguiente ───────────────────────────────────
    t0 = time.perf_counter()
    set_reference_column_fn(X_d, k+1, s_current)
    T6[k] = time.perf_counter() - t0

valid = np.arange(N_PROFILE_STEPS)
N_v   = len(valid)

# ── REPORTE ───────────────────────────────────────────────────────────────────
print("=" * 68)
print("  DIAGNÓSTICO DE TIEMPO POR FASE DEL LOOP  (promedio sobre {} pasos)".format(N_v))
print("=" * 68)

phases = {
    "T1  set_reference start":        T1[valid],
    "T2  set params horizonte":        T2[valid],
    "T3  acados SOLVE (total)":        T3[valid],
    "  T3a  linearización (time_lin)": T3_lin[valid],
    "  T3b  simulación   (time_sim)":  T3_sim[valid],
    "  T3c  QP solver    (time_qp)":   T3_qp[valid],
    "T4  acados integrador":           T4[valid],
    "T5  post-proceso":                T5[valid],
    "T6  set_reference next":          T6[valid],
}

total_loop = (T1+T2+T3+T4+T5+T6)[valid]

for name, arr in phases.items():
    mean_ms  = np.mean(arr)*1e3
    max_ms   = np.max(arr)*1e3
    pct      = 100*np.mean(arr)/np.mean(total_loop) if not name.startswith("  ") else None
    pct_str  = f"  ({pct:5.1f}% del total)" if pct is not None else ""
    print(f"  {name:<36s}  avg={mean_ms:7.3f} ms   max={max_ms:7.3f} ms{pct_str}")

print("-" * 68)
mean_total = np.mean(total_loop)*1e3
max_total  = np.max(total_loop)*1e3
print(f"  {'TOTAL loop':<36s}  avg={mean_total:7.3f} ms   max={max_total:7.3f} ms")
print(f"  Frecuencia máxima alcanzable: {1000/mean_total:.1f} Hz  (loop real @ {frec} Hz)")
print(f"  Margen temporal por paso:     {(1000/frec - mean_total):.3f} ms")
print("=" * 68)

print("\n[TIMING] Distribución percentiles T3 (solve OCP) [ms]:")
p = np.percentile(T3[valid]*1e3, [50, 75, 90, 95, 99])
print(f"  P50={p[0]:.2f}  P75={p[1]:.2f}  P90={p[2]:.2f}  P95={p[3]:.2f}  P99={p[4]:.2f}")

print("\n[TIMING] Distribución percentiles T2 (set params) [ms]:")
p = np.percentile(T2[valid]*1e3, [50, 75, 90, 95, 99])
print(f"  P50={p[0]:.2f}  P75={p[1]:.2f}  P90={p[2]:.2f}  P95={p[3]:.2f}  P99={p[4]:.2f}")

print("\n[TIMING] Distribución percentiles T4 (integrador) [ms]:")
p = np.percentile(T4[valid]*1e3, [50, 75, 90, 95, 99])
print(f"  P50={p[0]:.2f}  P75={p[1]:.2f}  P90={p[2]:.2f}  P95={p[3]:.2f}  P99={p[4]:.2f}")

print("\n[TIMING] Nodos del horizonte N_prediction =", N_prediction)
print("[TIMING] Tiempo disponible por paso         = {:.2f} ms".format(1000/frec))
print("[TIMING] Overhead de T2 por nodo            = {:.4f} ms/nodo".format(np.mean(T2[valid])*1e3/N_prediction))
print("=" * 68)
